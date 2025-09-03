import os, time, json, httpx, secrets, base64, hashlib, psycopg2
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from urllib.parse import urlencode
from itsdangerous import URLSafeSerializer
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from apscheduler.schedulers.background import BackgroundScheduler
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

load_dotenv()
USE_SQLITE = os.getenv("USE_SQLITE", "true").lower() == "true"

conn = None
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS tokens (provider TEXT PRIMARY KEY, data TEXT)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS recommendations (
                id SERIAL PRIMARY KEY,
                source TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        print("Database connection and table creation successful.")
    except Exception as e:
        print(f"Database connection or table creation failed: {e}")

def save_token(provider: str, data: dict):
    print(f"Attempting to save token for provider: {provider}")
    if not conn:
        print("Database connection not established. Cannot save token.")
        return
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO tokens (provider, data) VALUES (%s, %s) ON CONFLICT (provider) DO UPDATE SET data = EXCLUDED.data", (provider, json.dumps(data)))
        conn.commit()
        print(f"Token for {provider} saved successfully.")
    except Exception as e:
        print(f"Error saving token for {provider}: {e}")

def load_token(provider: str):
    print(f"Attempting to load token for provider: {provider}")
    if not conn:
        print("Database connection not established. Cannot load token.")
        return None
    try:
        cur = conn.cursor()
        cur.execute("SELECT data FROM tokens WHERE provider=%s", (provider,))
        row = cur.fetchone()
        if not row:
            print(f"No token found for {provider}.")
            return None
        token_data = json.loads(row[0])
        print(f"Token for {provider} loaded successfully.")
        return token_data
    except Exception as e:
        print(f"Error loading token for {provider}: {e}")
        return None

def save_recommendations(source: str, data: dict):
    print(f"Attempting to save recommendations for source: {source}")
    if not conn:
        print("Database connection not established. Cannot save recommendations.")
        return
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO recommendations (source, data) VALUES (%s, %s)", (source, json.dumps(data, ensure_ascii=False)))
        conn.commit()
        print(f"Recommendations for {source} saved successfully.")
    except Exception as e:
        print(f"Error saving recommendations for {source}: {e}")

SECRET_KEY = os.getenv("SECRET_KEY","changeme")
SIGNER = URLSafeSerializer(SECRET_KEY)
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN","http://localhost:7860")
app = FastAPI() # Removed root_path here
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'], allow_credentials=True)

class RecommendationRequest(BaseModel):
    youtube_subscriptions: List[str] = []
    sns_keywords: List[str] = []
    mbti: Optional[str] = None


@app.get("/oauth/google/start")
async def oauth_google_start():
    client_id = os.getenv("GOOGLE_CLIENT_ID","")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI","http://localhost:9000/oauth/google/callback")
    scope = "https://www.googleapis.com/auth/youtube.readonly"
    auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={client_id}"
        f"&redirect_uri={redirect_uri}"
        f"&response_type=code"
        f"&scope={scope}"
        f"&access_type=offline"
        f"&prompt=consent"
    )
    print(f"Redirecting to Google OAuth: {auth_url}")
    return RedirectResponse(auth_url)


@app.get("/oauth/google/callback")
async def oauth_google_callback(code: str):
    print(f"Google OAuth callback received with code: {code}")
    client_id = os.getenv("GOOGLE_CLIENT_ID","")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET","")
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI","http://localhost:9000/oauth/google/callback")
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code"
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(token_url, data=data)
        resp.raise_for_status()
        token_data = resp.json()
        save_token("google", token_data)
        print(f"Google OAuth token data: {token_data}")
        return {"status":"ok","token":token_data}


@app.get("/oauth/instagram/start")
async def oauth_instagram_start():
    return RedirectResponse("https://www.instagram.com/oauth/authorize")


@app.get("/fetch_data")
async def fetch_data():
    print("Fetching data endpoint called.")
    token = load_token("google")
    if not token:
        print("Google OAuth token not found in DB.")
        raise HTTPException(401,"Google OAuth token not found")
    creds = token.get("access_token")
    if not creds:
        print("Access token missing from loaded token data.")
        raise HTTPException(401,"Access token missing")

    try:
        subs = []
        page_token = None
        async with httpx.AsyncClient() as client:
            while True:
                params = {
                    "part": "snippet",
                    "mine": "true",
                    "maxResults": 50
                }
                if page_token:
                    params["pageToken"] = page_token
                resp = await client.get(
                    "https://www.googleapis.com/youtube/v3/subscriptions",
                    params=params,
                    headers={"Authorization": f"Bearer {creds}"}
                )
                resp.raise_for_status()
                data = resp.json()
                subs.extend([item["snippet"]["title"] for item in data.get("items", [])])
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
        print(f"Successfully fetched {len(subs)} YouTube subscriptions.")
    except Exception as e:
        print(f"YouTube API error during fetch_data: {e}")
        raise HTTPException(500, f"YouTube API error: {e}")

    return {
        "youtube_subscriptions": subs,
        "sns_keywords": [],
        "mbti": "ENFP"
    }


@app.post("/youtube/recommendations")
async def youtube_recommendations(request: RecommendationRequest):
    print("YouTube recommendations endpoint called.")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY is not set.")
        raise HTTPException(500, "GEMINI_API_KEY is not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    # 프롬프트에 구독 채널과 MBTI를 반영
    prompt = f"""
    사용자의 구독 채널: {', '.join(request.youtube_subscriptions)}
    사용자의 MBTI: {request.mbti}

    위 정보를 바탕으로 사용자의 성향과 관심사에 맞는 새로운 유튜브 채널 10개를 추천해 주세요.
    - 반드시 사용자의 구독 채널과 MBTI를 분석하여 새로운 채널을 제안해야 합니다.
    - 한국 채널을 최소 3개 이상 포함해야 합니다.
    - 추천 결과는 반드시 JSON 형식으로만 반환해야 합니다.
    - 각 추천 항목은 채널 이름(name)과 채널 주소(url)만 포함해야 합니다.

    {{
      "recommendations": {{
        "youtube": [
          {{"name": "채널 이름", "url": "https://youtube.com/..."}}
        ]
      }}
    }}
    """
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "topK": 40,
            "topP": 0.8,
            "maxOutputTokens": 2048
        }
    }

    async with httpx.AsyncClient(timeout=600) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        raw = response.json()

        text = raw.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","")
        if not text:
            print("No text content received from Gemini API.")
            return {"recommendations":{"youtube":[]}}

        try:
            # JSON 블록만 정규식으로 추출
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            cleaned = match.group(0) if match else text
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").replace("json","",1).strip()
            result = json.loads(cleaned)
            print("Gemini API response parsed successfully.")
        except Exception as e:
            print(f"JSON parsing failed: {e}, Original text: {text}")
            # fallback: 구독 채널과 MBTI 기반 기본 추천 생성 (10개 유튜브)
            youtube_list = (request.youtube_subscriptions or ["기본채널"]) * 10
            youtube_fallback = [
                {"name": f"{ch} 추천 채널 {i+1}", "url": f"http://youtube.com/{i+1}"} 
                for i, ch in list(enumerate(youtube_list))[:10]
            ]
            result = {
                "recommendations":{
                    "youtube": youtube_fallback
                },
                "error":f"JSON 파싱 실패: {e}, 원본: {text}"
            }

        return result
