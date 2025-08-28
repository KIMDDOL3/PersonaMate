import os, time, json, httpx, secrets, base64, hashlib, sqlite3
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlencode
from itsdangerous import URLSafeSerializer
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.path.join(os.path.dirname(__file__), "tokens.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("CREATE TABLE IF NOT EXISTS tokens (provider TEXT PRIMARY KEY, data TEXT)")
conn.commit()
def save_token(provider: str, data: dict):
    conn.execute("REPLACE INTO tokens (provider, data) VALUES (?,?)", (provider, json.dumps(data)))
    conn.commit()
def load_token(provider: str):
    row = conn.execute("SELECT data FROM tokens WHERE provider=?",(provider,)).fetchone()
    if not row: return None
    return json.loads(row[0])

SECRET_KEY = os.getenv("SECRET_KEY","changeme")
SIGNER = URLSafeSerializer(SECRET_KEY)
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN","http://localhost:7860")
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'], allow_credentials=True)
STORE = {}
def set_state(data: dict) -> str:
    state = secrets.token_urlsafe(16); STORE[state] = {'data':data,'ts':time.time()}; return state
def pop_state(state: str):
    return STORE.pop(state, None)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID","")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET","")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI","http://localhost:8000/oauth/google/callback")
GOOGLE_SCOPE = os.getenv("GOOGLE_SCOPE","https://www.googleapis.com/auth/youtube.readonly")
@app.get('/oauth/google/start')
def google_start():
    state = set_state({'provider':'google'})
    params = {'client_id':GOOGLE_CLIENT_ID,'redirect_uri':GOOGLE_REDIRECT_URI,'response_type':'code','scope':GOOGLE_SCOPE,'access_type':'offline','include_granted_scopes':'true','state':state,'prompt':'consent'}
    return RedirectResponse('https://accounts.google.com/o/oauth2/v2/auth?' + urlencode(params))
@app.get('/oauth/google/callback')
async def google_callback(code: Optional[str]=None, state: Optional[str]=None, error: Optional[str]=None):
    if error: raise HTTPException(400, error)
    if not code or not state: raise HTTPException(400, 'missing code/state')
    if not pop_state(state): raise HTTPException(400, 'invalid state')
    async with httpx.AsyncClient(timeout=30) as client:
        token_res = await client.post('https://oauth2.googleapis.com/token', data={'client_id':GOOGLE_CLIENT_ID,'client_secret':GOOGLE_CLIENT_SECRET,'code':code,'grant_type':'authorization_code','redirect_uri':GOOGLE_REDIRECT_URI})
    tok = token_res.json(); save_token('google', tok)
    payload = SIGNER.dumps({'provider':'google','ok':True})
    return RedirectResponse(f"{FRONTEND_ORIGIN}/?google={payload}")
@app.get('/youtube/subscriptions')
async def youtube_subscriptions():
    tok = load_token('google')
    if not tok: raise HTTPException(400, 'no google token')
    access = tok.get('access_token'); 
    if not access: raise HTTPException(401, 'no access_token')
    items=[]; url='https://www.googleapis.com/youtube/v3/subscriptions'; params={'part':'snippet','mine':'true','maxResults':50}; headers={'Authorization':f'Bearer {access}'}
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            r = await client.get(url, params=params, headers=headers); j = r.json()
            for it in j.get('items', []):
                title = it['snippet']['title']; churl = 'https://www.youtube.com/channel/' + it['snippet']['resourceId']['channelId']
                items.append({'name':title,'url':churl})
            token = j.get('nextPageToken'); 
            if not token: break
            params['pageToken'] = token
    return {'subscriptions': items}

FB_APP_ID = os.getenv('FB_APP_ID','')
FB_APP_SECRET = os.getenv('FB_APP_SECRET','')
IG_REDIRECT_URI = os.getenv('IG_REDIRECT_URI','http://localhost:8000/oauth/instagram/callback')
IG_SCOPE = os.getenv('IG_SCOPE','instagram_basic')
@app.get('/oauth/instagram/start')
def ig_start():
    state = set_state({'provider':'instagram'})
    params = {'client_id':FB_APP_ID,'redirect_uri':IG_REDIRECT_URI,'response_type':'code','scope':IG_SCOPE,'state':state}
    return RedirectResponse('https://www.facebook.com/v18.0/dialog/oauth?' + urlencode(params))
@app.get('/oauth/instagram/callback')
async def ig_callback(code: Optional[str]=None, state: Optional[str]=None, error: Optional[str]=None):
    if error: raise HTTPException(400, error)
    if not code or not state: raise HTTPException(400, 'missing code/state')
    if not pop_state(state): raise HTTPException(400, 'invalid state')
    async with httpx.AsyncClient(timeout=30) as client:
        token_res = await client.get('https://graph.facebook.com/v18.0/oauth/access_token', params={'client_id':FB_APP_ID,'client_secret':FB_APP_SECRET,'redirect_uri':IG_REDIRECT_URI,'code':code})
    tok = token_res.json(); save_token('instagram', tok)
    payload = SIGNER.dumps({'provider':'instagram','ok':True})
    return RedirectResponse(f"{FRONTEND_ORIGIN}/?instagram={payload}")
@app.get('/instagram/media')
async def ig_media():
    tok = load_token('instagram')
    if not tok: raise HTTPException(400, 'no instagram token')
    access = tok.get('access_token')
    fields='id,caption,media_type,media_url,thumbnail_url,permalink,timestamp'
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get('https://graph.instagram.com/me/media', params={'fields':fields,'access_token':access})
    return r.json()

X_CLIENT_ID = os.getenv('X_CLIENT_ID','')
X_CLIENT_SECRET = os.getenv('X_CLIENT_SECRET','')
X_REDIRECT_URI = os.getenv('X_REDIRECT_URI','http://localhost:8000/oauth/x/callback')
X_SCOPE = os.getenv('X_SCOPE','tweet.read users.read follows.read like.read offline.access')
def make_challenge():
    import base64, hashlib, secrets
    cv = secrets.token_urlsafe(64)
    cc = base64.urlsafe_b64encode(hashlib.sha256(cv.encode()).digest()).rstrip(b'=').decode()
    return cv, cc
@app.get('/oauth/x/start')
def x_start():
    cv, cc = make_challenge()
    state = set_state({'provider':'x','code_verifier':cv})
    params = {'response_type':'code','client_id':X_CLIENT_ID,'redirect_uri':X_REDIRECT_URI,'scope':X_SCOPE,'state':state,'code_challenge':cc,'code_challenge_method':'S256'}
    return RedirectResponse('https://twitter.com/i/oauth2/authorize?' + urlencode(params))
@app.get('/oauth/x/callback')
async def x_callback(code: Optional[str]=None, state: Optional[str]=None, error: Optional[str]=None):
    if error: raise HTTPException(400, error)
    if not code or not state: raise HTTPException(400, 'missing code/state')
    st = pop_state(state)
    if not st: raise HTTPException(400, 'invalid state')
    cv = st['data']['code_verifier']
    async with httpx.AsyncClient(timeout=30) as client:
        token_res = await client.post('https://api.twitter.com/2/oauth2/token', data={'grant_type':'authorization_code','code':code,'redirect_uri':X_REDIRECT_URI,'code_verifier':cv,'client_id':X_CLIENT_ID}, headers={'Content-Type':'application/x-www-form-urlencoded'})
    tok = token_res.json(); save_token('x', tok)
    payload = SIGNER.dumps({'provider':'x','ok':True})
    return RedirectResponse(f"{FRONTEND_ORIGIN}/?x={payload}")
@app.get('/x/me_following')
async def x_following():
    tok = load_token('x')
    if not tok: raise HTTPException(400, 'no x token')
    access = tok.get('access_token'); headers={'Authorization':f'Bearer {access}'}
    async with httpx.AsyncClient(timeout=30) as client:
        me = (await client.get('https://api.twitter.com/2/users/me', headers=headers)).json()
        uid = (me.get('data') or {}).get('id')
        if not uid: return {'data':[]}
        fol = (await client.get(f'https://api.twitter.com/2/users/{uid}/following', headers=headers, params={'max_results':200})).json()
    return fol
@app.get('/health')
def health(): return {'ok': True}
