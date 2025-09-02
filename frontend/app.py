import os, requests
import gradio as gr
from dotenv import load_dotenv

# 환경 변수 로드 (backend/.env에는 Vercel 배포용 환경이 포함되어 있어야 합니다)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"), override=True)

# Vercel 배포된 백엔드 URL (GitHub Space Secrets에 BACKEND_URL 설정)
BACKEND = os.getenv('BACKEND_URL', 'https://your-vercel-backend.vercel.app')

def fetch_data_fn():
    try:
        res = requests.get(f"{BACKEND}/fetch_data", timeout=60)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        return {"error": str(e)}

def run_recommendations(yt, sns, mbti, use_openai):
    try:
        payload = {
            "youtube_subscriptions": [s.strip() for s in yt.splitlines() if s.strip()],
            "sns_keywords": [s.strip() for s in sns.splitlines() if s.strip()],
            "mbti": mbti
        }
        res = requests.post(f"{BACKEND}/youtube/recommendations", json=payload, timeout=120)
        res.raise_for_status()
        data = res.json().get("recommendations", {})
    except Exception as e:
        data = {"youtube": [{"name": "추천 실패", "url": str(e)}], "web": []}

    rows = []
    for c in data.get("youtube", []) + data.get("web", []):
        rows.append([c.get("name", ""), c.get("url", "")])
    if not rows:
        rows = [["추천 실패", ""]]
    return rows

with gr.Blocks(title='PersonaMate Pro — OAuth 수집 + 추천 UI') as demo:
    gr.Markdown('## PersonaMate Pro — OAuth 수집 + 추천 UI')
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('### 1) OAuth 로그인')
            gr.HTML(f'<a href="{BACKEND}/oauth/google/start" target="_blank">Google (YouTube) 로그인</a>')
            gr.HTML(f'<a href="{BACKEND}/oauth/instagram/start" target="_blank">Instagram 로그인</a>')
            gr.HTML(f'<a href="{BACKEND}/oauth/x/start" target="_blank">X 로그인</a>')
        with gr.Column(scale=2):
            gr.Markdown('### 2) 자동 수집')
            yt_chk = gr.Checkbox(label='YouTube 구독 목록 사용', value=True)
            ig_chk = gr.Checkbox(label='Instagram 해시태그 사용', value=False)
            x_chk = gr.Checkbox(label='X 팔로잉 리스트 사용', value=False)
            fetch_btn = gr.Button('데이터 수집')
            fetch_result = gr.JSON(label="수집된 데이터 미리보기")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('### 3) 입력/MBTI')
            yt_text = gr.Textbox(lines=6, label='유튜브 구독 (수집/수동)')
            sns_text = gr.Textbox(lines=6, label='SNS 키워드/계정')
            mbti = gr.Dropdown(
                choices=['ISTJ','ISFJ','INFJ','INTJ','ISTP','ISFP','INFP','INTP','ESTP','ESFP','ENFP','ENTP','ESTJ','ESFJ','ENFJ','ENTJ'],
                value='ENFP',
                label='MBTI'
            )
            use_openai = gr.Checkbox(label='OpenAI 임베딩 사용', value=True)
            run_btn = gr.Button('추천 실행', variant='primary')
        with gr.Column(scale=3):
            gr.Markdown('### 4) 추천 결과')
            result_table = gr.Dataframe(headers=["채널 이름", "사이트 주소"], row_count=10, col_count=2)

    fetch_btn.click(fetch_data_fn, inputs=[], outputs=[fetch_result])
    run_btn.click(run_recommendations, [yt_text, sns_text, mbti, use_openai], [result_table])

if __name__ == '__main__':
    demo.launch()
