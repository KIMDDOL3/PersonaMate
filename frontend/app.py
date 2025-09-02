import os, json, requests
import gradio as gr
from dotenv import load_dotenv

from fastapi import FastAPI
# 환경 변수 로드
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"), override=True)

BACKEND = os.getenv('BACKEND_URL','https://your-vercel-backend.vercel.app')

with gr.Blocks(title='PersonaMate Pro (OAuth + Simplified UI)') as demo:
    gr.Markdown('## PersonaMate Pro — OAuth 수집 + 추천 UI')
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('### 1) OAuth 로그인')
            google_html = gr.HTML(f'<a href="{BACKEND}/oauth/google/start" target="_blank">Google (YouTube) 로그인 열기</a>')
            instagram_html = gr.HTML(f'<a href="{BACKEND}/oauth/instagram/start" target="_blank">Instagram 로그인 열기</a>')
            x_html = gr.HTML(f'<a href="{BACKEND}/oauth/x/start" target="_blank">X (Twitter) 로그인 열기</a>')
        with gr.Column(scale=2):
            gr.Markdown('### 2) 자동 수집')
            yt_chk=gr.Checkbox(label='YouTube 구독 목록 사용', value=True)
            ig_chk=gr.Checkbox(label='Instagram 해시태그 사용', value=False)
            x_chk=gr.Checkbox(label='X 팔로잉 사용자명 사용', value=False)
            fetch_btn=gr.Button('내 계정에서 데이터 수집')
            fetch_result=gr.JSON(label="수집된 데이터 미리보기")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('### 3) 입력/MBTI')
            yt_text=gr.Textbox(lines=6,label='유튜브 구독 (수집/수동 혼용)')
            sns_text=gr.Textbox(lines=6,label='SNS 키워드/계정 (수집/수동 혼용)')
            mbti=gr.Dropdown(choices=['ISTJ','ISFJ','INFJ','INTJ','ISTP','ISFP','INFP','INTP','ESTP','ESFP','ENFP','ENTP','ESTJ','ESFJ','ENFJ','ENTJ'], value='ENFP', label='MBTI')
            use_openai=gr.Checkbox(label='OpenAI 임베딩 사용', value=True)
            run_btn=gr.Button('분석 & 추천 실행', variant='primary')
            send_email_btn=gr.Button("추천 결과 이메일로 보내기 (Gmail)")
        with gr.Column(scale=3):
            gr.Markdown('### 4) 추천 결과')
            result_table=gr.Dataframe(headers=["채널 이름","사이트 주소"], row_count=10, col_count=2)

    # 버튼 동작 연결 제거 (Hugging Face에서는 webbrowser.open 사용 불가)
    # 대신 gr.Link 컴포넌트로 대체

    def fetch_data_fn():
        try:
            res = requests.get(f"{BACKEND}/fetch_data", timeout=60)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            return {"error": str(e)}
    fetch_btn.click(fetch_data_fn, inputs=[], outputs=[fetch_result])

    def _run(yt, sns, mbti, use_openai):
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
            data = {"youtube":[{"name":"추천 실패","url":str(e)}], "web":[]}

        rows = []
        youtube_list = data.get("youtube", [])
        web_list = data.get("web", [])
        if not youtube_list and not web_list:
            # fallback: 최소한 1개라도 표시
            youtube_list = [{"name":"추천 실패","url":"http://youtube.com"}]
            web_list = [{"name":"추천 실패","url":"http://example.com"}]

        for c in youtube_list + web_list:
            rows.append([
                c.get("name",""),
                c.get("url","")
            ])
        return rows

    run_btn.click(_run, [yt_text, sns_text, mbti, use_openai], [result_table])

# Vercel 배포를 위해 FastAPI 앱을 생성하고 Gradio 앱을 마운트합니다.
# Vercel은 이 'app' 객체를 찾아 서버리스 함수로 실행합니다.
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

# if __name__=='__main__':
#     demo.launch()
