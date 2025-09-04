import os, json, requests
import gradio as gr
from dotenv import load_dotenv
import httpx
import time

# 환경 변수 로드
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "backend", ".env"), override=True)

# Vercel 백엔드 URL
BACKEND = os.getenv('BACKEND_URL', 'https://personamate-kimddols-projects.vercel.app')

async def fetch_data_fn():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{BACKEND}/fetch_data", timeout=60)
            res.raise_for_status()
            return res.json()
    except Exception as e:
        return {"error": str(e)}

async def run_recommendations(yt, sns, mbti):
    try:
        payload = {
            "youtube_subscriptions": [s.strip() for s in yt.splitlines() if s.strip()],
            "sns_keywords": [s.strip() for s in sns.splitlines() if s.strip()],
            "mbti": mbti
        }
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{BACKEND}/youtube/recommendations", json=payload, timeout=120)
            res.raise_for_status()
            data = res.json()
    except Exception as e:
        return "<h3>추천 결과를 가져오는 데 실패했습니다.</h3>", f"API 호출 실패: {e}", None

    recommendations_data = data.get("recommendations", {})
    youtube_recs = recommendations_data.get("youtube", [])
    summary_reason = recommendations_data.get("summary_reason", "추천 사유를 생성하지 못했습니다.")

    if not youtube_recs:
        return "<h3>추천 결과가 없습니다.</h3>", summary_reason, None

    table_html = "<table><thead><tr><th>채널 이름</th><th>사이트 주소</th><th>추천 사유</th></tr></thead><tbody>"
    for c in youtube_recs:
        url = c.get("url", "")
        name = c.get("name", "")
        reason = c.get("reason", "")
        table_html += f'<tr><td>{name}</td><td><a href="{url}" target="_blank">{url}</a></td><td>{reason}</td></tr>'
    table_html += "</tbody></table>"
    
    # Store the necessary data for export/email
    state_data = {
        "recommendations": {"youtube": youtube_recs},
        "summary_reason": summary_reason
    }
    
    return table_html, summary_reason, state_data

async def export_file(file_type, recommendations_state):
    if not recommendations_state:
        return None, "먼저 추천을 실행해주세요."
        
    endpoint = f"{BACKEND}/youtube/recommendations/export/{file_type}"
    try:
        # The payload is already in the correct format in recommendations_state
        async with httpx.AsyncClient() as client:
            res = await client.post(endpoint, json=recommendations_state, timeout=60)
            res.raise_for_status()
            
            file_path = f"/tmp/recommendations_{int(time.time())}.{file_type}"
            with open(file_path, "wb") as f:
                f.write(res.content)
            return file_path, f"{file_type.upper()} 파일 생성 완료"
    except Exception as e:
        return None, f"파일 생성 실패: {e}"

async def send_email_fn(recipient_email, recommendations_state):
    if not recommendations_state:
        return "먼저 추천을 실행해주세요."
    if not recipient_email:
        return "이메일 주소를 입력해주세요."

    endpoint = f"{BACKEND}/youtube/recommendations/email"
    payload = {
        "recipient_email": recipient_email,
        "recommendations": recommendations_state.get("recommendations", {}),
        "summary_reason": recommendations_state.get("summary_reason", "")
    }
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(endpoint, json=payload, timeout=60)
            res.raise_for_status()
            return "이메일 전송 성공!"
    except Exception as e:
        return f"이메일 전송 실패: {e}"

with gr.Blocks(title='PersonaMate Pro (OAuth + Simplified UI)') as demo:
    recommendations_state = gr.State()

    gr.Markdown('## PersonaMate Pro — OAuth 수집 + 추천 UI')
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('### 1) OAuth 로그인')
            gr.HTML(f'<a href="{BACKEND}/oauth/google/start" target="_blank">Google (YouTube) 로그인</a>')
        with gr.Column(scale=2):
            gr.Markdown('### 2) 자동 수집')
            fetch_btn = gr.Button('내 계정에서 데이터 수집')
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
            run_btn = gr.Button('분석 & 추천 실행', variant='primary')
        with gr.Column(scale=3):
            gr.Markdown('### 4) 추천 결과')
            result_html = gr.HTML(label="추천 결과")
            gr.Markdown('### 5) 추천 사유')
            summary_output = gr.Markdown(label="추천 사유 요약")
    with gr.Row():
        gr.Markdown('### 6) 결과 저장 및 공유')
    with gr.Row():
        with gr.Column(scale=1):
            html_btn = gr.Button("HTML 저장")
            pdf_btn = gr.Button("PDF 저장")
            download_file = gr.File(label="다운로드")
        with gr.Column(scale=2):
            email_input = gr.Textbox(label="이메일 주소", placeholder="결과를 받을 이메일을 입력하세요...")
            email_btn = gr.Button("이메일로 보내기")
            status_output = gr.Textbox(label="상태", interactive=False)

    fetch_btn.click(fetch_data_fn, inputs=[], outputs=[fetch_result])
    run_btn.click(run_recommendations, [yt_text, sns_text, mbti], [result_html, summary_output, recommendations_state])
    
    html_btn.click(export_file, inputs=[gr.State("html"), recommendations_state], outputs=[download_file, status_output])
    pdf_btn.click(export_file, inputs=[gr.State("pdf"), recommendations_state], outputs=[download_file, status_output])
    email_btn.click(send_email_fn, inputs=[email_input, recommendations_state], outputs=[status_output])

if __name__ == '__main__':
    demo.launch()
