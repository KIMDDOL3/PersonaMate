import os, sys
from fastapi import FastAPI
import gradio as gr

# frontend 모듈 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "frontend"))

# frontend/app.py에서 정의한 Gradio Blocks를 가져옵니다.
from frontend.app import demo

# FastAPI 앱 생성 후 Gradio Blocks를 "/ "에 마운트
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

# Hugging Face Spaces는 uvicorn을 내부적으로 실행하므로, 여기서는 따로 launch 하지 않습니다.
