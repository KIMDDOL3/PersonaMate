# Hugging Face Spaces entrypoint
# 기존 frontend/app.py를 복사하여 루트에서 실행되도록 함

import sys
import os

# frontend 디렉토리를 모듈 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), "frontend"))

# frontend.app 실행
from app import *  # noqa
