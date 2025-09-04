# PersonaMate Pro: 기술 요구사항 및 프로젝트 회고 (TRD & Post-mortem)

## 1. 최종 아키텍처

이 프로젝트는 프론트엔드와 백엔드를 분리하여 독립적으로 배포하고 운영하는 **Headless 아키텍처**를 채택했습니다.

*   **프론트엔드 (UI):**
    *   **기술 스택:** Python, Gradio
    *   **배포:** Hugging Face Spaces
    *   **역할:** 사용자 인터페이스 제공, 사용자 입력 수집, Vercel 백엔드 API 호출 및 결과 렌더링.

*   **백엔드 (API):**
    *   **기술 스택:** Python, FastAPI
    *   **배포:** Vercel Serverless Functions
    *   **역할:** Google OAuth 인증 처리, YouTube API 데이터 수집, Google Gemini API를 이용한 추천 로직 수행, 결과 파일(HTML/PDF) 생성 및 이메일 전송.

*   **데이터베이스:**
    *   **서비스:** Supabase (PostgreSQL)
    *   **역할:** Google OAuth 토큰, 추천 결과 등 영구 데이터 저장.

*   **CI/CD:**
    *   **서비스:** GitHub Actions
    *   **역할:** `main` 브랜치에 코드 푸시 시, Hugging Face Spaces와 Vercel에 자동으로 프론트엔드와 백엔드를 각각 배포.

## 2. 주요 시행착오 및 해결 과정

### 2.1. 프론트엔드 (Hugging Face Spaces)

*   **문제 1: `gr.Dataframe`에서 HTML 링크 렌더링 불가**
    *   **현상:** 백엔드에서 `<a>` 태그를 포함한 HTML 문자열을 전달했으나, Gradio의 `Dataframe` 컴포넌트가 이를 일반 텍스트로 표시.
    *   **해결:** `gr.Dataframe` 대신 `gr.HTML` 컴포넌트를 사용하도록 변경. 백엔드 API 응답을 받아 프론트엔드에서 직접 `<table>` 태그를 포함한 전체 HTML 문자열을 생성하여 `gr.HTML`에 전달함으로써, 클릭 가능한 하이퍼링크를 구현.

*   **문제 2: OpenAI 체크박스 및 관련 로직 불필요**
    *   **현상:** 초기 기획에 있던 "OpenAI 임베딩 사용" 기능이 최종적으로 사용되지 않게 됨.
    *   **해결:** `frontend/app.py`에서 `gr.Checkbox` UI 컴포넌트와 관련 로직(`run_recommendations` 함수의 `use_openai` 인자 등)을 모두 제거하여 코드를 단순화.

*   **문제 3: Hugging Face Spaces 배포 설정 오류**
    *   **현상:** `README.md`의 메타데이터(SDK, entrypoint 등)가 프로젝트 구조와 일치하지 않아 배포 실패.
    *   **해결:** 루트 `README.md`에 `sdk: gradio`, `app_file: "frontend/app.py"` 등 정확한 메타데이터를 명시하여 Hugging Face가 올바른 파일을 실행하도록 수정.

### 2.2. 백엔드 (Vercel)

*   **문제 1: 서버리스 함수 크기 제한 초과 (250MB)**
    *   **현상:** `sentence-transformers`, `faiss-cpu` 등 대용량 ML 라이브러리를 `requirements.txt`에 포함하여 배포 시, Vercel의 서버리스 함수 최대 크기(250MB)를 초과하여 빌드 실패.
    *   **해결:**
        1.  **아키텍처 변경:** 대용량 라이브러리를 직접 실행하는 대신, 외부 AI 서비스인 **Google Gemini API**를 호출하는 방식으로 변경.
        2.  **의존성 분리:** `backend/requirements.txt`에는 FastAPI, httpx 등 백엔드 실행에 필수적인 경량 라이브러리만 남겨두어 용량 문제 해결.
        3.  **.vercelignore 추가:** `frontend/` 디렉토리, `app.py` 등 백엔드 배포에 불필요한 파일들을 `.vercelignore`에 추가하여 Vercel 빌드에서 제외.

*   **문제 2: `404 NOT_FOUND` 및 `500 Internal Server Error`**
    *   **현상:** Google OAuth 인증 시도 시 `404` 오류, 데이터 수집 및 추천 실행 시 `500` 오류 발생.
    *   **원인 분석 및 해결:**
        1.  **라우팅 문제:** `vercel.json`의 `routes` 설정이 잘못되어 `/oauth/google/start`와 같은 API 경로를 `backend/main.py`로 제대로 연결하지 못함. → `routes` 설정을 수정하여 모든 API 경로가 `backend/main.py`를 가리키도록 명확히 지정.
        2.  **환경 변수 누락/오류:** Vercel Production 환경에 `DATABASE_URL`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`, `GEMINI_API_KEY` 등이 누락되거나 잘못된 값으로 설정되어 있었음. → Supabase 및 Google Cloud Console에서 정확한 값을 확인하여 Vercel 환경 변수를 올바르게 설정.
        3.  **고정되지 않은 리디렉션 URI:** Vercel 배포 시마다 생성되는 프리뷰 URL을 `GOOGLE_REDIRECT_URI`로 사용하여 OAuth 인증 실패. → Vercel 프로젝트의 고정된 Production 도메인 별칭(`personamate-kimddols-projects.vercel.app`)을 찾아 `GOOGLE_REDIRECT_URI`로 설정하여 문제 해결.

*   **문제 3: Gemini API 응답 파싱 실패**
    *   **현상:** Gemini API가 유효하지 않은 JSON(중간에 잘리거나, 형식이 틀림)을 반환하여 백엔드에서 파싱 오류 발생.
    *   **해결:**
        1.  **프롬프트 엔지니어링:** 프롬프트에 JSON 형식에 대한 지시를 더 명확하고 상세하게 추가.
        2.  **`maxOutputTokens` 증가:** Gemini API의 최대 출력 토큰 수를 늘려 응답이 중간에 잘리는 것을 방지.
        3.  **`temperature` 조정:** `temperature` 값을 조정하여 더 안정적이고 예측 가능한 형식의 응답을 유도.

### 2.3. CI/CD (GitHub Actions)

*   **문제 1: 워크플로우 파일(`deploy.yml`) 오류**
    *   **현상:** 워크플로우 파일에 `name: deploy` 섹션이 중복되고, Docker Hub 로그인 정보가 잘못 설정되어 모든 배포 job이 실패.
    *   **해결:** 중복된 섹션을 제거하고, `deploy-huggingface`, `deploy-vercel`, `deploy-container` job을 하나의 워크플로우 아래에 통합. Docker Hub 로그인 정보도 `secrets.DOCKERHUB_USERNAME`, `secrets.DOCKERHUB_TOKEN`으로 통일.

*   **문제 2: Vercel CLI 비대화형 배포 실패**
    *   **현상:** GitHub Actions 환경에서 Vercel CLI가 배포 확인 프롬프트를 띄우면서 멈춤.
    *   **해결:** `vercel deploy` 명령어에 `--yes` 플래그를 추가하여 비대화형(non-interactive) 모드로 실행되도록 수정.

## 3. 결론 및 교훈

이번 프로젝트는 프론트엔드, 백엔드, 데이터베이스, CI/CD를 통합하는 과정에서 발생할 수 있는 다양한 문제들을 경험하고 해결하는 좋은 기회였습니다. 특히, 서버리스 환경의 제약(용량, 실행 시간)을 이해하고, 이를 외부 API(Gemini)와 데이터베이스(Supabase)를 활용하여 극복하는 방법을 배울 수 있었습니다.

**주요 교훈:**
*   **환경 변수 관리의 중요성:** 로컬, 프리뷰, 프로덕션 환경 간의 환경 변수(API 키, URL 등)를 정확하고 일관되게 관리하는 것이 매우 중요합니다.
*   **로그 기반 디버깅:** `500 Internal Server Error`와 같은 모호한 오류는 반드시 Vercel, Supabase, GitHub Actions 등의 상세 로그를 확인하여 근본적인 원인을 찾아야 합니다.
*   **프론트엔드와 백엔드의 명확한 분리:** Vercel 서버리스 배포 시, `.vercelignore`를 활용하여 백엔드에 불필요한 프론트엔드 파일이 포함되지 않도록 하는 것이 중요합니다.
*   **외부 API 연동 시 유연한 파싱 로직:** LLM API는 응답 형식이 항상 일정하지 않을 수 있으므로, JSON 파싱 등 후처리 로직을 견고하게 작성해야 합니다.
