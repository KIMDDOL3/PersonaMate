Run frontend:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export BACKEND_URL=http://localhost:8000
python app.py
