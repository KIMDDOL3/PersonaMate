# Backend Dockerfile for container deployment
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy backend requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend ./backend
COPY .env.example .env

# Expose port for FastAPI
EXPOSE 8000

# Launch UVicorn server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
