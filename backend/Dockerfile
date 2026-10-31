FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

# One worker. Cloud Run gives this process the whole CPU for the request.
CMD ["sh", "-c", "exec gunicorn main:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} --timeout 300 --graceful-timeout 30"]
