FROM python:3.11-slim

WORKDIR /app

COPY requirements-strict.txt .
RUN pip install --no-cache-dir -r requirements-strict.txt

COPY . .

EXPOSE 10000

ENV FLASK_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "2", "--preload"]
