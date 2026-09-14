FROM python:3.12-slim

WORKDIR /app

COPY requirements-strict.txt .
RUN pip install --no-cache-dir -r requirements-strict.txt

COPY . .

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 10000

ENV FLASK_HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:10000/health', timeout=4).status==200 else 1)"

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "2", "--preload"]
