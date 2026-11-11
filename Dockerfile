FROM python:3.12-slim

WORKDIR /bot
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY docs ./docs

# Ключи передаются через --env-file .env, в образ не попадают
CMD ["python", "-m", "app.bot"]
