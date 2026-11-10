FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config ./config
COPY src ./src
COPY app ./app
COPY models ./models

EXPOSE 8000

# Train first (python main.py) so models/ holds best_model.pkl and scaler.pkl
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
