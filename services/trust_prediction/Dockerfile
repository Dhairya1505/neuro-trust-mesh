FROM python:3.11-slim

WORKDIR /app

COPY libs/neurotrust_common /libs/neurotrust_common
COPY services/trust_prediction/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/trust_prediction/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
