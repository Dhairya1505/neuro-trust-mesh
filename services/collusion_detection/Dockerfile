FROM python:3.11-slim

WORKDIR /app

COPY libs/neurotrust_common /libs/neurotrust_common
COPY services/collusion_detection/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/collusion_detection/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
