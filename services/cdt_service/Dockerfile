FROM python:3.11-slim

WORKDIR /app

COPY libs/neurotrust_common /libs/neurotrust_common
COPY services/cdt_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/cdt_service/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
