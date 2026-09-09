FROM python:3.11-slim

WORKDIR /app

COPY libs/neurotrust_common /libs/neurotrust_common
COPY simulation/mesa_sim/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY simulation/mesa_sim/ .

CMD ["python", "run.py"]
