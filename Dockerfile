FROM apache/airflow:2.10.5-python3.11

USER airflow
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip check
