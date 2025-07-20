FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

    
COPY prediction/ prediction/
COPY templates/ templates/
COPY eps_api.py .
COPY label_encoders.pkl .

EXPOSE 8000

CMD ["uvicorn", "eps_api:app", "--host", "0.0.0.0", "--port", "8000"]
