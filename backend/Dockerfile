FROM python:3.11-slim-bookworm

WORKDIR /home/app

# Copy only requirements first to prevent bust cache
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]