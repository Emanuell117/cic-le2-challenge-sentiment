FROM python:3.11-slim-bookworm

WORKDIR /home/app

# Copy only requirements first to prevent bust cache
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Set the environment variable for the backend URL
ENV BACKEND_URL=http://backend:8000

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]