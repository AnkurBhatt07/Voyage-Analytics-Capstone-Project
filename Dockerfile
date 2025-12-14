# 1. Base Image

FROM python:3.10-slim

# 2. Set working directory inside container

WORKDIR /app

# 3. Copy requirements first(for caching)

COPY requirements.txt .

# 4. Install dependencies

RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy entire project
COPY . .

# 6. Expose flask port
EXPOSE 8000

# 7. Run flask application

CMD ["python", "app.py"]