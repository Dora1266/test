FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    tcl8.6-dev \
    tk8.6-dev \
    python3-tk \
    libharfbuzz-dev \
    libfribidi-dev \
    libxcb1-dev \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

RUN mkdir -p templates

RUN mkdir -p /app/uploads

ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=document_parser.py
ENV UPLOAD_FOLDER=/app/uploads

EXPOSE 5000

# 启动应用
CMD ["python", "document_parser.py", "server", "--host", "0.0.0.0", "--port", "5000"]