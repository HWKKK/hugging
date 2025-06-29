FROM python:3.10-slim

WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-nanum \
    fontconfig \
    libavif-dev \
    && rm -rf /var/lib/apt/lists/*

# 한글 폰트 설정 - 이미 fonts-nanum 패키지로 설치됨
RUN fc-cache -f -v

# 애플리케이션 파일 복사
COPY . /app/

# Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 데이터 디렉토리 생성
RUN mkdir -p /app/data/personas /app/data/conversations

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# 앱 실행
CMD ["python", "app.py"] 