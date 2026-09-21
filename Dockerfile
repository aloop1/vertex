# Vertex 크립 수명 예측 웹앱 (추론 전용)
#
# 빌드:  docker compose build
# 실행:  docker compose up -d
#
# 학습(models/)과 GA(ga/engine.py)는 이 이미지에 포함하지 않는다.
# 계산이 무겁고 pycalphad 등 컴파일 의존성이 필요해, 호스트에서 돌린 결과 파일을
# 볼륨으로 넘겨받는 구조다 (docker-compose.yml 참고).

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# torch 를 CPU 전용 휠로 먼저 설치한다.
# 기본 PyPI 인덱스의 torch 는 CUDA 런타임을 함께 끌어와 이미지가 수 GB 커진다.
COPY requirements-web.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "$(grep -E '^torch==' requirements-web.txt)" \
 && pip install -r requirements-web.txt

# 소스 복사. 실제로 쓰이는 것만 넣어 이미지와 재빌드 범위를 줄인다.
# (데이터/모델/GA 결과는 compose 에서 볼륨으로도 덮어쓴다 — 호스트에서 갱신하면
#  재빌드 없이 반영된다.)
COPY web/ ./web/
COPY models/ ./models/
COPY data/ ./data/
COPY ga/ ./ga/
COPY 데이터전처리.py data_preprocessing.py ./

# 비루트 실행. app.py 가 import 시점에 web/uploads 를 mkdir 하므로
# 해당 디렉터리에 쓰기 권한이 있어야 한다.
RUN useradd --create-home --uid 10001 vertex \
 && mkdir -p /app/web/uploads \
 && chown -R vertex:vertex /app/web
USER vertex

EXPOSE 5000

# /health 는 모델 로딩까지 확인한다. 모델 로딩에 시간이 걸리므로
# start-period 를 넉넉히 준다 (그 동안의 실패는 unhealthy 로 치지 않음).
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=8).status == 200 else 1)"

CMD ["python", "web/serve.py"]
