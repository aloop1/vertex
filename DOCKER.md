# Docker로 웹앱 실행하기

크립 수명 예측 웹앱을 컨테이너로 띄워, 파이썬 환경 설정 없이 언제든 켜고 끌 수 있게 한 구성입니다.

## 처음 한 번만

```bash
docker compose build
```

torch CPU 휠을 받느라 처음엔 몇 분 걸립니다. 이후로는 캐시가 재사용됩니다.

## 켜고 끄기

| 하고 싶은 것 | 명령 |
|---|---|
| 켜기 | `docker compose up -d` |
| 끄기 (컨테이너 유지 — 다시 켤 때 빠름) | `docker compose stop` |
| 다시 켜기 | `docker compose start` |
| 완전히 내리기 (컨테이너 삭제) | `docker compose down` |
| 상태 보기 | `docker compose ps` |
| 로그 보기 | `docker compose logs -f` |

켜고 나면 **http://localhost:5000** 으로 접속합니다.

Docker Desktop을 쓰신다면 명령어 대신 GUI의 컨테이너 목록에서 `vertex-web`의
시작/정지 버튼을 눌러도 똑같이 동작합니다.

### 재부팅하면?

`restart: unless-stopped` 정책이라 **PC를 껐다 켜도 자동으로 다시 올라옵니다.**
단, 직접 `docker compose stop`으로 멈춰둔 상태라면 그 의사를 기억해서 켜지 않습니다.
자동 시작이 싫으면 `docker-compose.yml`에서 해당 줄을 `restart: "no"`로 바꾸세요.

## 모델이나 GA 결과를 갱신했을 때

`models/`, `data/`, `ga/` 는 호스트 폴더를 그대로 마운트합니다. 그래서
호스트에서 모델을 다시 학습하거나 `python ga/engine.py`를 돌리면
**재빌드 없이** 컨테이너를 재시작하는 것만으로 반영됩니다.

```bash
python ga/engine.py          # 호스트에서 GA 재실행
docker compose restart       # 컨테이너만 재시작하면 새 결과가 보인다
```

소스 코드(`web/`)를 고쳤을 때만 재빌드가 필요합니다.

```bash
docker compose up -d --build
```

## 프로덕션 모델 쓰기

지금은 `models/transformer_tree_ensemble.pkl`이 없어서 **smoke 모델**
(3 epoch, 8 trees — 품질이 낮은 테스트용)로 자동 폴백합니다.
제대로 학습한 모델을 만들었다면 `models/`에 넣고, `docker-compose.yml`의
`VERTEX_MODEL_PATH` 주석을 해제하세요.

현재 어떤 모델이 로드됐는지는 아래로 확인할 수 있습니다.

```bash
curl http://localhost:5000/health
```

## 설정 바꾸기

`docker-compose.yml`의 `environment`에서 조정합니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `PORT` | `5000` | 컨테이너 내부 포트. 외부 포트는 `ports`에서 바꿉니다 |
| `VERTEX_THREADS` | `8` | waitress 워커 스레드 수 |
| `VERTEX_MODEL_PATH` | (없음) | 모델 아티팩트 경로 강제 지정 |

5000번 포트가 이미 쓰이고 있다면 `ports`를 `"8080:5000"` 처럼 바꾸면 됩니다.

## 이미지에 무엇이 들어있나

- 웹 추론에 필요한 것만 설치합니다 (`requirements-web.txt`).
  학습·GA용인 pycalphad, deap, optuna, xgboost, streamlit은 제외했습니다.
- torch는 CPU 전용 휠입니다. 기본 인덱스를 쓰면 CUDA 런타임까지 따라와
  이미지가 수 GB 더 커지는데, 추론 모델이 작아 CPU로 충분합니다.
- 비루트 사용자(`vertex`, uid 10001)로 실행합니다.
- 이미지 크기는 약 1.9GB이고, 대부분이 torch입니다.

GA 최적화와 모델 학습은 컨테이너에 넣지 않았습니다. 계산이 무겁고 실행 시간이
길어, 호스트에서 돌린 결과를 볼륨으로 넘겨받는 편이 실용적이기 때문입니다.
