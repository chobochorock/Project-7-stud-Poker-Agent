# 7-stud SOTA 대조군 (2026-11-05 시작)

ASWISS·k-means와 **같은 게임**(heads-up, 규칙 v3, 5구부터, heuristic H4), **같은 실행 파일**(`../aswiss_stud.cpp`), **같은 평가**(Policy-LBR deal·seed·particle, BR-gap 근사치 deal), **같은 시간 기록**(`log.csv`·`curve.csv`·`snapshots.csv`)으로 비교하려고 만든 신경망 대조군이다. 외부 라이브러리는 쓰지 않는다(torch 없음). 그래서 노트북에서도 추가로 설치할 것이 없다.

## 고른 대조군과 진행 상태

| 계열 | 방법 | 한 번 방문이 다른 정보집합을 바꾸는 방식 | 상태 |
|---|---|---|---|
| 정책 기울기 | **MMD 자기대전** (Sokota 외; Rudolph 외 ICLR 2026에서 가장 강했던 계열) | 공유 정책망 | **구현·검증 (이 문서)** |
| 신경망 CFR | ESCHER (McAleer 외 ICLR 2023) | 후회망 | 다음 |
| 실시간 탐색 | 청사진 + 7구 정확 resolving (엔진의 `ExactSeventhResolverPolicy`) | 학습에서 못 본 정보집합을 둘 때 직접 푼다 | 다음 |

## 파일

- `nn.hpp`: 두 은닉층 MLP(ReLU), 희소 첫 층(켜진 입력의 열만 더함), Adam(첫 층은 배치가 건드린 행만 갱신), 저장·불러오기. 기울기 검사 통과(상대 오차 2e-4, float 정밀도).
- `mmd.hpp`: 게임과 무관한 MMD. 같은 코드를 7-stud와 내보낸 트리(`../../compare/nn_tree.cpp`)에서 쓴다.
- `stud7_net.hpp`, `stud7_net_main.hpp`: 7-stud 연결. `train --partition net`, `eval --partition net`.

## MMD 구현

- 갱신마다 현재 정책으로 자기대전 `--mmd-episodes`판(기본 2,048)을 모은다. 두 좌석이 같은 망을 쓰고, 좌석은 입력에 들어 있다.
- 손실 (합법 행동에 대해 정확한 KL):
  L(θ) = E[−(π_θ(a|x)/π_old(a|x))·Â] + α·KL(π_θ ‖ magnet) + β·KL(π_θ ‖ π_old)
  - Â: 몬테카를로 수익 − 가치망 추정, 배치마다 정규화.
  - magnet: 기본은 균등 분포다. `--mmd-magnet-every K`면 K갱신마다 현재 정책으로 옮긴다(R-NaD식).
  - 가치망은 수익/`--mmd-reward-scale`(기본 10)을 회귀한다.
- 입력: 엔진의 Deep CFR 정보상태 텐서(`deep_cfr_tensor`, 1,832차원 one-hot)다. 내 숨은·보인 카드, 상대 보인 카드, 내가 버린 카드, 베팅 이력 24칸, 팟 정보, 합법 행동이 들어 있다.
- 기본값 (7-stud): 은닉 128×64(정책망 파라미터 243,400개), 갱신당 2,048판, 학습 반복 2회, 미니배치 512, 학습률 3e-4(가치 1e-3), α=0.2, β=1.
- 평가: MMD가 보장하는 것은 마지막 반복이라, 신경망은 **현재 정책 하나만** 평가한다. 같은 값을 평균 정책으로 복사하지 않는다(`evaluate.py`가 `partition net`이면 현재 정책만 돌린다).
- 1 root = 표본으로 뽑은 5구 root에서 시작한 자기대전 한 판이다. 스냅숏은 epoch 경계(1만 roots에서 두 배씩)를 넘은 첫 갱신에서 찍으므로, roots가 최대 2,047만큼 늦을 수 있다.

## 검증

- **재개:** 7-stud MMD를 6,144판에서 STOP으로 멈춘 뒤 같은 명령으로 3만 720판까지 이은 실행이, 끊지 않은 실행과 스냅숏 4개와 log(초 열 제외)가 바이트 단위로 같았다. checkpoint에는 두 망, Adam 상태, magnet, 난수 상태, 계수가 들어 있다.
- **표 기반 경로 불변:** 새 실행 파일로 `hist_bbf` 설정 7만 roots를 돌려 이전 실행 파일과 스냅숏, log, backward, fill이 바이트 단위로 같았다.
- **Kuhn (정확한 exploitability, one-hot 입력, 은닉 64×32, 갱신당 512판):**

  | 설정 | 50만 판 | 100만 판 | 비고 |
  |---|---:|---:|---|
  | α=0.2, 균등 magnet | 0.048 | 0.049 | 6.5만 판부터 0.03–0.05로 안정 |
  | α=0.2, magnet 200갱신마다 이동 | 0.015–0.039 | 0.035 | 더 낮아지기도 하지만 들쭉날쭉 |
  | α=0.5 | 0.103 | | 안정, 간격이 큼 |
  | α=0.05 (갱신당 4,096판) | 0.264 | | 0.13–0.28을 오감 |
  | 처음(거의 균등) | 0.43 | | |

  - MMD 이론대로 α가 클수록 정규화 균형(QRE)에 안정적으로 가고, Nash와의 간격은 커진다. α가 작으면 자기대전이 진동한다.
  - 표 기반 CFR(10⁻⁵ 이하)보다 훨씬 높다. 표본 기반 정책 기울기의 잡음 때문이다.
- **Leduc:** 실행 중(400만 판).
- **7-stud 짧은 실행 (이 컨테이너):** 3만 판에서 현재 정책 LBR 45–49(아직 거의 무작위)다. 같은 3만 roots의 `hist`는 2.1–16이다. 신경망은 판이 훨씬 많이 필요하다.

## 비용 (이 컨테이너, 4코어 중 1코어)

- 학습: 갱신 1번(2,048판, 약 1.6만 결정)에 약 0.6초, 즉 초당 약 3,400판이다. 1억 판이면 약 8시간이다. 기존 은닉 256×128·학습 반복 4회는 갱신당 5.2초로 9배 느렸다.
- 평가: 2,000 pair에 약 1분, 최종 20,000 pair에 약 10분이다.

## 실행

```powershell
# 다른 비교군과 같은 폴더, 같은 seed, 같은 roots
powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Arms mmd -SnapshotEvery 0
powershell -ExecutionPolicy Bypass -File agents\adaptive_buckets\aswiss\stud7\run_100m.ps1 -Arms mmd -Action eval
```

- `-Action plot`에 `mmd`를 넣으면 현재 정책 패널에만 그려진다. 평균 정책 패널은 비고, bucket 패널에는 파라미터 수가 찍힌다.
- 시간 비교는 `plot_stud_time.py`로 한다(`../README_KO.md`의 "학습 시간").
- 하이퍼파라미터를 바꾸려면 `--mmd-*` 옵션을 쓴다(`stud7_net_main.hpp`). 바꾸면 checkpoint 설정이 달라지므로 새 출력 폴더가 필요하다.

## 남은 일

1. ESCHER(신경망 CFR)
2. 7구 실시간 resolving을 기존 청사진(ASWISS, k-means)과 MMD에 붙여 평가
3. 같은 벽시계 시간 비교. 실행 순서와 동시 실행 수를 기록한다.
