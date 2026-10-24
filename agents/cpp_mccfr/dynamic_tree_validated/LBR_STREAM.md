# Monitor-skipping LBR

기존 학습 checkpoint의 분할 monitor payload를 seek로 건너뛰는 **평가 전용** 로더다.
Regret/평균전략/트리/atlas/게임/LBR 계산은 그대로 사용한다. 학습 소스의 hash와 원본 모델을
변경하거나 monitor 데이터를 삭제하지 않는다. 로드한 상태로 학습을 재개해서는 안 된다.

## Usage

프로젝트 root에서, 중단/완료된 checkpoint만 사용한다. Current/average DT/KM와 uniform
비교군을 같은 deal/seat/action seed로 평가한다. 기본2seed×250pair×2seat=정책당1,000hand.

```powershell
python -m agents.cpp_mccfr.dynamic_tree_validated.lbr_stream --run agents/cpp_mccfr/data/dynamic_tree_validated/log_gain_seed222_10m --output agents/cpp_mccfr/data/dynamic_tree_validated/lbr_log10m_20261009
```

출력은 새 폴더여야 한다. 첫 컴파일 후 다른 run 평가에는 `--reuse-build PREVIOUS_OUTPUT`으로
검증된 실행 파일을 재사용할 수 있다. 빌드/원본 source hash가 달라지면 거부한다.

동일 조건으로 평가한 동적/분할 중단 결과를 비교하려면 다음을 사용한다.
Deal/seat별 k-means 및 uniform 수익이 정확히 일치하는지 확인한 뒤 두 정책의 그래프와
paired difference 신뢰구간을 생성한다. 원본 평가 결과는 변경하지 않는다.

```powershell
python -m agents.cpp_mccfr.dynamic_tree_validated.lbr_compare --dynamic agents/cpp_mccfr/data/dynamic_tree_validated/lbr_log10m_20261009 --frozen agents/cpp_mccfr/data/dynamic_tree_validated/lbr_freeze10m_20261009 --output agents/cpp_mccfr/data/dynamic_tree_validated/lbr_10m_compare_20261009
```

2026-10-09 회귀 검사: `lbr_stream_parity_20261009`, seed2026100831/2pair/64particles의
5정책 raw CSV 전체가 기존 `lbr_freeze10m_20261008`의 첫2pair와 문자열까지 동일했다.
이는 큰 동적 모델 전체를 기존 로더로 다시 평가한 검사는 아니다.

## Scope and validation

- `solver.cpp`/`lbr.cpp`를 출력build폴더에 기계적으로 복사하고, load의 sample loop만
  bounds-checked seek로 바꾼다. 기존 table/node/root/RNG 검증은 유지한다. 앵커가 바뀌면 실패한다.
- 작은 nonempty-monitor 트리에서 원래 로더와 cursor, monitor 외 직렬화 전체, current/average
  정책의 일치를 검사한다. 잘린 checkpoint 거부 및 기존 privacy/legal/reproducibility/unit 검사.
- 실제 payload 값의 finite 검사는 생략한다. 평가에 쓰지 않는 payload의 의미적 무결성 검사는
  아니며 정책/트리 및 파일 경계만 검증한다. `seek`이므로 큰 checkpoint 전체 hash도 계산하지 않는다.
- 한 번 로드해 모든 평가seed를 실행한다. 읽은 트리/정책은 메모리에 남지만 monitor는 남지 않는다.
  HDD의 작은 header read/seek 비용은 여전히 발생할 수 있다. load시간/peak RSS는 `loader.json`.
- Writer lock과 source/snapshot/atlas/config identity/header-log offsets를 확인한다.
  입력 파일 크기/mtime은 전후 검사한다. 새checkpoint나resume export를 만들지 않는다.
- `summary.json`, `<seed>/paired_payoffs.csv`, `<seed>/evaluation.json`, `console.log`에
  수익/seed내deal-pair bootstrap95%CI/정책조회누락/ESS/메모리와 출처를 저장한다.
- 작은 예산의 기존 full-loader 결과와 같은 seed/pairs/particles로 실행한 뒤 raw CSV prefix의
  일치를 확인하면 end-to-end 회귀 검사가 된다. 학습seed 변동성이나 통계적 성능 증명은 아니다.

## Metric

[Lisy & Bowling (2017)](https://poker.cs.ualberta.ca/publications/aaai17ws-lisy-lbr.pdf)의
local best response와 관련된 기존 particle/check-down 구현을 그대로 사용한다. 새로운
평가기법이 아니며 원 논문의 hold'em 구현을 그대로 복제한 것도 아니다. 이번 변경은 로딩만
최적화한다. 수익은 LBR 공격자의 ante/hand, 낮을수록 target에 유리하다. 정확 exploitability나
학습 중 Local BR-gap과 혼동하지 않는다. H4는 양쪽 모두 기존 휴리스틱이며 별도12행동 모델은
이 평가에 연결하지 않는다. 희귀 all-in으로 CI가 넓어질 수 있다.
