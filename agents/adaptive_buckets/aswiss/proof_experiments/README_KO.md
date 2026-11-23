# 증명 개요 실험 (1단 행렬게임)

설명과 결과는 [../../ASWISS_PROOF_OUTLINE_KO.md](../../ASWISS_PROOF_OUTLINE_KO.md) 4–5절에 있다.

- `original/`: 받은 실험 기록(`aswiss_experiments.zip`, 2026-11-23) 그대로. `ds_ab.tar.gz`는 `ds_a.jsonl`, `ds_b.jsonl`.
- `repro_2026-11-23/outputs/`: README 명령을 그대로 다시 돌린 출력과 시간(`timings.txt`).
- `repro_2026-11-23/rule_compare/`: 추가 비교(최대 이득 규칙 대 순수 ASWISS의 전부 나누기 규칙)와 균형 유일성 제외 비율.
- 다시 돌리기: `bash repro_2026-11-23/run_repro.sh OUT [WORKERS]` (numpy, scipy; 4코어 약 10분).
