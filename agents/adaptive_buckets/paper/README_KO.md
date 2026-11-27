# 논문 초안: 순수 ASWISS 본문판 (2026-11-27)

`main.pdf`가 빌드된 초안이다. 원고는 `main.tex`와 `sections/*.tex`, 참고 문헌은 `refs.bib`다.

빌드 (XeLaTeX, kotex, 나눔 글꼴):

```bash
cd agents/adaptive_buckets/paper
latexmk -xelatex main.tex        # 또는 xelatex → bibtex → xelatex 두 번
```

## 구성

| 절 | 파일 | 내용 |
|---|---|---|
| 1 서론 | `sec1_intro.tex` | 문제, 순수 ASWISS 요약, 기여 |
| 2 관련 연구 | `sec2_related.tex` | 가상 플레이(FPIRA 포함), CFR 계열, MCCFR, 함수 근사(Deep CFR, DREAM, ESCHER), 1차 방법과 GDA(MMD 포함), 정보 추상화, CFR+IRA, 실시간 탐색 |
| 3 배경 | `sec3_background.tex` | CFR 정리 전체와 증명(성능 차이 항등식, 후회 분해, 후회 매칭, folk theorem, RM+), MCCFR 불편성 증명과 외부 샘플링 수렴 속도, CFR+IRA의 분할 개념과 수렴 정리(원문 정리 2), 휴리스틱 반례 |
| 4 순수 ASWISS | `sec4_method.tex` | 시작 분할, 누적 후회 라벨, 분할 규칙, 계보 라우팅, 재생 라벨, 판정 시점, 알고리즘, 이론판과 구현의 차이 |
| 5 이론 결과 | `sec5_theory.tex` | 정리 문장만 (증명은 부록 A) |
| 6 실험 | `sec6_experiments.tex` | 작은 게임(Kuhn, Leduc, mini-stud), CFR+IRA·k-means·MMD·ESCHER 비교, 순수판, 계보 비용 |
| 7 파생 기법 | `sec7_derived.tex` | 평균 전략 방향(정리 E), bucket 이력, 관측 분할·전략 분할(옛 prior–posterior), 고정 문턱과 1단 게임 반례, 세븐 포커 |
| 8 한계와 후속 연구 | `sec8_future.tex` | 초기 추상화, 분할 개선, 3인 이상, absent-mindedness, 관측 분할 구성 |
| 부록 A, B | `appA_proofs.tex`, `appB_repro.tex` | 증명, 재현 경로 |

## 제출 전에 확인할 것

- **CFR+IRA 정리:** 3.6절 정리(원문 정리 2)의 $\alpha(\delta)$는 arXiv PDF에서 추출한 글자로 옮겼다. 근호와 지수의 배치를 원문에서 다시 확인한다(본문에 빨간 [확인] 표시).
- **MCCFR 원 논문:** 이 세션에서는 Lanctot 외(2009) 원문을 열지 못했다. 그래서 3.4절은 다음 두 문헌에서 옮겼다.
  - 불편성: 원 논문 Lemma 1의 정의와 같은 식(VR-MCCFR 논문의 재인용으로 확인)
  - 외부 샘플링 수렴 속도: Farina 외(2020, arXiv:2002.08493)의 명제 2와 식 (6)

  원 논문의 상수 표기(정리 4)는 확인하지 못했다.
- **참고 문헌:** `refs.bib`의 서지 정보(쪽, 권호)는 확실한 것만 적었다. 모두 원문으로 다시 확인한다.
- **문헌 조사:** "표본 기반 세분 + 원 게임 수렴 증명"을 함께 다룬 선행 연구가 없다는 주장은 2026-11-06 조사 기준이다.
- **실험:** 실험은 seed 1개(일부 4–10개)다. 세븐 포커 수치는 LBR(exploitability의 근사 하한)이다. 작은 게임 수치는 정확한 exploitability(NashConv/2)다.
- **이론과 구현의 차이:** 정리는 이론판(RM+, 동시 갱신, 구간 균등 혼합)에 대한 것이다. 구현과의 차이는 4.7절 표에 있다.
- **파생 기법의 상태:** 7.4절 고정 문턱 판의 결과는 증명 개요(`../ASWISS_PROOF_OUTLINE_KO.md`) 단계다.
- **증명 검토:** 증명은 아직 다른 사람이 검토하지 않았다.

## 출처

- 증명: `../ASWISS_CONVERGENCE_KO.md` (Lemma 1–10, 정리 1·2·2′·A·E)
- 실험 수치:
  - `../aswiss/compare/README_KO.md`
  - `../PP_STUD7_DESIGN_KO.md` 22.12–22.13절
  - `../ASWISS_PROOF_OUTLINE_KO.md`
  - 세븐 포커: `../aswiss/stud7/README_KO.md`
- 이전 초안: 저장소 루트의 `졸업 논문 초안 (병합).md`. 원안 ASWISS(분할 시점 전략 하나의 방향)를 본문으로 둔 판이다. 이 초안은 순수 ASWISS(반복 전략들의 누적 후회 라벨)를 본문으로 하고, 원안과 다른 가속 기법을 7절 파생 기법으로 옮겼다.
