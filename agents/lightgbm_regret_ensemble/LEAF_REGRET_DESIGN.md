# Tree별 Leaf Regret: 구현 오류 정정과 설계

2026-09-20 후속 갱신. **Leaf별 구현, 200핸드 검증, refined 100k 학습과 독립 평가 완료.**
현재 기본값은 leaf별 저장이며, 과거 joint 방식은 명시적 옵션과 구형 loader로 보존한다.
[실행 명령과 최신 결과 상태](data/leaf_refined_100k_seed7_20260920/README.md).
아래 1~6절의 '현재'는 구현 수정 전 감사 시점을 뜻한다. 실제 채택한 계약은 7절을 따른다.
최신 정밀 대조는 6절을 따른다. Joint는 **한 구간 안의 boosting tree들**에 대한 것이며,
과거 구간과 현재 구간을 교차한 것이 아니다. 구간별 선형 합산과 새 regret 분리는 이미 구현돼 있다.

## 1. 원래 요구와 실제 구현의 차이

사용자의 원래 의도는 10k핸드마다 형성한 각 tree의 각 leaf에 해당 구간의 누적 regret을
저장하고, 조회 시 도달한 leaf들의 regret을 결합하는 것이다. 이것은 새로 제안된 변경이 아니라
처음부터 요구된 구조다. Assistant가 이를 모든 tree의 leaf ID 조합별 table로 잘못 구현했다.

현재 `GroupModel::route`는 모든 tree의 leaf ID를 이어 붙이고,
`EpochModel::query`는 그 전체 문자열이 table에 있는지 검사한다.
예를 들어 저장된 조합이 `(0,0)`, `(1,1)`뿐이면 새 입력 `(0,1)`은 각 tree에서 유효한
leaf에 도달하지만 joint table에는 없을 수 있다. 잘못된 것은 tree의 부등식 routing이 아니라
그 이후 regret 저장·조회 단위다. 미조회 시 zero regret, 전부 미조회 시 균등정책이 사용됐다.

### 1M 감사 근거

- `data/seven_stud_1m_seed7_20260919_204332/config.json`: 총 1M, 구간 10k, 100구간.
- `epoch_001/model.bin`~`epoch_100/model.bin`의 header 100개 모두 `EPOCH7V1`.
- 보관된 `sources/run_epoch_ensemble.py`: `unique_leaf_signatures`와 `np.add.at(table, routes, ...)`.
- 보관된 `sources/stud_epoch_ensemble.cpp`: signature 연결 후 table exact lookup.
- 보관된 두 소스의 SHA256은 당시 config와 일치한다.
  Python: `17cecdb0cce37f64e24fe2ef6a62fb083b3aaf2942c6f3362b2b7db8dcc5bdaa`.
  C++: `28083468d0fe6f4a30b25b9397d4138928bc3fbf1c7be11ee9bbc5692fd51357`.
- 이전 100k의 10개 모델, 이번 refined/soft 10k의 각 10개 모델도 모두 같은 header다.

**따라서 기존 1M 결과로 원래 아이디어가 실패했다고 판단할 수 없다.**
그 결과는 의도와 다른 모델의 실제 측정값으로 보존한다. 별도로 학습한 Hard-256 모델까지
이 오류로 무효가 되는 것은 아니다. 대전 기록도 삭제하지 않고 ensemble의 정체를 정정한다.

기존 joint table을 tree별로 재집계하는 사후 변환 자체는 가능하지만, 당시 행동과 이후
학습 데이터는 잘못된 정책으로 생성됐다. 변환만으로 올바른 방식의 1M 학습을 복구할 수 없다.

## 2. 확정된 수정 요구

- 10k핸드마다 abstraction 생성. 이후 비교는 각 방식 총 100k, 10개 구간 모델.
- 각 tree의 leaf별로 해당 구간에서 새로 발생한 signed regret 벡터를 합산한다.
- 과거 구간의 regret을 새 leaf에 다시 넣지 않는다. 구간 가중치 1,2,...는 조회 시 한 번 적용한다.
- player/street/legal-action-mask가 다른 정보집합을 같은 leaf에 넣지 않는다.
- leaf 조합을 key로 하는 새로운 lookup은 만들지 않는다.
- Refined 경계는 `[0,.05,.1,.2,.3,.4,.5,.95,1]`로 수정했다.
  마지막 두 구간은 `[.5,.95)`, `[.95,1]`이다. Soft CE target은 연속 확률 그대로다.

## 3. Regret 저장과 결합 제안

구간 e의 정보집합 행 i에는 그 구간의 새 누적 regret 벡터 `delta_r[e,i,a]`가 있다.
같은 정보집합의 여러 방문은 이미 그 행에 누적되어 있다. Tree j의 leaf 함수는 `leaf[e,j](x)`다.

$$R_{e,j,\ell}(a)=\sum_{i:\,\operatorname{leaf}_{e,j}(x_i)=\ell}\Delta r_{e,i}(a).$$

이는 **leaf 내부에서 합산한 누적 regret**이다. 정보집합 수나 방문 횟수로 나누지 않는다.
음수도 저장하며 clipping은 결합 후 regret matching의 양수 부분을 취할 때만 적용한다.

Tree 사이 결합은 다음 평균을 제안한다. 이 정규화는 assistant의 설계 제안이며,
사용자가 원래부터 평균을 명시했다고 간주하지 않는다.

$$B_e(x,a)=\frac{1}{M_e}\sum_{j=1}^{M_e}R_{e,j,\operatorname{leaf}_{e,j}(x)}(a).$$

같은 증분이 M개 tree에 각각 저장되므로 M으로 나누어 tree 수가 많다는 이유만으로
해당 구간의 영향이 커지는 것을 막는다. Leaf 내부의 합을 평균으로 바꾸는 것은 아니다.
Tree 수가 구간마다 다르고 현재 구간의 exact regret과도 결합되므로, 합과 평균은 일반적으로
서로 교환 가능한 설정이 아니다. LightGBM의 leaf prediction이나 boosting 계수를 regret 가중치로 쓰지 않는다.

현재 구간이 E일 때 기존 구간 가중치와 exact residual의 계약을 유지한다면:

$$R_{\mathrm{eff}}(x,a)=\sum_{e<E} e B_e(x,a)+E\Delta R_E(I,a),$$
$$\sigma(a\mid I)=\frac{[R_{\mathrm{eff}}(x,a)]_+}
{\sum_{b\in A(I)}[R_{\mathrm{eff}}(x,b)]_+}.$$

분모가 0이면 합법 행동에 균등 분포를 쓴다. 구간 종료 후 `Delta R_E`만 새 leaf들에 넣고
exact residual을 비우며, 과거 구간 가중치나 이전 regret을 이 저장에 중복 포함하지 않는다.
새 frozen leaf 정책은 압축된 근사이므로 경계 직전 exact 정책과 반드시 같지는 않다.

### 일반화와 남는 편향

이 결합은 다음 유사도 커널로 해석할 수 있다.

$$K_e(x,x_i)=\frac{1}{M_e}\sum_j
\mathbf1\{\operatorname{leaf}_{e,j}(x)=\operatorname{leaf}_{e,j}(x_i)\},\qquad
B_e(x,a)=\sum_i K_e(x,x_i)\Delta r_{e,i}(a).$$

즉, 모든 leaf가 일치해야 하는 기존 방식 대신 일부 tree에서 같은 leaf에 들어간 상태도
기여한다. 새로운 조합의 조회 실패는 없어지지만, 넓은 leaf에서 서로 다른 전략이 섞이고
많이 방문한 영역의 regret이 더 크게 반영되는 편향은 남는다. Exact CFR 수렴 보장은 주장하지 않는다.
Leaf별 support 수는 진단용으로 남기되 방문수 정규화는 별도 실험으로 분리한다.

## 4. 구현 범위와 검증 기준

- 새 모델 format을 분리한다. 과거 `EPOCH7V1`을 새 leaf 모델로 읽거나 덮어쓰지 않는다.
- 각 tree에 split nodes와 `leaf_id -> 8-action regret vector`를 저장한다.
  저장량은 joint signature 수가 아니라 전체 leaf 수에 비례한다. Raw 학습 행 크기는 별개다.
- Tree가 하나도 만들어지지 않은 상수 label 그룹은 leaf 하나인 상수 tree로 취급한다.
- 학습된 그룹의 유효 입력은 모든 tree에서 leaf에 도달하고 해당 regret을 찾을 수 있어야 한다.
  아예 학습되지 않은 legal-mask 그룹은 별도의 `group_missing`으로 기록한다.
- Python fitting/export, C++ load/query, release 시 mass check, root-gap 평가기의 coverage와
  LBR/리그의 공통 loader를 함께 검증한다. `matched_epoch_models`의 새 의미도 문서화한다.
- 각 tree에서 leaf regret을 전부 더하면 구간의 해당 그룹 증분 합과 같아야 한다.
- 새 leaf 조합도 조회되는지, 같은 tree 복제 시 평균 결합이 변하지 않는지 검사한다.
- 음수 보존, 합법 행동만 사용, 확률합 1, 상수 tree, 저장/로딩 왕복, 과거 형식 호환을 검사한다.

기존 teacher 수집 시점의 좌석 비대칭은 별개다. 이번 수정에 몰래 섞지 않고, 수정 전후 비교와
수집 시점 대칭화의 실험을 구별한다. 평균 teacher를 구간 끝의 현재 정책으로 바꾸는 것도 별도 변경이다.

## 5. 다음 실행 조건과 현재 상태

설계 확인 후 leaf 구현·검증을 끝내고, 새 디렉터리에서 refined / soft CE 각각
`hands=100000, epoch_hands=10000`으로 처음부터 학습한다. 1k 간격 파일럿을 이어 돌리지 않는다.
기존 atlas, 규칙, seed 7, tree 예산, 비교군을 유지하고 매 핸드 Local BR-gap과
경계 audit, 최종 독립 평가, leaf support, 저장량을 기록한다. Local gap과 실제 강함은 구분한다.

**이번 작업에서 완료한 것:** 과거 모델 감사, refined 경계 수정 및 테스트, 이 설계와 보고서 정정.
**미실행:** leaf-wise learner 구현, 수정 방식의 100k 학습, 새 LBR/리그.

## 6. 원안과의 정밀 대조: 구간과 Boosting Tree를 구별한다

사용자가 알고리즘 순서를 다시 제시하고 `1*R_1+...+t*R_t`의 구간별 선형 가중치를 재확인했다.
여기서 k는 **10k핸드마다 만드는 abstraction의 번호**다. LightGBM 내부 tree 번호 j와 다르다.

현재 구현은 구간 k마다 여러 `T[k,j]`를 만들고 그 구간 내부의 leaf ID 조합을 하나의 bucket으로 쓴다.
구간 k와 k+1의 partition을 다시 교차하거나, 옛 regret을 새 구간 table에 복사한 것이 아니다.
Assistant의 앞선 설명을 전체 학습 흐름이 다른 것으로 읽을 수 있었다면 범위가 과도했다.

| 원안의 단계 | 현재 코드에서 확인한 동작 |
|---|---|
| 이전 abstraction에서 위치 조회 | 구간별로 독립 조회. 다만 각 구간 내부는 joint leaf signature |
| 과거 regret의 1..t 가중합 | `Ensemble::past`에서 이미 수행 |
| 임시 regret으로 순간전략 계산 | `policy`에서 과거 합 + 현재 구간 exact residual에 regret matching |
| 새 순간 후회 따로 보관 | `Entry::regrets += Q(a)-V`; 과거 합은 `Entry::past`와 별도 |
| 새 abstraction에는 새 누적 후회만 저장 | `dump`는 `entry.regrets`를 쓰고 Python은 그 값만 bucket에 합산 |
| 평균전략 보존 | `strategy_sum`을 구간 내에 쌓지만 raw rows에는 정규화 teacher만 저장. Frozen 모델에는 전략 누적량 없음 |
| regret와 평균전략을 기준으로 분할 | 현재 supervised split target은 teacher 정책 확률뿐. Regret은 분할 후 집계값이지 fitting target이 아님 |
| 현재 정보집합에서 여러 번 local search | 현재는 새 5th-street deal에서 플레이어별 traversal 한 번씩. 개별 온라인 query를 반복 푸는 구현은 아님 |

### 가중합과 이번 구간의 증분

사용자가 확정한 과거 부분은 다음과 같다. `B_k`는 구간 k의 abstraction에서 조회한 regret이다.

$$P_t(I,a)=\sum_{k=1}^t k B_k(I,a).$$

이것으로 시작해 새 후회 `r_n(I,a)`를 구하고, 별도 `D(I,a) += r_n(I,a)`에 저장하는 것은 일치한다.
현재 구현의 실제 정책 입력은 `P_t+(t+1)*D`이다. 원안을 일반적인 warm start로 읽으면 `P_t+D`도 가능하다.
사용자가 이번에 확정한 것은 이전 tree의 1..t 가중치이며, **아직 생성되지 않은 현재 구간 D에 즉시 t+1배를
할지**는 별도 항목이다. 위 3절 수식을 원안의 자동 확정으로 취급하지 않는다. 재실행 전 명세에 표시해야 한다.
평균전략에 어떤 시간 가중치를 쓸지도 regret 가중치와 구별한다.

### 현재 방법은 External-Sampling MCCFR형이며 CFR-D가 아니다

코드는 shuffled deal로 chance를 샘플링하고, 갱신 대상 플레이어의 모든 합법 행동을 전개하며,
상대 행동은 현재 전략에서 하나 샘플링한다. Signed regret을 유지하고 CFR+의 누적 clipping은 하지 않는다.
이것은 external-sampling MCCFR에 epoch 압축과 가중 regret prior를 붙인 근사 알고리즘이다.
표준 MCCFR의 수렴 정리를 이 반복 압축 방식에 그대로 적용할 수 있다는 증명은 없다.
또한 매 핸드 2회 traversal이지 그 정보집합에서 충분히 수렴할 때까지 재탐색하는 방식은 아니다.
[MCCFR 원 논문](https://papers.nips.cc/paper_files/paper/2009/file/00411460f7c92d2124a67ea0f4cb5f85-Paper.pdf)

CFR-D는 trunk/subgame 분해와 경계 counterfactual value를 처리하는 별도 알고리즘이다.
현재 코드에는 그 분해가 없다. Perfect recall만 유지한다고 임의의 한 정보집합 밑에서 다시 푸는 것이
안전한 subgame solving이 되지는 않는다. Root의 가능한 hidden histories 및 도달 분포,
양 플레이어의 정보집합 경계와 counterfactual 값 조건도 다뤄야 한다.
[CFR-D 원 논문](https://poker.cs.ualberta.ca/publications/aaai2014-cfrd.pdf)

### Perfect Recall: 조회 가능성과 학습 보장은 다른 문제

특징이 자신의 관측 정보만 사용하고 routing이 정의되면 imperfect-recall 압축이어도 leaf 조회는 가능하다.
Perfect recall은 자신의 과거 정보집합과 행동 순서를 잊지 않는 조건이며, legal-action-mask 일치만으로는 부족하다.
이 코드의 `exact_key`는 관측 카드와 전체 베팅 history를 유지하므로 아래 반례들을 구별한다.
반면 abstraction은 34개 요약 feature를 나누므로 전체 history를 보존하도록 제약하지 않는다.

기존 1M의 첫 모델과 마지막 모델에서 실제로 반례를 찾았다. 마지막 모델의 group 448에서는:

- 경로 A: actor 1이 BBING, actor 0이 HALF, 다시 actor 1의 차례.
- 경로 B: actor 1이 QUARTER, actor 0이 HALF, 다시 actor 1의 차례.
- 두 상태는 같은 저장된 joint bucket이며, actor 1의 과거 자기 행동은 다르다.
- 실제 seed와 합법 행동으로 재생해 같은 bucket과 서로 다른 `exact_key`를 확인했다.

따라서 **검사한 개별 abstraction을 게임의 정보집합 분할로 해석하면 imperfect recall**이다.
전체 100개 모델의 조합이 모든 차이를 회복하는지, exact key가 전체 게임에서 완전한지는 이번 유한 진단으로
증명하지 않았다. Exact 정보집합에서 탐색하고 leaf는 regret 초기값/근사치 조회에만 쓰는 것과,
leaf를 탐색 노드 ID로 써서 서로 다른 기억을 합치는 것을 구별해야 한다.
전자는 실행 가능하지만 반복 압축 오차가 남고, 후자는 표준 perfect-recall CFR 보장을 바로 쓸 수 없다.
Imperfect recall이면 무조건 실패하는 것은 아니며 well-formed 등 추가 조건 아래의 결과가 있으나,
여기의 임의 LightGBM 분할이 그 조건을 만족한다는 검증은 없다.
[Imperfect recall과 CFR 연구](https://icml.cc/2012/papers/58.pdf)

[실제 반례와 재현 명령](data/recall_audit_20260920/README.md)

### 평균전략은 정규화 전 누적량을 보존한다

Full CFR에서 평균전략을 위한 기본 누적량은 자신의 reach로 가중한 전략합이다.

$$S(I,a)=\sum_n w_n\pi_i^{\sigma_n}(I)\sigma_n(a\mid I),\qquad
W(I)=\sum_n w_n\pi_i^{\sigma_n}(I),\qquad \bar\sigma(I,a)=S(I,a)/W(I).$$

MCCFR에서는 sampling 방식에 맞는 누적 추정기를 사용한다. 현재 코드에서 explicit reach 곱이
없다는 사실만으로 틀렸다고 판단하지 않는다. External sampling에서 비갱신 플레이어의 방문 시 전략을
더하는 누적 방식과 그 기대값을 먼저 확인해야 한다. 압축/재시작/구간 가중치의 효과는 별도 문제다.

새 보관 대상은 각 exact 정보집합의 **해당 구간 증분 `Delta R`, 전략합 `Delta S`, 그 mass `Delta W`**다.
평균 정책만 남기고 mass를 버리면 서로 다른 구간의 평균을 정확히 재결합할 수 없다.
현재 raw row의 `visits`는 갱신·비갱신 방문을 모두 세므로 strategy mass와 같다고 볼 수 없다.
Leaf에는 새 구간의 이 통계들을 별도로 집계하고, 과거 전략합을 복사하지 않는다.
`Delta W=sum_a Delta S(a)`로 재구성 가능한 규약이면 중복 저장할 필요는 없지만 의미는 명시한다.
이것도 leaf 압축 후에는 원래 exact 정보집합의 평균전략에 대한 근사다.

## 7. 첫 Leaf별 실험에서 채택한 계약

사용자가 후보 중 실험 하나를 선택하도록 위임하여 refined 100k/10k를 먼저 실행한다.
Soft CE, K-means, 100k 간격 변경은 동시에 섞지 않는다.

- 구간 내 tree 평균, 구간 k의 가중치 k, 현재 exact 증분 가중치 t+1을 사용한다.
  이는 본 실험의 명시적 선택이며 사용자 원안이 유일하게 이 정규화를 요구했다는 의미가 아니다.
- 구간별 새 regret과 정규화 전 strategy sum만 각 tree의 leaf에 합산한다.
  leaf 내부는 행 평균이 아니다. 모든 leaf의 합에 대한 보존 검사를 regret/strategy 각각 수행한다.
- 새 raw format은 `ROWS7V02`(340 bytes/row), inference format은 `EPOCH7L1`이다.
  원래 `ROWS7V01`/`EPOCH7V1`과 구별한다. 과거 모델을 자동 변환하거나 덮어쓰지 않는다.
- 평균전략 조회: leaf의 전략합을 tree 평균, 구간 선형 가중합한 뒤 행동별로 정규화한다.
  mass가 0이면 균등정책이며 legacy 모델에는 이 평균전략이 없다고 명시적으로 거부한다.
- 학습되지 않은 그룹은 zero contribution; 학습된 그룹은 모든 leaf payload를 저장한다.
  평가의 `matched_epoch_models`는 L1에서 해당 root 그룹을 가진 모델 수다.
- 200핸드/100간격 smoke, Python 5개 검사, C++ self-test/engine/spool 검사를 통과했다.
  신규 joint 조합, signed regret, tree 복제 불변성, 평균전략 정규화 및 legacy 조회를 검사했다.
- Teacher 수집 시점과 fixed player traversal 순서는 유지했다. Recall 보장이나 수렴 보장을
  새로 얻은 것이 아니며, 6절의 한계는 계속 적용된다.
- 완료한 독립 평가의 current gap은 0.3259 대 Hard-256 1.5417이지만,
  average gap은 3.0042 대 2.5403이다. Actor 1 시작 그룹은 10개 모델 모두 단일 leaf다.
  상세한 조건·비용·한계와 그래프는 상단 실험 보고서에 보관했다. Soft CE/K-means는 미실행이다.
