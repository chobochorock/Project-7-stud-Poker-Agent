# CLAUDE.md

Read `AGENTS.md` too (shared workspace rules and the report format for abstraction comparisons).

## 1. 사용자의 ASWISS 구상 (원문 그대로, 2026-11-15)

사용자: "앞으로 이것은 중요한 구상이니 claude.md에 적어놓든가 하도록". ASWISS와 prior–posterior의 설계, 실험, 보고의
기준이다. 아래 원문은 사용자의 지시 없이 고치지 않는다. 구현이 원문과 다르면 다른 점을 "내가 정한 것"으로 따로
보고한다(지금 구조와의 차이: `agents/adaptive_buckets/PP_STUD7_DESIGN_KO.md` 20절).

* 지표
   * bucket 수
   * 작은 게임에서
      * exploitability
      * bucket의 regret합산(=\sum bucket.regret) 및 exploitability - \sum bucket.regret
         * 전자는 앞으로의 개선 가능성을 보여주고, 후자는 그 개선에서 추상화 오차가 어느정도 되는지를 확인하는 지표다.
      * CFR(또는 MCCFR)의 수렴 정도
   * 큰 게임에서
      * Gap proxy (확실한 상한이라고 claude가 보장 \to 나중에는 직접 검토해 봐야할 것)
      * 매 핸드 학습 전 작은 particle LBR 평가 이동평균(log 적으로 분할 구간마다 이동 평균 구간도 2배씩 늘려야 할 것 같다)
      * 분할 직전 큰 particle LBR 평가
      * CPU 시간에 따른 데이터 변화
      * k-means abstraction기법과의 차이: k256, k1024와의 비교
      * 가장 최신 snapshot에 대한 최대한의 particle LBR평가 (또는 초반 call도 particle로 평가)
* 알고리즘
   * 순수 ASWISS
      * 초반 거친 (posterior) abstraction
      * 분할 시점 기준 전까지 MCCFR로 추상화 게임 전략 개선
      * 분할 시점: 전체 \sum bucket.regret <\varepsilon 또는 T_{k}=2^{k}\times 1000마다 분할
      * 분할 기준: 특정 행동 a에 대한 R(I, a)/\sqrt{ r^{2}(I, a) } 또는 \lvert r(I,a)-\mu_{R(B, a)} \rvert>\sigma_{R(B,a)}(bucket의 분산도보다 큰 오차에 대한 분할 및 할당)
      * 이 때에 순간 전략들의 합을 통한 r(I, a)를 구하여야 한다.
         * 만약, 평균 전략을 활용한다면, 그 i번째 분할 시점 \tau_{i}에 대해서 r^{\bar{\sigma}^{\tau_{i}}}(I, a)=v^{\bar{\sigma}^{\tau_{i}}}(I, a)-v^{\bar{\sigma}^{\tau_{i}}}(I)로 구한다.
         * 그러나 이런 순간 전략의 업데이트 방향(regret)과 평균 전략의 방향이 다를 수 있는 반례가 존재한다고 한다. (이에 대해서는 검토해 볼 것)
         * 또한, 전부 정확한 가치를 훑어볼 수 없는 경우가 발생할 수 있어 분산이 높아져매번 다른 bucket에 할당될 수 있다.
            * 즉, 오직 샘플링에 대해서 온전히 sampling할 수 있어야 같은 bucket에 할당함을 보증할 수 있다.
            * 큰 게임에서는 보증할 수 없다.
   * prior-posterior abstraction
      * prior abstraction은 정보집합의 데이터를 통해서 정보집합을 분리한다.
      * posterior abstraction은 순수 ASWISS의 추상화를 일컫는다.
      * 각각의 거친 추상화를 받는다.
         * 특히나 prior abstraction은 각 정보집합간의 거리가 정확하게 정해지지 않는다. 따라서, 오직 같은 time step을 가진 정보집합들을 묶는 거친 추상화를 사용한다.
         * 이는 posterior의 거친 추상화와 유사해 보이나, posterior는 오직 동일한 constraint에 대해서 거친 추상화를 진행한다.
         * 그런데 이런 구조라면, prior abstraction은 동일하지 않은 합법 행동집합을 가지는 경우가 발생할 수 있다.
         * 이를 방지하기 위해서 prior는 예외적으로 합법 행동집합의 동일성까지 조건으로 달아야겠다.
      * 이렇게 posterior abstraction 기반에서 전략을 업데이트한다.
      * prior bucket은 샘플링을 한 뒤에 그 샘플이 posterior에 적당한 bucket에 할당될 경우 앞으로 그 bucket을 가리키는 bucket으로 정한다.
         * 즉, prior bucket에서 처음 샘플된 표본이 해당 prior bucket의 posterior bucket연결을 결정한다.
      * posterior의 분할은 그 regret양이 0에 수렴하는 경우 또는 일정 구간이 지나간 이후 분할한다.
         * 이후 분할 할당은 순수 ASWISS와 거의 동일하나, 정보집합 개개에 대해서가 아니라 그 정보집합을 담고 있는 prior bucket에 대해서 진행한다.
      * 이 경우 나중에는 posterior bucket에 할당할 prior bucket이 부족해지는 경우가 발생한다. 이 때 prior bucket을 분할한다.
         * 기본적으로 decision tree로 분할하며, label을 실제 샘플을 분할 전 전략을 통한 regret vector나 value vector간의 유사도(유클리드 거리?)를 통한 분리를 진행한다.
         * TARA가 좋은 예시가 될 수 있으려나? 잘 모르겠다 이는 검토해 봐야겠다.
         * 이후 모든 prior bucket들의 posterior bucket 재할당을 위한 탐색 기간을 준다.
      * 이를 반복한다.



* 단일 트리 실험
   * 지금 누적 regret을 통한 기준을 설립한 적이 없다. 지금 한 것은 분할 시점에 순간 전략들(또는 평균 전략)에 대한 업데이트 방향(argmax regret action)이다.
      * 기존 ASWISS 분할은 posterior처럼 하고 있으나, 매번 업데이트 방향을 계산해야 하는 약점을 안고 있다.
      * 그래서 decision tree의 분할 라벨로 이 업데이트 방향을 줌으로써 prior-posterior의 병합을 사용하였다.
      * 그러나 성능은 별로 좋지 못하였다.
   * (제안): 그래서 단일 트리에서는 현재 leaf에서 미리 분할한 pre-bucket을 생성해서 각 분할에 대해서 누적 regret을 별도로 구한다.
      * 충분히 방문하였을 때 그 누적 regret의 비율의 차이가 크다면, 그 다음 분할 대상으로 삼는다.
      * 아니라면, 다른 분할을 시도해 본다.
* prior-posterior실험
   * prior abstraction에서는 posterior abstraction의 적절한 leaf를 가르키는 포인터를 사용한다.
   * posterior abstraction에서는 전략을 계산하고, 업데이트 방향을 통한 분할을 진행한다.
   * (제안): prior abstraction에서는 추가로 각 leaf에 대해서 누적 regret을 보관한다.
      * 생각을 해보자. 누적 regret을 prior에서 저장한다면, 하나의 prior leaf는 하나의 posterior leaf를 가리킨다.
      * 이 경우 반드시 수렴하는 posterior leaf의 누적 regret은 prior의 것과 괴리가 생길 것이다.
         * 괴리는 정규화된 regret벡터간의 거리로 한다.
         * 괴리가 큰 경우는 posterior가 잘못된 것이다. 이 경우에는 posterior의 분할이 필요하다.
            * posterior 분할 이후 prior leaf의 포인터 재할당시 prior의 누적 regret도 초기화 해야 한다.
         * 괴리가 작은데 prior leaf의 방문이 많은 경우에는 너무 일반화가 되어 있을 확률이 높다. 이 경우 prior leaf의 분할을 진행한다.
         * 이후 사용하지 않는 posterior leaf는 삭제한다. (어차피 사용하지 않으니 상관 없으려나?)
      * 방문이 많다의 기준은 같은 bucket에서 방문률이 50%이상이거나 해당 posterior leaf를 혼자 사용하고 있는 경우라고 하자.
         * 이 leaf들에서 단일 트리에서 사용하던 k기준과 그 누적 regret을 적용해 보는 것을 고려한다.
         * 별 차이 없다면, 그냥 냅둔다.
         * 분할 이후에는 pre-bucket에서 상속받은 누적 regret에서 다시 누적한다.
      * 학습 파이프라인은 다음과 같아진다: \dots\to 학습 \to prior leaf 분할 \to 학습 \to posterior leaf 분할 \to 학습 \to\dots
      * (대안): posterior분할은 실시간으로 해당 leaf가 분할 기준을 충족하면, 분할하는 것도 고려할 만하다. 단, 완전 상속을 기준으로 한다.
      * (대안): posterior leaf에서 누적 regret을 모으지 말고, 전체 prior leaf들의 누적 regret을 각 도달확률 가중을 통한 regret으로 전략을 생성하는 것으로 하는 것을 생각해 볼 수 있다.

## 2. 작업 규칙 (사용자가 이 프로젝트에서 지시한 것)

- 답은 한국어로 한다.
- 실험은 mini-stud에서 먼저 한다. 맞으면 7-stud로 간다.
- 7-stud 비교는 주 실험에 비교군으로 넣는다: `agents/adaptive_buckets/aswiss/stud7/run_100m.ps1`(Linux `run_100m.sh`)의
  `runs_fair`(비교군마다 학습 CPU 24시간, 온라인 LBR 1,000 roots마다 64파티클, 같은 deal, ASWISS `hist_bbf`·k-means·SOTA와
  같은 폴더). 따로 실험 폴더나 드라이버를 만들어 돌리게 하지 않는다. 지표는 위 원문의 "큰 게임" 목록이다.
- 오래 걸리는 실험은 사용자 노트북(Windows, PowerShell)에서 돈다. 백그라운드로 돌고 창을 붙잡지 않는 명령과, 상태를
  계속 볼 수 있는 명령(`-Action status -Watch 60`)을 함께 준다.
- 실행 이름은 고정 ID로 부르고 실험 지도(`PP_STUD7_DESIGN_KO.md` 19.0)에 적는다. 내가 만든 용어를 쓰지 않는다
  (예: "게이트"가 아니라 "대기 조건").
- 사용자의 원안과 대안을 그대로 구현한다. 고치는 방법을 지어내지 않는다. 구현 전에 구조를 먼저 말하고, 정의 문제만
  묻는다. 내가 정한 기본값은 따로 밝힌다. 판단은 사용자가 지표를 보고 한다.
- 보고는 `AGENTS.md`의 형식을 따른다(k-means 기준선 겹치기, 같은 예산의 비율, 평균·현재 정책 분리, 평가 조건 맞추기,
  외삽 금지, LBR은 exploitability가 아님).
