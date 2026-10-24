# Frozen abstraction Policy-LBR 평가

제한형DT/무제한DT/hand-only k-means+abstract-recall의 저장된현재·평균전략을
기존 `PolicyLBR`와 대전한다. 모델끼리풀리그가아니고각정책을공격하는LBR의실제수익을잰다.
학습코드/모델을수정하거나학습하지않는다.

## 방법과 기준

가까운선행연구는Lisý & Bowling의
[Equilibrium Approximation Quality of Current No-Limit Poker Bots](https://arxiv.org/abs/1612.07547).
LBR는고정상대정책의취약성을공략하는근사BR다.이번에는논문의hold'em수치를재현하거나
새알고리즘을제안하는것이아니라기존7-stud PolicyLBR를동일조건으로적용하는통제평가다.

- 게임:C++7-studv3 HU/H4/5~7구/ante1000/stack1000ante,기존유한action집합.
- 기존 `stud_mccfr.cpp`의H4조건부particle posterior/관측된상대행동likelihood/
  one-step fold-equity 및check-down근사를그대로사용한다. 매실제결정에서다시LBR를고른다.
  실제미래패/상대숨은패를정답으로읽지않는다.
- 실제terminal payoff를**LBR 공격자 ante/hand**로기록한다. 양수일수록target에게불리하다.
  Local BR-gap의max-Q를대전수익으로대체하지않는다. 정확exploitability가아니며음수도보존한다.
- 같은shuffle deal을양쪽좌석에서평가한다. 전모델/정책에동일deal/action/LBR seed를쓴다.
  좌석2개는독립표본으로세지않고pair평균을분석단위로사용한다.
- 기본예산:seed2026100401/2026100402,각250pairs,64particles.
  총500pairs=1,000played hands/정책. 결과를보고조기종료하거나예산을늘리지않는다.
- 모든run에model current/average,matched k-means current/average,uniform을넣는다.
  제한형/무제한형의baseline과세run의uniform대전결과는정확히일치해야한다.
- DT두군은1M학습. Hand-recall은저장중단당시예산(이번53601hand)이다.
  따라서모델간같은학습예산효율순위로해석하지않는다.
- DT비교군은hard256+betting-summary+seat. Hand-recall의비교군은현재hand bucket만
  보존하되전체betting history는정확히보존한다. 둘은같은k-means라도서로다른모델이다.
- Atlas는과거100M정책epsilon0.2 self-play100k/seed37301 자료로만든frozenhard256이다.
  기존데이터정책편향은남고이번에는새clustering/abstraction interval변경이없다.

## 실행

학습을일시중단하고실행한다. Hand-recall 평가기는기존engine.lock을독점해새writer와
겹치지않게하고SQLite를READONLY/query_only/read transaction으로연다. DB복제/수정없음.

```powershell
Set-Location D:\Experiment\Project-7-stud-Poker-Agent
python -m agents.cpp_mccfr.abstraction_lbr.run --output agents/cpp_mccfr/data/abstraction_lbr/NEW_RUN
python -m unittest agents.cpp_mccfr.abstraction_lbr.test_evaluation -v
```

Output은새폴더여야한다. `--pairs`, `--particles`, `--seeds`로예산을명시할수있고
`--build-only`는컴파일만한다. 기본세run경로는run.py의RUNS에있다.
평가기재실행은별도output에서수행한다. 중간실행이실패하면기존결과를완료로표시하지않는다.

## 어댑터와 검증

- DT는기존loader로checkpoint를읽고학습용split monitor만RAM에서해제한다.
  파일은그대로보존한다. 로드순간에는무제한checkpoint의약1GB monitor RAM도필요하다.
  추론은기존route/key/regret matching/average 함수를쓴다. 기존Target과반환policy일치를검사한다.
- Hand-recall은기존정보집합key와lossless node codec을재사용한다.
  DB없음과cache miss를구별하며read-onlycache100k/query-cache128MiB를사용한다.
- 기존소스hash에영향을주지않도록빌드폴더에엔진source include를기계적으로생성한다.
  원코드의standalone main이름1개와hand_recall.cpp include경로1개만바꾼다.
  변경anchor개수를검사한다. 원source/manifest의수정이나일괄hash업데이트를하지않는다.
- 각실제모델로합법확률합/숨은정보비참조/LBR비참조/같은seed수익재현/always-fold상대
  1ante수익을검사한후본대전을진행한다. 테스트seed결과는본통계에포함하지않는다.
- 모델파일size/mtime가처음과끝에같은지,source/atlas hash,CSV평균과engine결과,
  딜pair연속성/범위/seed중복/공통대조군결과일치를확인한다.
- LBR missing/ESS는attacker가가설상의정보상태정책을조회한진단이지trainingcoverage가아니다.

## 산출물과 해석

`run.json`에조건/source/binary hash/완료상태를남긴다. 각model+seed폴더의
`paired_payoffs.csv`와`evaluation.json`은실제수익/시간/조회/ESS다.
`summary.json`에는평균,SE,normal95%CI,seed별평균,seed-stratified pairbootstrap95%CI,
tail범위와paired model-minus-control 차이가있다. 큰수익outlier를제거하지않는다.
`lbr_comparison.png`는음수/0을보존하는선형축이다. CI는deal sampling에대한것이며
trainingseed/모델선택/근사LBR의오차까지포괄하지않는다.1,000hands는예비screen이다.

모델이LBR에손해를덜봤다고원게임균형/논문기여를보장하지않고,양의수익이불확실하다고
악용불가능을증명하지않는다. 큰신뢰구간이면더많은독립deal로확인해야한다.
