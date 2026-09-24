# cpp_mccfr — 리팩터링 · 확장성 · 변경사항

> 2026-09-17 갱신: 아래는 과거 시점의 진단/계획이다. 현재 정식 경로는
> `agents/cpp_mccfr/`, 생성물은 `data/`와 `bin/`, 공유 게임 규칙은
> `environments/seven_stud/stud_rules.hpp`로 분리했다. 현재 소스를 static
> libstdc++/libgcc로 새로 빌드한 self-test는 통과했다.
> 최신 구조는 프로젝트 `PROJECT_STRUCTURE.md`, 검토는 상위
> `EXPERIMENT_REVIEW_2026-09-17.md`를 따른다.

> 이 문서는 (1) 이번 세션의 변경사항, (2) 현재 코드의 정직한 진단, (3) 확장성을
> 위한 목표 아키텍처와 단계별 마이그레이션 계획을 담는다. 코드 자체는 잘 동작하지만
> **단일 모놀리스라 확장이 막혀 있고**, 품질 게이트(`--self-test`)가 깨져 있다.

---

## 0. 이번 세션 변경사항 (Changelog)

| 구분 | 내용 | 상태 |
| --- | --- | --- |
| 확장성 | `stud_mccfr.cpp`의 `main`을 `#ifndef STUD_MCCFR_NO_MAIN` 로 가드 | ✅ 완료·검증 |
| 정리 | 출력물 이동: `logs/`(로그 97개), `results/`(LBR/heuristic 결과 28개) | ✅ 완료 |
| 발견·해결 | 모든 exe 가 즉시 세그폴트 → 원인은 **환경의 깨진 libstdc++ iostream**(프로젝트 코드 무결). `-static-libstdc++ -static-libgcc` 로 빌드하면 `--self-test` 통과 | ✅ 원인규명·우회 검증(§4) |
| 계획 | `src/ models/ bin/ docs/` 재배치 + README 경로 갱신 | 📋 미실행(§6) |

가드 외에는 기존 소스 로직을 건드리지 않았다. 정상 빌드는 그대로 동작한다(아래 검증).

---

## 1. 현재 상태 진단

- `stud_mccfr.cpp` **8,639줄** 단일 파일 + 보조 5개(`stud5_mccfr`, `stud5_team_lbr`,
  `deep_cfr_*`). 폴더에 **421개 파일**(bin 175, exe 39, json 72, log/err/progress
  97, txt 17, cpp 6, md 3).
- 전체가 하나의 익명 namespace 안에 있고, `main` 하나가 거대한 `Options` 파서로
  수십 개 모드를 분기한다.
- **확장성 문제:** 새 프로브/에이전트(예: multi-street LBR)를 추가하려면 이 8천 줄
  파일을 직접 수정해야 한다. 파일 경계·헤더가 없어 재사용이 불가능하다.
- **품질 게이트 붕괴:** `.\stud_mccfr.exe --self-test` 가 SIGSEGV(139)로 죽는다(§4).
- `.exe` 39개는 같은 소스를 시점마다 다른 이름으로 빌드한 **낡은 스냅샷**이다
  (`*.bin`, `*.exe` 는 `.gitignore` 됨).

---

## 2. 모놀리스 모듈 지도

파일은 이미 논리적으로 분리 가능한 블록으로 구성돼 있다. 줄 번호는 근사치다.

| 줄 범위 | 논리 모듈 | 핵심 심볼 |
| --- | --- | --- |
| ~189–940 | **engine** (게임) | `Card` `Player` `Event` `State`, `valid_mask` `apply_action` `terminal_net_search`, `best_hand`/`evaluate_five`, `fresh_deck`, `heuristic_strength` |
| ~940–1260 | **infokey** | `InfoKey` `InfoKeyHash`, `make_power_key` |
| ~1261–1930 | **abstraction** | `PowerAtlas`, `PowerObservationKey`, `ActionRangeModel` |
| ~1926–2360 | **h4** | `H4Policy` `H4CFR` `H4QPolicy`, `make_h4_view` |
| ~2360–4160 | **solver** | `MCCFR`(핵심), `RegretNode`, `train_root`, `play_hand`, `imitate_policy` |
| ~4164–4535 | **asymp** | `BucketAsymP`, `BehaviorGradient` |
| ~4535–5260 | **probes/agents** | `BeliefBR`, `PolicyLBR` ← 새 에이전트가 들어갈 자리 |
| ~5262–5370 | **baselines** | `HeuristicPolicy` `HalfPolicy` `ConditionalParticipationPolicy` |
| ~6027–7880 | **cli** | `Options`, `parse_options`, `run_*`, `self_test` |
| 7883–end | **main** | 모드 분기 |

---

## 3. 확장성 브리지 (완료·검증)

새 에이전트/프로브를 **모놀리스 수정 없이** 별도 파일로 추가할 수 있게 하는 최소
변경을 넣었다. `main` 만 가드했으므로, 새 TU가 엔진 전체를 include 로 재사용한다.

```cpp
// my_probe.cpp — 엔진 전체(State, 규칙, MCCFR, BeliefBR, PolicyLBR, atlas...) 재사용
#define STUD_MCCFR_NO_MAIN
#include "stud_mccfr.cpp"
#include <cstdio>

int main() {
    // 여기서 새 프로브/에이전트 구현. 모든 엔진 심볼 접근 가능.
    State state; state.ante = 1; state.street = 5;
    printf("valid=%u\n", valid_mask(state, 0));
    return 0;
}
```

```powershell
g++ -O2 -std=c++17 my_probe.cpp -o my_probe.exe   # stud_mccfr.cpp 는 안 건드림
```

익명 namespace라 include 하는 TU마다 내부 링크 사본을 가지므로 ODR 충돌이 없다.
이 브리지는 §4의 self-test 세그폴트를 **모놀리스를 건드리지 않고** 구간까지 좁히는
진단 프로브를 짜는 데 실제로 사용했다. 이것이 브리지의 유용성을 그대로 보여준다.

> 이 방식은 **다리(bridge)** 이지 최종 설계가 아니다. 8천 줄 include 는 컴파일이
> 느리고 결합이 강하다. 최종형은 §5의 헤더 분리다. 새 실험을 지금 당장 붙일 때는
> 브리지를, 구조를 고칠 때는 §5·§6을 따른다.

---

## 4. ✅ 해결됨: 세그폴트의 원인은 코드가 아니라 **환경의 깨진 libstdc++**

> **정정.** 이 절은 원래 "self-test 세그폴트 = 코드 결함", "-O3 빌드가 깨짐 = UB"
> 라고 기록했었다. **둘 다 틀렸다.** 그 판단은 (a) 파이프라인 종료코드를 프로그램
> 종료코드로 잘못 읽은 측정 오류와 (b) 잘못된 귀인에서 나왔다. 실제 원인은 아래와
> 같고, **프로젝트 코드에는 이 결함이 없다.**

### 4-1. 증상

`stud_mccfr.cpp` 로 빌드한 모든 exe 가 인자 없이도 즉시 SIGSEGV(139), 출력 0.
`--self-test`, `--bucket power`, `--bucket legacy` 전부. 최적화 레벨 무관
(-O0/-O2/-O3). 7월에 빌드된 기존 exe 들도 지금 실행하면 똑같이 죽는다.

### 4-2. 진짜 원인

프로젝트 코드가 하나도 없는 16줄 프로그램으로 재현된다.

```cpp
#include <cstdio>
#include <iostream>
int main(){
    std::printf("1: printf works\n"); std::fflush(stdout);
    std::cout << "2: cout works\n";   // <-- Segmentation fault
}
```

```text
1: printf works
Segmentation fault (139)
```

**이 환경의 C++ iostream(libstdc++ 런타임)이 깨져 있다.** `printf`(msvcrt)는
멀쩡하고 `std::cout`/`std::cerr`/`std::ofstream` 만 죽는다. PATH 상의 다른
`libstdc++-6.dll`(다른 툴체인에서 온 것)이 MinGW 것을 가리는 전형적인 DLL
불일치다. iostream 은 전역 초기화/locale 상태를 쓰기 때문에 불일치 시 즉사한다.

이 하나로 모든 관측이 설명된다.

- exe 가 출력 0 으로 즉사 → 첫 `std::cout`/`std::cerr` 사용 시점에 크래시
- 7월 exe 도 지금 죽음 → 코드가 아니라 지금의 런타임이 문제
- `PowerAtlas::save` 크래시 → `std::ofstream`
- 브리지 진단 프로브는 전부 정상 → `printf` 만 썼기 때문

### 4-3. 해결책 (검증 완료)

정적 링크하면 즉시 해결된다.

```powershell
g++ -O2 -std=c++17 -static-libstdc++ -static-libgcc cpp_mccfr\stud_mccfr.cpp -o cpp_mccfr\stud_mccfr.exe
.\cpp_mccfr\stud_mccfr.exe --self-test
# {"self_test":"ok","buckets":516,"power_buckets":15}
```

**`--self-test` 가 통과한다.** 즉 4-1 의 모든 증상은 사라지고, 기존 결과 `.bin`/
`.json` 의 신뢰성을 의심할 이유도 없다.

> **권고:** README 의 빌드 명령에 `-static-libstdc++ -static-libgcc` 를 추가할 것.
> (또는 PATH 에서 올바른 MinGW `bin` 이 먼저 오도록 정리.) 이것이 이 저장소에서
> 가장 값싸고 효과 큰 수정이다.

### 참고: 브리지로 좁혀 간 과정

브리지 진단 프로브로 `self_test`(6745~)의 앞부분을 재현했을 때 아래는 전부
통과했고(=엔진은 멀쩡), 이 대비가 원인을 iostream 으로 좁히는 결정적 단서였다.
  - 핸드 평가(`evaluate_five`/`best_hand`/`best_after_*`), 규칙(`valid_mask`/
    `apply_action`), 정책(`ConditionalParticipationPolicy`/`HalfPolicy`),
    heuristic variants, `RegretNode`, `MCCFR`+`play_hand`(5·7구),
    `collect_power_samples`, `PowerAtlas::fit`/`fit_hard_em`,
    `collect_lbr_power_samples`, `sample_fifth_street_root`/`deep_cfr_tensor`,
    `imitate_policy`.
  - 이 블록들은 `printf` 만 쓰므로 통과했고, exe 는 첫 iostream 사용에서 죽었다.
    이 대비가 곧 원인이었다.

> **검증 게이트 (복구됨):** `-static-libstdc++ -static-libgcc` 로 빌드한 뒤
> `--self-test` 를 쓴다. 통과가 확인됐다.
>
> **교훈 (측정 규약):** 종료코드를 파이프라인 뒤에서 읽지 말 것.
> `prog | tail; echo $?` 는 `tail` 의 종료코드라 세그폴트를 성공으로 오독한다.
> 반드시 `prog > out.txt 2>&1; echo $?` 형태로 읽는다. 이 함정 때문에 이 문서의
> 이전 판이 "-O2 정상 / -O3 크래시" 라는 틀린 결론을 기록했다.

---

## 5. 목표 아키텍처 (확장성 설계 보완)

모놀리스를 **헤더 선언 + 구현 + 얇은 CLI**로 분리한다. 목표 레이아웃:

```text
cpp_mccfr/
  src/
    engine.hpp/.cpp        # Card/Player/State, 규칙, best_hand, deck, heuristic
    abstraction.hpp/.cpp   # InfoKey, PowerAtlas, ActionRangeModel, make_power_key
    h4.hpp/.cpp            # H4Policy 계열
    solver.hpp/.cpp        # MCCFR, RegretNode, play_hand, imitate, BucketAsymP
    probes.hpp/.cpp        # BeliefBR, PolicyLBR, baseline 정책  ← 확장 지점
    options.hpp/.cpp       # Options + parse_options
    cli_stud_mccfr.cpp     # main (얇은 드라이버)
  agents/                  # 새 에이전트/프로브: engine 등 헤더에 링크
    multi_street_lbr.cpp   # 예: 1-ply LBR의 multi-street 일반화(§7)
  tests/
    self_test.cpp          # 분해된 단위 테스트(현 self_test를 블록별로)
  models/  results/  logs/  docs/
  README.md  REFACTORING.md
```

핵심 원칙:
- **환경 경계 최소화**: `IMPLEMENTATION_GUIDE.md` 가 이미 4개 연산
  (`valid_mask`/`apply_action`/`terminal_net_search`/`make_information_key`)로
  경계를 정의했다. 이 4개를 `engine.hpp` 의 공개 API로 고정하면 solver·probe가
  엔진 세부에 의존하지 않는다.
- **probe/agent 는 플러그인**: `probes.hpp` 의 인터페이스(예: `Action choose(State,
  seat)`, `TargetPolicy`)만 구현하면 새 에이전트가 붙는다. 새 파일 하나 = 새 실험.
- **CLI 는 얇게**: 거대한 `Options`/`main` 은 각 모드를 서브커맨드 함수로 쪼개고,
  드라이버는 등록·분기만 담당한다.
- **테스트 분해**: 지금의 통짜 `self_test` 를 블록별 단위 테스트로 쪼개면 세그폴트가
  즉시 국소화되고, 회귀가 조기에 잡힌다(§4 문제의 근본 예방).

헤더 분리 자체는 동작을 바꾸지 않는 순수 리팩터이므로, 각 모듈 추출마다 "빌드 +
실제 명령 exit 0" 게이트를 통과시키며 점진적으로 진행한다.

---

## 6. 마이그레이션 계획 (단계별, 검증 게이트 포함)

**Phase 1 — 완료**: `logs/`·`results/` 정리 + 확장성 브리지.

**Phase 2 — 산출물/소스 폴더 재배치 (README 경로 갱신 동반).**
`*.bin`·`*.exe` 는 gitignore 되어 VCS 영향이 없다. 아래는 즉시 실행 가능한 이동이며,
실행 후 README 의 `--load`/`--save`/`-o`/`.\...exe` 경로를 새 폴더로 갱신해야 한다.

```powershell
mkdir models, bin, docs -Force
Move-Item *.bin models\ ; Move-Item *.exe bin\
Move-Item IMPLEMENTATION_GUIDE.md, BUCKET_GROWTH.md docs\   # README 는 최상단 유지
```

> README 갱신이 병행돼야 하므로 이 커밋에서는 실행하지 않았다. 승인하면 이동 +
> README 경로 일괄 갱신을 한 번에 처리한다(참조 .bin 은 10~15개 수준).

**Phase 3 — 헤더 분리 (§5).** 의존성이 가장 적은 `engine` 부터 추출한다:
`engine.hpp/.cpp` → 빌드·스모크 → `abstraction` → `solver` → `probes` → `cli`.
각 단계마다 게이트 통과 필수. `self_test` 는 `tests/`로 옮기며 블록별로 분해한다.

---

## 7. 정리 체크리스트 (우선순위)

1. ⭐ **`--self-test` 세그폴트 수정** (§4). 깨진 게이트는 다른 모든 검증을 무력화.
2. **`self_test` 분해** → `tests/` 블록별 단위 테스트(재발 방지 + 국소화).
3. **`.exe` 39개 정리**: 낡은 스냅샷 제거, 각 변종의 **재빌드 레시피**를 문서화
   (소스 하나에서 플래그/모드로 재생성 가능해야 함).
4. **거대 `Options`/`main` 분해**: 모드별 서브커맨드 함수로.
5. **헤더 분리**(§5)로 새 에이전트가 모놀리스를 안 건드리게.
6. 예시 확장으로 **multi-street LBR**(1-ply `PolicyLBR::action_ev` 일반화)을
   `agents/` 에 추가 — 단, 순진한 rollout 은 optimizer's curse 로 exploit 를 못
   올리고 분산만 키운다(Python PoC 확인). 다중 continuation·분산감소가 필수.
