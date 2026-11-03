# ASWISS 발표 자료

## 발표 자료 (2026-11-03): `aswiss_deck.tex` (LaTeX Beamer)

본문 21장 + 부록 12장. **본문 방법은 정보집합 단위의 구간 누적 후회 라벨로, 업데이트 방향이 갈린 bucket을 그 자리에서 나누는 것**이다(`../../ASWISS_CONVERGENCE_KO.md` 정리 1). 평균 전략 라벨, bucket 이력, 역방향 분할, 원자, 판정 문턱, 혼합 평균은 부록의 트릭(T1–T6)이다.

- `aswiss_deck.tex`: Beamer(metropolis) 원고. 글꼴과 색은 이전 원고와 같다(Latin Modern Sans, 나눔바른고딕). 게임 트리는 TikZ, 그래프는 pgfplots.
- `aswiss_deck.pdf`(33쪽), `aswiss_deck_notes.pdf`(각 슬라이드 뒤에 노트 쪽, 66쪽), `aswiss_deck_notes_KO.md`(노트만, `python notes_md.py`로 다시 만듦).
- `CHANGES_KO.md`: 원래 자료(`aswiss_slides.pdf`)와의 대응, 원본에 없던 내용과 출처, 요청과 다른 점.
- `plotdata/`: `leduc_infoset.dat`(`make_leduc_infoset_data.py`), `ministud_long.dat`(`make_ministud_data.py`), 부록 트릭용 `leduc_conv.dat`·`leduc_gain.dat`(`make_data.py`).
- `pptx_v1/`: 이전 재구성판(bucket 이력 + 역방향 분할을 본문으로 둔 23+10장)의 PowerPoint와 생성 코드. 지금 본문과 맞지 않는다.

```bash
xelatex aswiss_deck.tex && xelatex aswiss_deck.tex          # 두 번 (질문 표시 위치와 쪽 번호)
xelatex -jobname=aswiss_deck_notes "\def\withnotes{1}\input{aswiss_deck}"   # 노트 판, 역시 두 번
```

## 이전 자료: `aswiss_slides.tex` (LaTeX Beamer)

- `aswiss_slides.tex`: Beamer(metropolis 테마) 원고, 16:9. 도식은 TikZ, 그래프는 pgfplots, 표는 booktabs로 그린다. 각 frame에 `\note{}`로 발표자 노트가 있다.
- `aswiss_slides.pdf`: 컴파일한 결과(표지, 본문 16장, 절 구분 4장).
- `plotdata/`: Leduc 그래프 두 장의 데이터. `make_data.py`가 Kuhn·Leduc 실행 결과(`../small/run_small.sh`의 출력, 또는 `../results_data/small_games_summaries.tar.gz`를 푼 폴더)에서 만든다.

## 빌드

```bash
xelatex aswiss_slides.tex && xelatex aswiss_slides.tex   # 두 번 (쪽 번호와 진행 막대)
```

필요한 것: XeLaTeX, beamer와 metropolis 테마, pgfplots, ko.TeX(kotex), 한글 글꼴. Debian/Ubuntu에서는
`texlive-xetex texlive-latex-recommended texlive-latex-extra texlive-pictures texlive-lang-korean fonts-nanum`이다.

## 글꼴 바꾸기

원고 맨 위 "글꼴" 부분의 네 줄만 고치면 된다. 지금은 라틴 문자 Latin Modern Sans, 수식 Computer Modern, 한글 나눔바른고딕이다.
- 한글을 다른 고딕으로: `\setsanshangulfont{NanumGothic}` 또는 `\setsanshangulfont{Noto Sans CJK KR}`
- 본문 전체를 명조 계열로: `\usefonttheme{serif}`를 더하고 `\setmainhangulfont{NanumMyeongjo}`(또는 `Noto Serif CJK KR`)
- 발표자 노트를 PDF에 함께 넣기: `\setbeameroption{show notes on second screen=right}`
