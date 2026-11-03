# ASWISS 발표 자료

## 새 자료 (2026-11-03): `aswiss_deck.tex` (LaTeX Beamer)

CFR과 불완전 정보 게임을 모르는 청중을 위해 다시 짠 자료다. 본문 23장(여덟 질문에 차례로 답함) + 부록 10장. 하나의 H/L 게임 트리를 2·7–11장에서 단계별로 바꿔 가며 정방향(파랑)과 역방향(주황) 분할을 보인다.

- `aswiss_deck.tex`: Beamer(metropolis) 원고. 글꼴과 색은 이전 원고와 같다(Latin Modern Sans, 나눔바른고딕, 남색·파랑·주황). 게임 트리는 TikZ, 그래프는 pgfplots(`plotdata/`).
- `aswiss_deck.pdf`: 컴파일 결과(33쪽).
- `aswiss_deck_notes.pdf`: 각 슬라이드 뒤에 발표자 노트 쪽을 넣은 판(66쪽).
- `aswiss_deck_notes_KO.md`: 발표자 노트만 모은 것.
- `CHANGES_KO.md`: 이전 자료와의 대응표, 추가·삭제·이동한 것, 원본 PDF에 없던 내용과 출처, 요청과 다른 점.
- `aswiss_deck.pptx`와 `deck/`: 같은 내용의 PowerPoint 판과 그 생성 코드(편집 가능한 PPTX가 필요할 때).

```bash
xelatex aswiss_deck.tex && xelatex aswiss_deck.tex          # 두 번 (질문 표시 위치와 쪽 번호)
xelatex -jobname=aswiss_deck_notes "\def\withnotes{1}\input{aswiss_deck}"   # 노트 판, 역시 두 번
```

- 원고 맨 위 "글꼴" 네 줄만 고치면 글꼴이 바뀐다(아래 "글꼴 바꾸기"와 같다).
- 절 구분 슬라이드는 만들지 않는다(`\metroset{sectionpage=none}`). 절은 PDF 책갈피에만 남는다.

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
