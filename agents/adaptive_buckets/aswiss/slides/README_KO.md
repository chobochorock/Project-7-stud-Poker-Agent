# ASWISS 발표 자료

## 새 자료 (2026-11-03): `aswiss_deck.pptx` / `aswiss_deck.pdf`

- CFR과 불완전 정보 게임을 모르는 청중을 위해 다시 짠 자료다. 본문 23장(여덟 질문에 차례로 답함) + 부록 10장. 하나의 H/L 게임 트리를 2·7–11장에서 단계별로 바꿔 가며 정방향(파랑)과 역방향(주황) 분할을 보인다.
- `aswiss_deck.pptx`: 편집 가능한 PowerPoint. 글꼴은 테마 글꼴(맑은 고딕)이라 한 곳에서 바꿀 수 있다. 슬라이드마다 발표자 노트가 있다.
- `aswiss_deck.pdf`: 같은 자료의 PDF(LibreOffice 렌더링).
- `aswiss_deck_notes_KO.md`: 발표자 노트만 모은 것.
- `CHANGES_KO.md`: 이전 자료와의 대응표, 추가·삭제·이동한 것, 원본 PDF에 없던 내용과 출처, 요청과 다른 점.
- `deck/`: 생성 코드. `cd deck && npm install && node build_deck.js ../aswiss_deck.pptx ../plotdata`, PDF는 `python make_pdf.py ../aswiss_deck.pptx ../aswiss_deck.pdf`(LibreOffice 필요).

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
