# ASWISS 알고리즘 설명 슬라이드 (LaTeX Beamer)

- `aswiss_slides.tex`: Beamer(metropolis 테마) 원고, 16:9. 도식은 TikZ, 그래프는 pgfplots, 표는 booktabs로 그린다. 각 frame에 `\note{}`로 발표자 노트가 있다.
- `aswiss_slides.pdf`: 컴파일한 결과(표지, 본문 16장, 절 구분 4장).
- `data/`: Leduc 그래프 두 장의 데이터. `make_data.py`가 Kuhn·Leduc 실행 결과(`../small/run_small.sh`의 출력, 또는 `../results_data/small_games_summaries.tar.gz`를 푼 폴더)에서 만든다.

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
