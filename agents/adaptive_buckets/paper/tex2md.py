"""Make the Markdown copy of the paper (aswiss_paper.md) from the LaTeX source, once.

Usage: python tex2md.py            (run in agents/adaptive_buckets/paper after a LaTeX build; reads main.aux)
The Markdown copy is the editing copy afterwards: edits are carried into the .tex by hand, so do not rerun this
over an edited aswiss_paper.md (it would overwrite the edits). Needs pandoc.

What it does before pandoc: inlines \\input files; numbers sections, theorem-like environments, equations and
tables as LaTeX does (and checks them against main.aux); writes \\ref and \\eqref as numbers and \\citep as
(author, year); turns \\paragraph, description lists, \\todo and the two algorithmic blocks into plain forms.
After pandoc (gfm, $ math): puts display math on lines of its own and adds the title, abstract and references.
"""
import re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "main.tex").read_text(encoding="utf-8")
aux = (HERE / "main.aux").read_text(encoding="utf-8")
bib = (HERE / "refs.bib").read_text(encoding="utf-8")

# ---- labels and citation order from main.aux
labels = {m.group(1): m.group(2).strip("{}") for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{(\{?[^}]*)\}", aux)}
cite_order = [m.group(1) for m in re.finditer(r"\\bibcite\{([^}]*)\}", aux)]

# ---- bibliography entries
def bib_entries(text):
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n(?=@|\Z)", text + "\n@", re.S):
        kind, key, body = m.group(1), m.group(2).strip(), m.group(3)
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*\{", body):
            depth, j = 1, f.end()
            while depth:
                depth += {"{": 1, "}": -1}.get(body[j], 0); j += 1
            fields.setdefault(f.group(1).lower(), body[f.end():j - 1])
        out[key] = (kind, fields)
    return out
entries = bib_entries(bib)

def detex(s):
    s = re.sub(r"\{\\v\{?([a-zA-Z])\}?\}", lambda m: {"r": "ř", "c": "č", "s": "š", "C": "Č"}.get(m.group(1), m.group(1)), s)
    s = re.sub(r"\\v\{?([a-zA-Z])\}?", lambda m: {"r": "ř", "c": "č", "s": "š", "C": "Č"}.get(m.group(1), m.group(1)), s)
    s = re.sub(r"\{\\'\\i\}|\\'\\i|\\'\{\\i\}", "í", s)
    s = re.sub(r"\{\\'([a-zA-Z])\}|\\'([a-zA-Z])", lambda m: {"y": "ý", "a": "á", "e": "é", "i": "í"}.get(m.group(1) or m.group(2), m.group(1) or m.group(2)), s)
    s = s.replace("{\\~n}", "ñ").replace("{\\i}", "ı").replace("{\\c{c}}", "ç").replace("\\c{c}", "ç")
    s = s.replace("{", "").replace("}", "").replace("\\&", "&").replace("--", "–")
    return s

def surname(author):
    a = detex(author).strip()
    return a.split(",")[0].strip() if "," in a else a.split()[-1]

def cite_text(key):
    kind, f = entries[key]
    authors = [a.strip() for a in f["author"].split(" and ")]
    if len(authors) == 1: who = surname(authors[0])
    elif len(authors) == 2: who = f"{surname(authors[0])} & {surname(authors[1])}"
    else: who = f"{surname(authors[0])} 외"
    return f"{who}, {f['year']}"

# ---- expand \input
def expand(text):
    return re.sub(r"\\input\{([^}]*)\}", lambda m: expand((HERE / (m.group(1) + ".tex")).read_text(encoding="utf-8")), text)
body = expand(src)
body = body[body.index("\\begin{document}") + len("\\begin{document}"): body.index("\\end{document}")]
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1)
body = body[body.index("\\clearpage") + len("\\clearpage"):]
body = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}", "", body)

# ---- brace-matching helper
def group_end(s, i):
    """index just past the {...} group starting at s[i] == '{'"""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{": depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0: return j + 1
    raise ValueError("unbalanced")

def strip_wrapper(s, cmd, nargs_before):
    """\\cmd{a}{b}{CONTENT} -> CONTENT (nargs_before groups dropped)"""
    while True:
        i = s.find(cmd)
        if i < 0: return s
        j = i + len(cmd)
        for _ in range(nargs_before): j = group_end(s, j)
        k = group_end(s, j)
        s = s[:i] + s[j + 1:k - 1] + s[k:]

body = strip_wrapper(body, "\\resizebox", 2)

# ---- the two algorithmic blocks: replaced by hand-written Markdown after pandoc
ALG = []
def take_alg(m):
    ALG.append(m.group(0)); return f"\n\nALGPLACEHOLDER{len(ALG) - 1}\n\n"
body = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", take_alg, body, flags=re.S)
body = re.sub(r"\\begin\{algorithmic\}.*?\\end\{algorithmic\}", take_alg, body, flags=re.S)

# ---- numbering walk: sections, theorem-like environments, equations, tables
THM = {"theorem": "정리", "lemma": "보조정리", "proposition": "명제", "corollary": "따름정리", "definition": "정의",
       "assumption": "가정", "example": "예", "remark": "비고"}
out, pos = [], 0
sec, sub, thm, eq, tab, appendix = 0, 0, 0, 0, 0, False
token = re.compile(r"\\appendix\b|\\section\{|\\subsection\{|\\begin\{(" + "|".join(THM) + r"|equation|table|proof)\}|\\end\{(" + "|".join(THM) + r"|equation|table|proof)\}")
proof_title = []
for m in token.finditer(body):
    out.append(body[pos:m.start()]); pos = m.end()
    t = m.group(0)
    if t == "\\appendix": appendix, sec = True, 0; continue
    if t in ("\\section{", "\\subsection{"):
        k = group_end(body, m.end() - 1); title = body[m.end():k - 1]; pos = k
        if t == "\\section{":
            sec += 1; sub = 0; thm = 0
            num = chr(64 + sec) if appendix else str(sec)
            out.append("\\section{" + (f"부록 {num}. " if appendix else f"{num}. ") + title + "}")
        else:
            sub += 1; num = (chr(64 + sec) if appendix else str(sec)) + f".{sub}"
            out.append("\\subsection{" + f"{num} " + title + "}")
        continue
    env = m.group(1) or m.group(2); opening = m.group(1) is not None
    if env in THM:
        if opening:
            thm += 1; num = (chr(64 + sec) if appendix else str(sec)) + f".{thm}"
            opt = ""
            if body.startswith("[", pos):
                k = pos; depth = 0
                for k in range(pos, len(body)):
                    if body[k] == "[": depth += 1
                    elif body[k] == "]":
                        depth -= 1
                        if depth == 0: break
                opt = body[pos + 1:k]; pos = k + 1
            lab = re.match(r"\s*\\label\{([^}]*)\}", body[pos:])
            if lab:
                assert labels.get(lab.group(1)) == num, (lab.group(1), labels.get(lab.group(1)), num)
                pos += lab.end()
            out.append(f"\n\n\\textbf{{{THM[env]} {num}" + (f" ({opt})" if opt else "") + ".} ")
        else:
            out.append("\n\n")
    elif env == "proof":
        if opening:
            opt = "증명"
            if body.startswith("[", pos):
                k = body.index("]", pos); opt = body[pos + 1:k]; pos = k + 1
            out.append(f"\n\n\\textit{{{opt}.}} ")
        else:
            out.append(" ∎\n\n")
    elif env == "equation":
        if opening:
            lab = re.search(r"\\label\{([^}]*)\}", body[pos:pos + 200])
            tag = re.search(r"\\tag\{([^}]*)\}", body[pos:pos + 200])
            if tag is None:
                eq += 1
                if lab: assert labels.get(lab.group(1)) == str(eq), (lab.group(1), labels.get(lab.group(1)), eq)
                out.append("\\[\\tag{" + str(eq) + "}")
            else:
                out.append("\\[")
        else:
            out.append("\\]")
    elif env == "table":
        if opening:
            end = body.index("\\end{table}", pos); inner = body[pos:end]; pos = end + len("\\end{table}")
            inner = re.sub(r"^\[[^\]]*\]", "", inner)
            cap = re.search(r"\\caption\{", inner)
            head = ""
            if cap:
                tab += 1; k = group_end(inner, cap.end() - 1)
                head = f"\\textbf{{표 {tab}.}} " + inner[cap.end():k - 1] + "\n\n"
                inner = inner[:cap.start()] + inner[k:]
                lab = re.search(r"\\label\{([^}]*)\}", inner)
                if lab: assert labels.get(lab.group(1)) == str(tab), (lab.group(1), labels.get(lab.group(1)), tab)
            inner = re.sub(r"\\label\{[^}]*\}|\\centering|\\small", "", inner)
            out.append("\n\n" + head + inner + "\n\n")
out.append(body[pos:])
body = "".join(out)

# ---- references, citations and the rest
body = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(" + labels[m.group(1)] + ")", body)
body = re.sub(r"~?\\ref\{([^}]*)\}", lambda m: " " + labels[m.group(1)] if m.group(0).startswith("~") else labels[m.group(1)], body)
def citep(m):
    note, keys = m.group(1), [k.strip() for k in m.group(2).split(",")]
    text = "; ".join(cite_text(k) for k in keys)
    return " (" + text + (f", {note}" if note else "") + ")"
body = re.sub(r"~?\\citep(?:\[([^\]]*)\])?\{([^}]*)\}", citep, body)
body = re.sub(r"\\label\{[^}]*\}", "", body)
body = re.sub(r"\\paragraph\{", r"\\textbf{", body)
body = re.sub(r"\\todo\{", r"\\textbf{[확인: ", body)
body = re.sub(r"\\textbf\{\[확인: ((?:[^{}]|\{[^{}]*\})*)\}", r"\\textbf{[확인: \1]}", body)
body = re.sub(r"\\textsc\{([^}]*)\}", r"\\texttt{\1}", body)
body = body.replace("\\mbox{}", "")
def description(m):
    items = re.split(r"\\item\[", m.group(1))[1:]
    res = []
    for it in items:
        k = it.index("]"); res.append("\\item \\textbf{" + it[:k] + "} " + it[k + 1:])
    return "\\begin{itemize}" + "".join(res) + "\\end{itemize}"
body = re.sub(r"\\begin\{description\}(.*?)\\end\{description\}", description, body, flags=re.S)

preamble = src[:src.index("\\begin{document}")]
macros = "\n".join(l for l in preamble.splitlines() if l.startswith(("\\newcommand", "\\DeclareMathOperator")) and "todo" not in l)
doc = "\\documentclass{article}\n" + macros + "\n\\begin{document}\n" + body + "\n\\end{document}\n"
(HERE / "_md_input.tex").write_text(doc, encoding="utf-8")
md = subprocess.run(["pandoc", "-f", "latex", "-t", "gfm", "--wrap=preserve", "--shift-heading-level-by=1", str(HERE / "_md_input.tex")],
                    check=True, capture_output=True, text=True).stdout
abs_md = subprocess.run(["pandoc", "-f", "latex", "-t", "gfm", "--wrap=preserve"], input="\\documentclass{article}\n" + macros + "\n\\begin{document}\n" + abstract + "\\end{document}",
                        check=True, capture_output=True, text=True).stdout
(HERE / "_md_input.tex").unlink()

def display_lines(text):
    # $$...$$ on lines of their own
    return re.sub(r"[ \t]*\$\$(.+?)\$\$[ \t]*", lambda m: "\n\n$$" + m.group(1).strip("\n") + "$$\n\n", text, flags=re.S)
md, abs_md = display_lines(md), display_lines(abs_md)
md = re.sub(r"\n{3,}", "\n\n", md); abs_md = re.sub(r"\n{3,}", "\n\n", abs_md)
def tight_lists(text):
    # drop the blank line between two list items (pandoc writes loose lists)
    item = re.compile(r"^\s*(?:[-*]|\d+\.)\s")
    lines = text.split("\n"); keep = []
    for k, line in enumerate(lines):
        if line == "" and keep and k + 1 < len(lines) and item.match(lines[k + 1]):
            prev = next((l for l in reversed(keep) if l != ""), "")
            j = len(keep) - 1
            while j >= 0 and keep[j] != "" and not item.match(keep[j]): j -= 1
            if j >= 0 and keep[j] != "" and item.match(keep[j]): continue
        keep.append(line)
    return "\n".join(keep)
md = tight_lists(md)

ALG_MD = {
    0: """> **알고리즘 1** 순수 ASWISS (방식 E, 재생 라벨)
>
> **입력:** 시작 분할 $\\mathcal{P}_0$, 첫 epoch 길이 $t_0$, 저장 개수 $K_k$
>
> 1. 분할 사건 목록 $\\mathcal{E}\\gets\\emptyset$, $k\\gets0$, 모든 bucket의 표 $Q_B\\gets0$, 평균 표 $S_B\\gets0$
> 2. **for** $t=1,2,\\dots$ **do**
>    - **for** traverser $i\\in\\{1,2\\}$ **do**
>      - 외부 샘플링으로 순회한다. 정보집합 $I$에 닿으면 $B\\gets$ `Route`$(I,k)$로 찾는다.
>      - $\\sigma(I)\\propto Q_B^+$로 행동하고, 표본 후회를 $Q_B$에, 평균을 $S_B$에 쌓는다.
>    - **if** $t$가 epoch $k$에서 뽑아 둔 $K_k$개의 반복 중 하나 **then** 현재 bucket 전략표를 $\\mathcal{S}_k$에 저장
>    - **if** $t=t_{k+1}$ (epoch 끝) **then**
>      - **for** 각 bucket $B$ **do**
>        - 멤버의 라벨 $\\ell(I)$를 $\\mathcal{S}_k$의 재생으로 구한다(4.5절)
>        - **if** 라벨이 두 종류 이상 **then** $B$를 라벨별로 나누고 자식 표를 0에서 시작한다
>      - 사건 $E_{k+1}$(나뉜 bucket, 자식 라벨, $\\mathcal{S}_k$)을 $\\mathcal{E}$에 더한다. $k\\gets k+1$
""",
    1: """> **function** `Route`$(I,k)$
>
> 1. **if** $k=0$ **then return** $\\mathcal{P}_0(I)$
> 2. $B\\gets$ `Route`$(I,k-1)$
> 3. **if** $E_k$에서 $B$가 나뉘지 않았다 **then return** $B$
> 4. $\\ell\\gets$ $\\mathcal{S}_k$로 계산한 $I$의 라벨(자식 라벨 중에서 고름)
> 5. **return** $E_k$에서 라벨이 $\\ell$인 $B$의 자식
""",
}
# the algorithm environment was taken first, the inline algorithmic second
for i, text in ALG_MD.items():
    md = md.replace(f"ALGPLACEHOLDER{i}", text.strip("\n"))

refs = []
for key in cite_order:
    kind, f = entries[key]
    authors = ", ".join(detex(a.strip()) for a in f["author"].split(" and "))
    venue = f.get("journal") or f.get("booktitle") or f.get("note", "")
    extra = ""
    if f.get("volume"): extra += f", {f['volume']}"
    if f.get("number"): extra += f"({f['number']})"
    if f.get("pages"): extra += f", {detex(f['pages'])}"
    note = f", {f['note']}" if f.get("note") and venue != f.get("note") else ""
    refs.append(f"- {authors} ({f['year']}). {detex(f['title'])}. *{detex(venue)}*{extra}{note}.")

title = re.search(r"\\title\{(.*?)\\\\\[4pt\]\s*\\large (.*?)\}\s*\\author", src, re.S)
header = (f"# {title.group(1).strip()}\n\n**{title.group(2).strip()}**\n\n"
          "> 이 파일은 논문 초안(`main.tex`, `main.pdf`)의 Markdown 사본이자 편집본이다. 문장을 바로 고치거나, "
          "고칠 곳 아래에 `>> 요청` 줄을 적는다. 고친 문장은 .tex에 그대로 옮긴다.\n\n"
          "## 초록\n\n" + abs_md.strip() + "\n\n")
final = header + md.strip() + "\n\n## 참고 문헌\n\n" + "\n".join(refs) + "\n"
(HERE / "aswiss_paper.md").write_text(final, encoding="utf-8")
print("wrote aswiss_paper.md:", len(final.splitlines()), "lines")
