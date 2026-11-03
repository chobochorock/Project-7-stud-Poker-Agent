"""aswiss_deck_notes_KO.md from the \\note{...} of every frame in aswiss_deck.tex (usage: python notes_md.py)."""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'aswiss_deck.tex'), encoding='utf-8').read()


def braced(s, i):  # s[i] == '{' -> (content, index after the closing brace)
    depth, j = 0, i
    while True:
        if s[j] == '\\':
            j += 2; continue
        if s[j] == '{': depth += 1
        elif s[j] == '}':
            depth -= 1
            if depth == 0: return s[i + 1:j], j + 1
        j += 1


out = ['# ASWISS 발표 자료: 발표자 노트', '',
       '`aswiss_deck.tex`의 각 frame에 있는 `\\note{}`를 옮긴 것이다(`python notes_md.py`로 다시 만든다). '
       '`aswiss_deck_notes.pdf`에는 같은 노트가 각 슬라이드 뒤에 있다.', '']
n = 0
for m in re.finditer(r'\\begin\{frame\}(\[[^\]]*\])?', src):
    end = src.index('\\end{frame}', m.end())
    body = src[m.end():end]
    n += 1
    title = braced(body, body.index('{'))[0] if body.lstrip().startswith('{') else 'ASWISS (표지)'
    k = body.find('\\note{')
    note = braced(body, k + 5)[0] if k >= 0 else ''
    out.append(f'## {n}. {title}'); out.append('')
    for para in note.split('\\par'):
        para = ' '.join(para.split()).replace('\\_', '_').replace('\\%', '%')
        if para: out += [para, '']
open(os.path.join(HERE, 'aswiss_deck_notes_KO.md'), 'w', encoding='utf-8').write('\n'.join(out))
print(n, 'frames')
