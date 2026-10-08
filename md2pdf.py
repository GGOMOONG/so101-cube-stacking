#!/usr/bin/env python3
"""마크다운 → PDF (한글 조판: 단어 단위 줄바꿈, 양쪽 정렬). 사용: python md2pdf.py 입력.md [출력.pdf] [--break-h2]
--break-h2: ## 제목(예: 날짜별 절)마다 새 페이지에서 시작   --split: 긴 글머리 항목을 문장마다 줄바꿈"""
import sys, os, subprocess, markdown
BREAK = "--break-h2" in sys.argv; SPLIT = "--split" in sys.argv; sys.argv = [a for a in sys.argv if a not in ("--break-h2", "--split")]
import re
def split_long_bullets(md, limit=160):
    """긴 줄(글머리 항목·문단)을 읽기 좋게 나눔: 초기 노트의 인라인 ' - ' 구분자는 들여쓴 줄로, 그 뒤 문장 단위로 줄바꿈. 코드 블록·표·제목은 건드리지 않음."""
    out = []; in_code = False
    for line in md.splitlines():
        st = line.strip()
        if st.startswith("```"): in_code = not in_code; out.append(line); continue
        if in_code or st.startswith("|") or st.startswith("#") or st.startswith("!") or len(line) <= limit: out.append(line); continue
        m = re.match(r"^(\s*[-*] )?(.*)$", line); head, body = (m.group(1) or ""), m.group(2)
        segs = re.split(r"\s+-\s+(?=\S)", body)                      # 인라인 ' - ' 구분자
        pieces = []
        for i, seg in enumerate(segs):
            sents = re.split(r"(?<=[.!?。])\s+(?=[가-힣A-Za-z(\*\d\"“'])", seg.strip())
            pieces.append(("" if i == 0 else "<br>&nbsp;&nbsp;– ") + "<br>&nbsp;&nbsp;&nbsp;&nbsp;".join(sents))
        out.append(head + "".join(pieces))
    return "\n".join(out)
src = os.path.abspath(os.path.expanduser(sys.argv[1])); out = os.path.abspath(os.path.expanduser(sys.argv[2])) if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".pdf"
text = open(src, encoding="utf-8").read()
# 목록 앞에 빈 줄이 없으면 파서가 목록으로 보지 않음 → 문단 뒤에 바로 오는 목록 항목 앞에 빈 줄 삽입
_lines = text.splitlines(); _fixed = []; _code = False
for _i, _l in enumerate(_lines):
    if _l.strip().startswith("```"): _code = not _code
    if (not _code and re.match(r"^\s*([-*]|\d+\.) ", _l) and _fixed and _fixed[-1].strip() and not re.match(r"^\s*([-*]|\d+\.) ", _fixed[-1]) and not _fixed[-1].strip().startswith("|")):
        _fixed.append("")
    _fixed.append(_l)
text = "\n".join(_fixed)
if SPLIT: text = split_long_bullets(text)
body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
body = re.sub(r'src="(?!https?://|/|file:)', f'src="file://{os.path.dirname(src)}/', body)   # 상대 경로 그림을 입력 md 기준 절대 경로로(임시 HTML이 출력 폴더에 생기므로)
html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8"><title>{os.path.basename(src)}</title>
<style>
@page {{ size: A4; margin: 18mm 17mm; }}
body {{ font-family: "Helvetica Neue", "Apple SD Gothic Neo", -apple-system, sans-serif; font-size: 10.5pt; line-height: 1.6; color: #111;
        word-break: keep-all; overflow-wrap: break-word; text-align: justify; text-justify: inter-character; }}
h1 {{ font-size: 19pt; border-bottom: 2px solid #333; padding-bottom: 4px; text-align: left; }}
h2 {{ font-size: 14pt; margin-top: 22px; border-bottom: 1px solid #999; padding-bottom: 2px; text-align: left; {"page-break-before: always;" if BREAK else ""} }}
h1 + h2, h1 + p + h2 {{ page-break-before: auto; }}
h3 {{ font-size: 12pt; margin-top: 14px; text-align: left; }}
p {{ margin: 6px 0; }}
ul, ol {{ padding-left: 22px; }}
li {{ text-align: left; margin: 9px 0; line-height: 1.65; }}
li li {{ margin: 4px 0; }}
code {{ font-family: Menlo, monospace; font-size: 9pt; background: #f2f2f2; padding: 1px 3px; border-radius: 3px; white-space: nowrap; }}
pre {{ background: #f6f6f6; padding: 8px; border-radius: 4px; font-size: 8.5pt; white-space: pre-wrap; text-align: left; }}
pre code {{ white-space: pre-wrap; background: none; }}
table {{ border-collapse: collapse; margin: 8px 0; font-size: 9.5pt; text-align: left; }}
th, td {{ border: 1px solid #bbb; padding: 3px 6px; vertical-align: top; text-align: left; }}
th {{ background: #eee; }}
img {{ max-width: 100%; height: auto; display: block; margin: 10px auto; }}
em {{ color: #444; }}
blockquote {{ border-left: 3px solid #999; margin: 8px 0; padding: 4px 12px; background: #fafafa; }}
</style></head><body>{body}</body></html>"""
tmp = out.rsplit(".", 1)[0] + ".html"; open(tmp, "w", encoding="utf-8").write(html)
subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", f"file://{tmp}"], capture_output=True)
os.remove(tmp); print("저장:", out)
