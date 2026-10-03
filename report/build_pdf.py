"""Render report/REPORT.md to report/REPORT.html (KaTeX + Mermaid) and print it to report/REPORT.pdf with headless Chrome."""

import pathlib
import re
import subprocess

import markdown

root = pathlib.Path(__file__).resolve().parent
md = (root / "REPORT.md").read_text()
# Python-Markdown needs a blank line before a list that follows a paragraph line (GitHub does not)
md = re.sub(r"(?m)^(?!\s*(?:[-*]|\d+\.)\s)(?=\S)(.+)\n(?=(?:[-*]|\d+\.)\s)", r"\1\n\n", md)
md = re.sub(r"(?m)^  ((?:[-*]|\d+\.) )", r"    \1", md)  # nested lists need 4-space indentation
body = markdown.markdown(md, extensions=["tables", "fenced_code", "pymdownx.arithmatex"],
                         extension_configs={"pymdownx.arithmatex": {"generic": True}})
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>YOLOv8 + DINOv2 on Pascal VOC</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
 body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; max-width: 820px; margin: auto; color: #222; }}
 h1 {{ font-size: 19pt; }} h2 {{ font-size: 14pt; border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 22px; }} h3 {{ font-size: 12pt; }}
 table {{ border-collapse: collapse; margin: 8px 0; font-size: 9pt; page-break-inside: avoid; }}
 th, td {{ border: 1px solid #bbb; padding: 3px 6px; }} th {{ background: #f0f0f0; }}
 img {{ max-width: 100%; page-break-inside: avoid; }} code {{ font-size: 9pt; background: #f4f4f4; padding: 0 2px; }}
 pre {{ background: #f4f4f4; padding: 8px; font-size: 8.5pt; overflow-x: auto; }}
 .mermaid {{ text-align: center; page-break-inside: avoid; }} .mermaid svg {{ max-height: 520px; }}
 @page {{ margin: 16mm 14mm; }}
</style></head><body>{body}
<script>
 document.querySelectorAll("code.language-mermaid").forEach(c => {{ const d = document.createElement("div"); d.className = "mermaid"; d.textContent = c.textContent; c.parentElement.replaceWith(d); }});
 mermaid.initialize({{ startOnLoad: false, theme: "neutral" }});
 mermaid.run().then(() => renderMathInElement(document.body, {{ delimiters: [{{left: "\\\\[", right: "\\\\]", display: true}}, {{left: "\\\\(", right: "\\\\)", display: false}}] }}));
</script></body></html>"""
(root / "REPORT.html").write_text(html)
subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new", "--disable-gpu",
                "--no-pdf-header-footer", "--virtual-time-budget=15000", f"--print-to-pdf={root / 'REPORT.pdf'}",
                (root / "REPORT.html").as_uri()], check=True, stderr=subprocess.DEVNULL)
print("wrote", root / "REPORT.pdf")
