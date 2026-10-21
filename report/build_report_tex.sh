#!/bin/zsh
# 用教授给的 neurips_2021.sty 编译最终报告。用法： ./build_report_tex.sh
cd "$(dirname "$0")"
export PATH="/usr/local/bin:$PATH"
SRC=${1:-report.tex}
OUT=${2:-MH6805_Group_Project_Report}

# A stale or truncated .aux from an interrupted direct pdflatex run poisons pass 1 (every \ref
# prints "??"), so start from a clean slate; three passes below rebuild it.
rm -f "$OUT.aux" "$OUT.out" "$OUT.toc"

pdflatex -interaction=nonstopmode -jobname="$OUT" "$SRC" >/dev/null 2>&1
pdflatex -interaction=nonstopmode -jobname="$OUT" "$SRC" >/dev/null 2>&1
pdflatex -interaction=nonstopmode -jobname="$OUT" "$SRC" > _tex.log 2>&1

if grep -q "^!" _tex.log; then
  echo "❌ LaTeX 报错："
  grep -A4 "^!" _tex.log | head -30
  echo "   完整日志： $PWD/_tex.log"
  exit 1
fi

OVER=$(grep -c "Overfull \\\\hbox" _tex.log)
python3 - "$OUT" "$OVER" <<'PY'
import sys, pymupdf, re
out, over = sys.argv[1], int(sys.argv[2])
d = pymupdf.open(out + ".pdf")
# 正文 = 附录开始之前（找 "Appendix" 一级标题所在页）
app = None
for i, p in enumerate(d):
    t = p.get_text()
    if re.search(r"^\s*A\s+Appendix|^\s*Appendix\b", t, re.M) and i > 2:
        app = i + 1; break
body = (app - 1) if app else d.page_count
print(("✅" if body <= 7 else "❌") + f" 正文页数：{body}" + ("" if body <= 7 else "  —— 超过 7 页上限（含参考文献），必须删！"))
print(f"   附录从第 {app} 页开始" if app else "   （未检测到附录）")
print(f"   总页数：{d.page_count}")
print(("✅" if over == 0 else "⚠️ ") + f" 出界的行：{over}")
PY
rm -f "$OUT.aux" "$OUT.out" "$OUT.log" "$OUT.toc" _tex.log
