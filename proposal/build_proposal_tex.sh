#!/bin/zsh
# 用教授给的 neurips_2021.sty 编译提案。用法： ./build_proposal_tex.sh
cd "$(dirname "$0")"
export PATH="/usr/local/bin:$PATH"
SRC=${1:-proposal.tex}
OUT=${2:-MH6805_Group_Project_Proposal}

pdflatex -interaction=nonstopmode -jobname="$OUT" "$SRC" >/dev/null 2>&1
pdflatex -interaction=nonstopmode -jobname="$OUT" "$SRC" > _tex.log 2>&1

if grep -q "^!" _tex.log; then
  echo "❌ LaTeX 报错（下面是出错的地方，改完再跑一次）："
  grep -A4 "^!" _tex.log | head -30
  echo
  echo "   完整日志： $PWD/_tex.log"
  exit 1
fi

OVER=$(grep -c "Overfull \\\\hbox" _tex.log)
python3 - "$OUT" "$OVER" <<'PY'
import sys, pymupdf
out, over = sys.argv[1], int(sys.argv[2])
d = pymupdf.open(out + ".pdf")
n = d.page_count
print(("✅" if n <= 4 else "❌") + f" 页数：{n}" + ("" if n <= 4 else "  —— 超过 4 页上限，必须删内容！"))
print(("✅" if over == 0 else "⚠️ ") + f" 出界的行（overfull hbox）：{over}")
for i, p in enumerate(d):
    print(f"     第 {i+1} 页 {len(p.get_text().split())} 词")
if n > 4:
    bs = [b for b in d[3].get_text('blocks') if b[3] < 740]
    print(f"\n   第 4 页内容到 y={max(b[3] for b in bs):.0f}（版心底部是 722），第 5 页起是溢出的部分。")
PY
rm -f "$OUT.aux" "$OUT.out" "$OUT.log" _tex.log
