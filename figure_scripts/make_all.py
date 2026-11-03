"""Run from any directory: python figure_scripts/make_all.py."""
from pathlib import Path
import subprocess
import sys
from common import OUTPUT
HERE = Path(__file__).resolve().parent
for number in range(1, 6):
    subprocess.run([sys.executable, str(HERE / f"fig{number}.py")], check=True)
source = HERE / "tikz" / "schematic_ssl_data2vec.tex"
log = OUTPUT / "tikz-build.log"
with log.open("w") as handle:
    result = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                             f"-outdir={OUTPUT}", source.name], cwd=source.parent,
                            stdout=handle, stderr=subprocess.STDOUT)
if result.returncode:
    raise RuntimeError(f"TikZ build failed; inspect {log}")
print(f"Main-text figure assets written to {OUTPUT}")
