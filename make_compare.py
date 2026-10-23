"""Build site/compare.html (test 3: is MCTS needed for the explanation?) from compare_<id>.json.

  uv run --no-project --with chess python -I make_compare.py 0s9eK
"""
import json, sys
from pathlib import Path
import chess, chess.svg

HERE = Path(__file__).resolve().parent
pid = sys.argv[1] if len(sys.argv) > 1 else "0s9eK"
d = json.load(open(HERE / f"compare_{pid}.json"))
b = chess.Board(d["fen"])
mv = chess.Move.from_uci(d["best_uci"]) if d.get("best_uci") else None
d["board_svg"] = chess.svg.board(b, size=300, flipped=not b.turn,
                                 arrows=[chess.svg.Arrow(mv.from_square, mv.to_square, color="#d9731f")] if mv else [])
verdict = HERE / f"compare_{pid}_verdict.html"
d["verdict"] = verdict.read_text() if verdict.exists() else ""
html = (HERE / "compare_template.html").read_text().replace("/*DATA*/", "const D = " + json.dumps(d) + ";")
(HERE / "site" / "compare.html").write_text(html)
print("wrote site/compare.html", len(html) // 1024, "KB")
