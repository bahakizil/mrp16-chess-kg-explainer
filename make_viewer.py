"""Turn run_<mode>.json into a single HTML page (board + knowledge graph + MCTS replay).

  uv run --no-project --with chess python -I make_viewer.py [llm|offline|<puzzle_id>]
"""
import json, sys
from pathlib import Path
import chess, chess.svg

HERE = Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else "llm"
run = json.load(open(HERE / f"run_{tag}.json"))

created = {"position": 0}
for ev in run["events"]:
    for c, _p in ev.get("expanded", []):
        created[" > ".join(ev["selected"] + [c])] = ev["sim"]
for t in run["tree"]:
    t["created"] = created.get(t["id"], 0)

board = chess.Board(run["fen"])
mv = chess.Move.from_uci(run.get("best_uci", "b5c7"))
svg = chess.svg.board(board, size=300, flipped=not board.turn, arrows=[chess.svg.Arrow(mv.from_square, mv.to_square, color="#d9731f")])
run["board_svg"] = svg

html = (HERE / "viewer_template.html").read_text().replace("/*DATA*/", "const RUN = " + json.dumps(run) + ";")
out = HERE / f"chess_kg_demo_{tag}.html"
out.write_text(html)
print("wrote", out)
