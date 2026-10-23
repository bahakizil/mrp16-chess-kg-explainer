"""Build site/pipeline.html: a step-by-step view of how the demo works, from a logged run.

  uv run --no-project --with chess python -I make_pipeline.py run_2Tl6d.json
"""
import json, sys
from pathlib import Path
import chess, chess.svg

HERE = Path(__file__).resolve().parent
run = json.load(open(HERE / (sys.argv[1] if len(sys.argv) > 1 else "run_2Tl6d.json")))
nodes, edges = run["nodes"], run["edges"]
board = chess.Board(run["fen"])
me = board.turn

# Step 3: motifs python-chess found after every move of every engine line
succ = {}
for a, b, rel in edges:
    succ.setdefault(a, []).append((b, rel))
lines = []
for score, cand, san_line, score_txt in run["engine_full"]:
    plies, b = [], board.copy()
    for i, san in enumerate(san_line):
        nid = f"line:{cand}|{i}|{san}"
        motifs = [nodes[t]["label"] for t, rel in succ.get(nid, []) if nodes.get(t, {}).get("kind") == "motif"]
        mv = b.parse_san(san)
        plies.append({"san": san, "side": "White" if b.turn else "Black", "motifs": motifs})
        b.push(mv)
    lines.append({"cand": cand, "score": score_txt, "plies": plies})

# small boards along the solution line (what python-chess "looks at")
sol_boards, b = [], board.copy()
for ply in lines[0]["plies"]:
    mv = b.parse_san(ply["san"]); b.push(mv)
    sol_boards.append({"san": ply["san"], "side": ply["side"], "motifs": ply["motifs"],
                       "svg": chess.svg.board(b, size=170, lastmove=mv, flipped=not me, coordinates=False)})

# baseline legality, checked with python-chess
base = run.get("baseline", {})
try:
    board.parse_san(base.get("move", "")); base["legal"] = True
except Exception:
    base["legal"] = False

log = run["llm_log"]
tin = sum(x["in_tokens"] or 0 for x in log); tout = sum(x["out_tokens"] or 0 for x in log)
data = {
    "fen": run["fen"], "puzzle": run["puzzle"], "best_move": run["best_move"],
    "board_svg": chess.svg.board(board, size=320, flipped=not me,
                                 arrows=[chess.svg.Arrow(chess.Move.from_uci(run["best_uci"]).from_square,
                                                         chess.Move.from_uci(run["best_uci"]).to_square, color="#d9731f")]),
    "lines": lines, "sol_boards": sol_boards,
    "nodes": nodes, "edges": edges, "ppr": run["ppr"], "sub_nodes": run["sub_nodes"], "sub_edges": run["sub_edges"],
    "accepted": run["accepted"], "events": run["events"], "llm_log": log, "explanation": run["explanation"],
    "baseline": base, "check": run["check"], "calls": run["calls"], "T": run["T"],
    "seconds": run["search_seconds"], "tokens_in": tin, "tokens_out": tout,
    # gpt-4.1-mini list price (USD per 1M tokens): 0.40 input, 1.60 output
    "cost_usd": round(tin * 0.40e-6 + tout * 1.60e-6, 4), "model": run["model"],
}
html = (HERE / "pipeline_template.html").read_text().replace("/*DATA*/", "const D = " + json.dumps(data) + ";")
(HERE / "site" / "pipeline.html").write_text(html)
print("wrote site/pipeline.html", len(html) // 1024, "KB; calls", len(log), "tokens", tin, tout, "cost", data["cost_usd"])
