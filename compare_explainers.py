"""Test 3: is MCTS needed to explain a puzzle? Compare four explainers on the same puzzle and model.

  0  board only            : LLM sees the position only                        (from the MCTS run's baseline)
  1  engine lines          : LLM sees the position + Stockfish's top-5 lines     (no search)
  2  all verified facts    : LLM sees the position + the whole knowledge graph   (no search)
  3  MCTS + LLM            : LLM writes from the chains the search accepted     (from run_<id>.json)

Conditions 1 and 2 are repeated REPS times (one LLM call each). Every explanation is scored automatically.

  uv run --no-project --with chess,openai,python-dotenv python -I compare_explainers.py 0s9eK [REPS]
"""
import json, os, re, sys, time
from pathlib import Path
import chess

HERE = Path(__file__).resolve().parent
PID = sys.argv[1] if len(sys.argv) > 1 else "0s9eK"
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MODEL = "openai/gpt-4.1-mini"
run = json.load(open(HERE / f"run_{PID}.json"))
board = chess.Board(run["fen"]); side = "White" if board.turn else "Black"
best = run["best_move"]; themes = run["puzzle"]["themes"]

from dotenv import load_dotenv
from openai import OpenAI
for env in (HERE / ".env", HERE.parent / ".env", *[p / ".env" for p in HERE.parents]):
    load_dotenv(env)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

TASK = f"Write a 3-4 sentence explanation for a club player of {side}'s winning move and why it works."
HEAD = f"Chess position (FEN): {run['fen']}\n{board}\n{side} to move.\n\n"

lines_txt = "\n".join(f"{i+1}. {cand} ({score}): {' '.join(line)}" for i, (_, cand, line, score) in enumerate(run["engine_full"]))
P1 = HEAD + f"Stockfish analysis (depth 18), top 5 moves with evaluation and expected line:\n{lines_txt}\n\n" + TASK
lab = lambda n: run["nodes"][n]["label"]
facts_txt = "\n".join(f"- {lab(a)} --{rel}--> {lab(b)}" for a, b, rel in run["edges"])
P2 = (HEAD + "Verified facts from a chess knowledge graph (every fact was checked with Stockfish or python-chess):\n"
      + facts_txt + "\n\nUse ONLY these facts. " + TASK)


def ask(prompt):
    for _ in range(4):
        try:
            t0 = time.time()
            r = client.chat.completions.create(model=MODEL, temperature=0, messages=[{"role": "user", "content": prompt}])
            return {"text": r.choices[0].message.content.strip(), "in": r.usage.prompt_tokens, "out": r.usage.completion_tokens,
                    "seconds": round(time.time() - t0, 2)}
        except Exception as e:
            err = e; time.sleep(3)
    raise err


# ---------- automatic scoring ----------
THEME_WORDS = {"pin": r"\bpin", "sacrifice": r"sacrific", "fork": r"\bfork",
               "attraction": r"attract|lur(e|es|ing)\b|decoy|draws? the king|drawing the king|forc\w* the king (to|onto) g7",
               "deflection": r"deflect", "discoveredAttack": r"discover", "discoveredCheck": r"discovered check",
               "hangingPiece": r"hanging|undefended|unprotected", "kingsideAttack": r"king ?side|attack on the king|king attack|attack(s|ing)? the king",
               "exposedKing": r"exposed king|king safety|exposes? the king", "mate": r"\bmate\b|checkmate", "clearance": r"clearance"}
GENERIC = {"advantage", "crushing", "long", "short", "veryLong", "middlegame", "endgame", "opening", "master", "equality"}
SAN_RE = re.compile(r"\b(?:O-O(?:-O)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|[a-h](?:x[a-h])?[1-8](?:=[QRBN])?)[+#]?")
known = set()
for _, _, line, _ in run["engine_full"]:
    known |= {m.rstrip("+#") for m in line}
known |= {board.san(m).rstrip("+#") for m in board.legal_moves}
norm = lambda m: m.rstrip("+#")


def score(text):
    t = text or ""
    moves = SAN_RE.findall(t)
    # bare squares ("e5", "the g7 pawn") are not moves: keep tokens with a piece letter, a capture or a check
    moves = [m for m in moves if re.match(r"[KQRBNO]", m) or "x" in m or m.endswith(("+", "#"))]
    unver = sorted({m for m in moves if norm(m) not in known})
    hit = [th for th in themes if th not in GENERIC and th in THEME_WORDS and re.search(THEME_WORDS[th], t, re.I)]
    return {"correct_move": norm(best) in {norm(m) for m in moves}, "themes_hit": hit,
            "themes_total": [th for th in themes if th not in GENERIC], "unverified_moves": unver,
            "words": len(t.split())}


results = {"puzzle": run["puzzle"], "fen": run["fen"], "best_move": best, "best_uci": run.get("best_uci"), "model": MODEL,
           "reps": REPS, "engine_lines": lines_txt, "conditions": {}}
base = run.get("baseline", {})
bl = next((c for c in run.get("llm_log", []) if c["kind"] == "baseline"), None)
btxt = f"Move: {base.get('move')}. {base.get('explanation', '')}"
results["conditions"]["0"] = {"name": "Board only", "saw": "the position only",
    "runs": [{"text": btxt, "in": bl and bl["in_tokens"], "out": bl and bl["out_tokens"], "score": score(btxt)}],
    "calls_per_run": 1, "prompt": bl and bl["prompt"]}
for key, name, saw, prompt in [("1", "Engine lines, no search", "the position + Stockfish's top-5 lines and scores", P1),
                               ("2", "All verified facts, no search", "the position + the whole knowledge graph at once", P2)]:
    runs = []
    for i in range(REPS):
        a = ask(prompt); a["score"] = score(a["text"]); runs.append(a)
        print(f"[{key}] rep {i+1}: move={a['score']['correct_move']} themes={a['score']['themes_hit']} unverified={a['score']['unverified_moves']} in={a['in']}")
    results["conditions"][key] = {"name": name, "saw": saw, "runs": runs, "calls_per_run": 1, "prompt": prompt}
log = run.get("llm_log", [])
mcts_calls = [c for c in log if c["kind"] in ("prior", "eval", "explain")]
ex = next((c for c in log if c["kind"] == "explain"), None)
results["conditions"]["3"] = {"name": "MCTS + LLM (ours)", "saw": "only the chains of verified facts that the search accepted",
    "runs": [{"text": run["explanation"], "score": score(run["explanation"]),
              "in": sum(c["in_tokens"] or 0 for c in mcts_calls), "out": sum(c["out_tokens"] or 0 for c in mcts_calls)}],
    "calls_per_run": len(mcts_calls), "prompt": ex and ex["prompt"], "accepted": len(run["accepted"]),
    "graph_motifs": run["check"]["motifs"], "seconds": run.get("search_seconds")}
results["graph"] = {"nodes": len(run["nodes"]), "edges": len(run["edges"]), "facts_tokens_estimate": len(facts_txt) // 4}
json.dump(results, open(HERE / f"compare_{PID}.json", "w"), indent=1)
print("MCTS:", results["conditions"]["3"]["runs"][0]["score"], "calls", len(mcts_calls))
print("saved", f"compare_{PID}.json")
