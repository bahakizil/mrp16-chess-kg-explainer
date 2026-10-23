"""TESSERA-style LLM-guided MCTS over the chess knowledge graph built by chess_kg.py.

Same ingredients as TESSERA (Jakhar, Dumontier & Celebi 2026):
  - state      = path from the start node (path-dependent, like s = (u, H, z))
  - actions    = out-neighbours of the last node, pre-filtered by personalised PageRank to the target
  - prior P    = LLM ranking of the candidate next steps
  - value V    = LLM evaluator score (0-3 rubric) of the partial chain, mapped to [-1, 1]
  - selection  = PUCT
  - reaching the target accepts the path and closes the branch
  - output     = union of all accepted paths = explanation subgraph

Run (from this folder):
  uv run --no-project --with chess,networkx,openai,python-dotenv python -I tessera_chess.py [llm|offline] [T] [lichess_puzzle_id]
"""
import json, math, os, sys, time
from pathlib import Path

import networkx as nx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chess_kg import build_kg
import chess
MODE = sys.argv[1] if len(sys.argv) > 1 else "llm"
T = int(sys.argv[2]) if len(sys.argv) > 2 else 40
PUZZLE = sys.argv[3] if len(sys.argv) > 3 else None   # Lichess puzzle id -> engine-built KG
K = 8                               # candidates kept per state (TESSERA: top-18 by PPR)
C_PUCT = 2.0
MODEL = "openai/gpt-4.1-mini"

if PUZZLE:
    from chess_kg_engine import build_kg_engine, load_puzzle, THEME_MAP
    PZ = load_puzzle(PUZZLE, HERE / "puzzles_2500.csv")
    FEN = PZ["fen"]
    EXPECTED_THEMES = set(PZ["themes"])
    board, G, results = build_kg_engine(FEN, plies=len(PZ["solution"]) + 1)
    TAG = PUZZLE
else:
    FEN = "r3k2r/pp3ppp/8/1N1q4/8/3P4/PPP2PPP/R2QK2R w - - 0 1"
    EXPECTED_THEMES = {"fork"}      # what a Lichess-style theme tag for this puzzle would say
    THEME_MAP = {"fork": "fork", "pin": "pin", "sacrifice": "sacrifice", "checkmate": "mate", "hanging": "hangingPiece"}
    PZ = None
    board, G, results = build_kg(FEN)
    TAG = MODE
SIDE = "White" if board.turn else "Black"
def personalised_pagerank(g, target, alpha=0.85, iters=100):
    """PPR toward `target` on the reversed graph (power iteration; no scipy needed)."""
    r = g.reverse(copy=True)
    s = {n: 0.0 for n in r}; s[target] = 1.0
    for _ in range(iters):
        nxt = {n: (1 - alpha) * (1.0 if n == target else 0.0) for n in r}
        for n in r:
            out = list(r.successors(n))
            for m in out:
                nxt[m] += alpha * s[n] / len(out)
        s = nxt
    return s


ppr = personalised_pagerank(G, "goal")

calls = {"prior": 0, "eval": 0, "explain": 0}
if MODE == "llm":
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(HERE / ".env"); load_dotenv(HERE.parent / ".env")   # repo root first
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

CONTEXT = (f"Chess position (FEN): {FEN}\n{board}\n"
           f"{SIDE} to move. We are building an EXPLANATION, as a chain of facts in a chess knowledge graph, "
           f"of how {SIDE} reaches a decisive advantage. Nodes are moves, pieces and tactical motifs; "
           "edges are verified relations such as gives, creates, exploits, targets, answered_by, accepted_by, opens, captured_by.")


def lbl(n):
    return G.nodes[n]["label"]


def chain_text(path):
    parts = [lbl(path[0])]
    for a, b in zip(path, path[1:]):
        parts.append(f"--{G.edges[a, b]['rel']}--> {lbl(b)}")
    return " ".join(parts)


LLM_LOG, CUR = [], {"sim": 0, "path": []}


def log_call(kind, prompt, r, t0):
    u = getattr(r, "usage", None)
    LLM_LOG.append({"kind": kind, "sim": CUR["sim"], "path": list(CUR["path"]), "prompt": prompt,
                    "response": r.choices[0].message.content, "seconds": round(time.time() - t0, 2),
                    "in_tokens": getattr(u, "prompt_tokens", None), "out_tokens": getattr(u, "completion_tokens", None)})


def ask_json(prompt, kind):
    calls[kind] += 1
    for _ in range(3):
        try:
            t0 = time.time()
            r = client.chat.completions.create(model=MODEL, temperature=0,
                                               response_format={"type": "json_object"},
                                               messages=[{"role": "user", "content": prompt}])
            out = json.loads(r.choices[0].message.content)
            log_call(kind, prompt, r, t0)
            return out
        except Exception as e:  # retry on transient API/JSON errors
            err = e; time.sleep(1)
    raise err


def candidates(path):
    last = path[-1]
    nbrs = [n for n in G.successors(last) if n not in path]
    nbrs.sort(key=lambda n: -ppr.get(n, 0))
    return nbrs[:K]


def prior(path, cands):
    if MODE != "llm" or len(cands) == 1:
        return {c: 1 / len(cands) for c in cands}
    opts = "\n".join(f"{i}: --{G.edges[path[-1], c]['rel']}--> {lbl(c)}" for i, c in enumerate(cands))
    out = ask_json(f"{CONTEXT}\n\nCurrent partial chain:\n{chain_text(path)}\n\nCandidate next steps:\n{opts}\n\n"
                   "Score each candidate 0-10 for how likely it continues a CORRECT and RELEVANT explanation. "
                   'Return JSON {"scores": {"0": s0, "1": s1, ...}}.', "prior")
    s = [float(out.get("scores", {}).get(str(i), 0)) for i in range(len(cands))]
    z = [math.exp(x / 2) for x in s]  # softmax with temperature 2
    return {c: v / sum(z) for c, v in zip(cands, z)}


def value(path):
    if MODE != "llm":
        return 2 * min(1.0, ppr.get(path[-1], 0) / max(ppr.values())) - 1
    out = ask_json(f"{CONTEXT}\n\nPartial chain:\n{chain_text(path)}\n\n"
                   "Rate 0-3 how promising this partial chain is as part of a correct, chess-meaningful explanation "
                   "of how White wins (3 = directly explains the winning idea, 2 = relevant, 1 = weak, "
                   '0 = irrelevant or misleading). Return JSON {"score": s, "reason": "<10 words>"}.', "eval")
    return (float(out.get("score", 0)) - 1.5) / 1.5


class Node:
    def __init__(self, path, p=1.0):
        self.path, self.p, self.n, self.w = path, p, 0, 0.0
        self.children, self.expanded, self.closed, self.v = [], False, False, None


root = Node(["position"])
accepted, events, tree_nodes = [], [], {}


def key(node):
    return " > ".join(node.path)


T_START = time.time()
for sim in range(1, T + 1):
    CUR["sim"] = sim
    node, trail = root, [root]
    while node.expanded and node.children:
        open_kids = [c for c in node.children if not c.closed]
        if not open_kids:
            node.closed = True; break
        node = max(open_kids, key=lambda c: (c.w / c.n if c.n else 0) + C_PUCT * c.p * math.sqrt(node.n + 1) / (1 + c.n))
        trail.append(node)
    if node.closed and node is root:
        break
    ev = {"sim": sim, "selected": [n for n in node.path]}
    CUR["path"] = node.path
    if node.closed:  # every child already finished: just back up the stored value
        v = node.v if node.v is not None else 0.0
    elif node.path[-1] == "goal":
        v = 1.0
    else:
        cands = candidates(node.path)
        if not cands:
            node.closed, v = True, -1.0
        else:
            P = prior(node.path, cands)
            node.children = [Node(node.path + [c], P[c]) for c in cands]
            node.expanded = True
            ev["expanded"] = [[c, round(P[c], 3)] for c in cands]
            for ch in node.children:
                tree_nodes[key(ch)] = ch
                if ch.path[-1] == "goal":
                    ch.closed = True
                    accepted.append(ch.path); ev.setdefault("accepted", []).append(ch.path)
            v = value(node.path)
    node.v = v
    ev["value"] = round(v, 3)
    for t in trail:
        t.n += 1; t.w += v
    events.append(ev)
    print(f"sim {sim:3d}  v={v:+.2f}  {chain_text(node.path)[:110]}")

sub_nodes = sorted({n for p in accepted for n in p})
sub_edges = sorted({(a, b) for p in accepted for a, b in zip(p, p[1:])})
print(f"\naccepted paths: {len(accepted)}  subgraph: {len(sub_nodes)} nodes, {len(sub_edges)} edges")
for p in accepted:
    print("  *", chain_text(p))

explanation = ""
if MODE == "llm" and accepted:
    calls["explain"] += 1
    facts = "\n".join(f"- {lbl(a)} --{G.edges[a, b]['rel']}--> {lbl(b)}" for a, b in sub_edges)
    CUR["sim"], CUR["path"] = "after search", []
    t0 = time.time()
    EXPL_PROMPT = (f"{CONTEXT}\n\nUse ONLY these verified facts from the knowledge graph:\n{facts}\n\n"
        f"Write a 3-4 sentence explanation for a club player of {SIDE}'s winning move and why it works. "
        "Do not add facts that are not listed.")
    r = client.chat.completions.create(model=MODEL, temperature=0, messages=[{"role": "user", "content": EXPL_PROMPT}])
    log_call("explain", EXPL_PROMPT, r, t0)
    explanation = r.choices[0].message.content.strip()
    print("\nExplanation:\n" + explanation)

motifs = {G.nodes[n]["label"].split(":")[0].split(" ")[0].lower() for n in sub_nodes if G.nodes[n]["kind"] == "motif"}
mapped = {THEME_MAP[m] for m in motifs if m in THEME_MAP}
GENERIC = {"advantage", "crushing", "long", "short", "veryLong", "middlegame", "endgame", "opening", "master",
           "masterVsMaster", "superGM", "oneMove", "equality", "mate", "rookEndgame", "pawnEndgame", "queenEndgame"}
theme_hit = sorted(EXPECTED_THEMES & mapped)
theme_missed = sorted((EXPECTED_THEMES - GENERIC) - mapped)
best_move = results[0][1]
named = any(n == f"move:{best_move}" or n.endswith(f"|0|{best_move}") for n in sub_nodes)
print(f"\nCheck vs ground truth: engine move {best_move} in explanation = {named}; "
      f"motifs {sorted(motifs)} -> themes hit {theme_hit}, missed {theme_missed}")

# Baseline (a): the same LLM, FEN only, no knowledge graph and no engine
baseline = {}
if MODE == "llm":
    calls["baseline"] = 1
    CUR["sim"], CUR["path"] = "baseline", []
    BASE_PROMPT = (f"Chess position (FEN): {FEN}\n{board}\n{SIDE} to move. Find the best move "
                   "and explain in 3-4 sentences why it works. Return JSON {\"move\": \"<SAN>\", \"explanation\": \"...\"}.")
    t0 = time.time()
    r = client.chat.completions.create(model=MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": BASE_PROMPT}])
    log_call("baseline", BASE_PROMPT, r, t0)
    baseline = json.loads(r.choices[0].message.content)
    norm = lambda m: (m or "").replace("+", "").replace("#", "").strip()
    baseline["correct"] = norm(baseline.get("move")) == norm(best_move)
    print("\nBaseline (FEN only):", baseline)
print("LLM calls:", calls)

b0 = chess.Board(FEN); best_uci = b0.parse_san(best_move).uci()
tree = [{"id": key(n), "parent": " > ".join(n.path[:-1]), "node": n.path[-1], "n": n.n,
         "q": round(n.w / n.n, 3) if n.n else None, "p": round(n.p, 3), "v": n.v} for n in tree_nodes.values()]
json.dump({"fen": FEN, "mode": MODE, "model": MODEL if MODE == "llm" else "offline", "T": T, "puzzle": PZ,
           "nodes": {n: {**d} for n, d in G.nodes(data=True)},
           "edges": [[a, b, d["rel"]] for a, b, d in G.edges(data=True)],
           "accepted": accepted, "sub_nodes": sub_nodes, "sub_edges": sub_edges,
           "events": events, "tree": tree, "explanation": explanation, "calls": calls, "baseline": baseline,
           "engine": [[g, s] for g, s, *_ in results[:5]], "engine_full": [list(x) for x in results], "llm_log": LLM_LOG, "search_seconds": round(time.time() - T_START, 1), "ppr": {n: round(v, 6) for n, v in ppr.items()}, "best_move": best_move, "best_uci": best_uci,
           "check": {"engine_move_named": named, "motifs": sorted(motifs), "themes": sorted(EXPECTED_THEMES),
                     "theme_hit": theme_hit, "theme_missed": theme_missed}},
          open(HERE / f"run_{TAG}.json", "w"), indent=1, default=str)
