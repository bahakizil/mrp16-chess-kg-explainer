"""Engine-built chess knowledge graph for a Lichess puzzle (search layer + concept layer).

Search layer : position -> top-k candidate moves (Stockfish multipv) -> principal variation
               (alternating our moves / best replies) -> outcome (engine eval) -> goal
Concept layer: motifs detected with python-chess after every move of every line
               (check, capture, fork, pin, sacrifice, hanging piece, checkmate), linked to the
               pieces they target; pieces link forward to the move that later captures them.
Every edge carries `src` = "stockfish" or "python-chess", i.e. it is verifiable
(Péter: "given position --move--> check is verifiable").
"""
import chess, chess.engine
import networkx as nx

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100}
NAME = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
        chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}
# our motif kinds -> Lichess theme tags they correspond to
THEME_MAP = {"fork": "fork", "pin": "pin", "sacrifice": "sacrifice", "checkmate": "mate",
             "hanging": "hangingPiece", "discovered": "discoveredAttack"}


def plabel(color, ptype, sq):
    return f"{'White' if color else 'Black'} {NAME[ptype]} {chess.square_name(sq)}"


def pkey(color, ptype, sq):
    return f"piece:{'W' if color else 'B'}{chess.piece_symbol(ptype).upper()}@{chess.square_name(sq)}"


def score_text(s):
    if s.is_mate():
        return f"mate in {s.mate()}" if s.mate() > 0 else f"mated in {-s.mate()}"
    return f"{s.score() / 100:+.2f}"


def build_kg_engine(fen, multipv=5, depth=18, plies=6, goal_cp=300):
    board = chess.Board(fen)
    me = board.turn
    G = nx.DiGraph()

    def node(n, kind, label, **kw):
        if n not in G:
            G.add_node(n, kind=kind, label=label, **kw)

    def edge(a, b, rel, src="python-chess"):
        G.add_edge(a, b, rel=rel, src=src)

    node("position", "position", f"Position ({'White' if me else 'Black'} to move)")
    node("goal", "goal", f"Goal: decisive advantage (≥ +{goal_cp / 100:.0f} or forced mate)")

    with chess.engine.SimpleEngine.popen_uci("stockfish") as eng:
        infos = eng.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
        results = []
        for info in infos:
            pv = info["pv"][:plies]
            sc = info["score"].pov(me)
            b = board.copy()
            prev, cand = "position", None
            pending = []   # (motif, relation) to link to the next move of this line
            for i, mv in enumerate(pv):
                mover = b.turn
                san = b.san(mv)
                if i == 0:
                    cand = san
                nid = f"line:{cand}|{i}|{san}"
                kind = "move" if mover == me else "reply"
                node(nid, kind, f"{san}" + ("" if i == 0 else (" (reply)" if kind == "reply" else " (then)")))
                edge(prev, nid, "candidate" if i == 0 else ("best_reply" if kind == "reply" else "then"), "stockfish")
                for m_, rel_ in pending:
                    edge(m_, nid, rel_)
                pending = []
                # pieces captured by this move were created earlier -> link forward
                if b.is_capture(mv):
                    cap_sq = mv.to_square if not b.is_en_passant(mv) else mv.to_square + (-8 if mover else 8)
                    cp = b.piece_at(cap_sq)
                    pk = pkey(cp.color, cp.piece_type, cap_sq)
                    node(pk, "piece", plabel(cp.color, cp.piece_type, cap_sq))
                    edge(pk, nid, "captured_by")
                pinned_before = {s for s in chess.SQUARES if b.piece_at(s) and b.piece_at(s).color != mover
                                 and b.is_pinned(not mover, s)}
                att_before = {sq: set(b.attacks(sq)) for sq in chess.SQUARES
                              if b.piece_at(sq) and b.piece_at(sq).piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN)}
                moved_val = VAL[b.piece_at(mv.from_square).piece_type]
                cap_val = VAL[b.piece_at(mv.to_square).piece_type] if b.piece_at(mv.to_square) else 0
                b.push(mv)
                to = mv.to_square
                mp = b.piece_at(to)
                tag = f"{cand}|{i}"
                if b.is_checkmate():
                    m = f"motif:mate|{tag}"; node(m, "motif", "Checkmate")
                    edge(nid, m, "delivers")
                elif b.is_check():
                    m = f"motif:check|{tag}"
                    answers = [b.san(x) for x in b.legal_moves]
                    node(m, "motif", f"Check on the {'Black' if mover else 'White'} king "
                                     f"(legal answers: {', '.join(answers[:6])}{'…' if len(answers) > 6 else ''})")
                    edge(nid, m, "gives")
                    pending.append((m, "answered_by"))
                    k = b.king(not mover); kk = pkey(not mover, chess.KING, k)
                    node(kk, "piece", plabel(not mover, chess.KING, k)); edge(m, kk, "targets")
                # fork: moved piece attacks >= 2 valuable enemy pieces
                tg = [t for t in b.attacks(to) if b.piece_at(t) and b.piece_at(t).color != mover
                      and VAL[b.piece_at(t).piece_type] >= 3]
                if len(tg) >= 2 and mover == me:
                    m = f"motif:fork|{tag}"
                    node(m, "motif", f"Fork by {NAME[mp.piece_type]} on {chess.square_name(to)}")
                    edge(nid, m, "creates")
                    for t in tg:
                        q = b.piece_at(t); pk = pkey(q.color, q.piece_type, t)
                        node(pk, "piece", plabel(q.color, q.piece_type, t)); edge(m, pk, "targets")
                # new absolute pins against the opponent
                for s in chess.SQUARES:
                    q = b.piece_at(s)
                    if mover == me and q and q.color != mover and q.piece_type != chess.KING and b.is_pinned(q.color, s) and s not in pinned_before:
                        m = f"motif:pin|{tag}|{chess.square_name(s)}"
                        node(m, "motif", f"Pin: {plabel(q.color, q.piece_type, s)} pinned to its king")
                        edge(nid, m, "creates")
                        pk = pkey(q.color, q.piece_type, s)
                        node(pk, "piece", plabel(q.color, q.piece_type, s)); edge(m, pk, "targets")
                # sacrifice: we leave a more valuable piece en prise and the reply takes it
                if mover == me and i + 1 < len(pv) and pv[i + 1].to_square == to and moved_val >= cap_val + 2:
                    m = f"motif:sac|{tag}"
                    node(m, "motif", f"Sacrifice: {NAME[mp.piece_type]} on {chess.square_name(to)} is given up")
                    edge(nid, m, "makes")
                    pending.append((m, "accepted_by"))
                # pin that shields the moved piece: an enemy piece attacks it but may not capture (pinned to its king)
                if mover == me:
                    for s2 in b.attackers(not mover, to):
                        q = b.piece_at(s2)
                        if q.piece_type != chess.KING and b.is_pinned(q.color, s2) and not b.is_legal(chess.Move(s2, to)):
                            m = f"motif:pinshield|{tag}|{chess.square_name(s2)}"
                            node(m, "motif", f"Pin: {plabel(q.color, q.piece_type, s2)} is pinned to its king, "
                                             f"so it cannot take on {chess.square_name(to)}")
                            edge(nid, m, "exploits")
                            pk = pkey(q.color, q.piece_type, s2); node(pk, "piece", plabel(q.color, q.piece_type, s2)); edge(m, pk, "targets")
                            mk = pkey(mp.color, mp.piece_type, to); node(mk, "piece", plabel(mp.color, mp.piece_type, to)); edge(m, mk, "shields")
                # lines opened by the move: a slider that did not move now attacks a valuable piece it did not attack before
                for sq, before in att_before.items():
                    if sq in (mv.from_square, mv.to_square) or not b.piece_at(sq):
                        continue
                    sl = b.piece_at(sq)
                    for t in set(b.attacks(sq)) - before:
                        q = b.piece_at(t)
                        if q and q.color != sl.color and 3 <= VAL[q.piece_type] < 100 and sl.color == me:
                            m = f"motif:open|{tag}|{chess.square_name(sq)}{chess.square_name(t)}"
                            who = "Discovered attack" if sl.color == mover else "Opened line"
                            node(m, "motif", f"{who}: after {san}, {plabel(sl.color, sl.piece_type, sq)} attacks {plabel(q.color, q.piece_type, t)}")
                            edge(nid, m, "opens")
                            pk = pkey(q.color, q.piece_type, t); node(pk, "piece", plabel(q.color, q.piece_type, t)); edge(m, pk, "targets")
                # hanging enemy pieces (attacked by us, undefended), value >= 3
                for s in chess.SQUARES:
                    q = b.piece_at(s)
                    if mover == me and q and q.color != mover and 3 <= VAL[q.piece_type] < 100 and b.attackers(mover, s) and not b.attackers(q.color, s):
                        m = f"motif:hang|{tag}|{chess.square_name(s)}"
                        node(m, "motif", f"Hanging: {plabel(q.color, q.piece_type, s)}")
                        edge(nid, m, "leaves")
                        pk = pkey(q.color, q.piece_type, s)
                        node(pk, "piece", plabel(q.color, q.piece_type, s)); edge(m, pk, "targets")
                prev = nid
            oc = f"outcome:{cand}"
            node(oc, "outcome", f"After {cand}: {score_text(sc)} (Stockfish d{depth})",
                 cp=sc.score(mate_score=100000))
            edge(prev, oc, "results_in", "stockfish")
            if sc.score(mate_score=100000) >= goal_cp:
                edge(oc, "goal", "achieves", "stockfish")
            bb = board.copy(); san_line = []
            for m in pv:
                san_line.append(bb.san(m)); bb.push(m)
            results.append((sc.score(mate_score=100000), cand, san_line, score_text(sc)))
    results.sort(reverse=True)
    return board, G, results


def load_puzzle(pid, path):
    import csv
    for r in csv.DictReader(open(path)):
        if r["PuzzleId"] == pid:
            b = chess.Board(r["FEN"])
            moves = r["Moves"].split()
            b.push_uci(moves[0])  # Lichess FEN is before the opponent's last move
            sol, bb = [], b.copy()
            for u in moves[1:]:
                m = chess.Move.from_uci(u); sol.append(bb.san(m)); bb.push(m)
            return {"id": pid, "fen": b.fen(), "rating": int(r["Rating"]), "themes": r["Themes"].split(),
                    "solution": sol, "plays": int(r["NbPlays"]), "url": r["GameUrl"]}
    raise KeyError(pid)


if __name__ == "__main__":
    import sys
    from collections import Counter
    from pathlib import Path
    pz = load_puzzle(sys.argv[1] if len(sys.argv) > 1 else "2Tl6d", Path(__file__).with_name("puzzles_2500.csv"))
    print(pz)
    board, G, res = build_kg_engine(pz["fen"], plies=len(pz["solution"]) + 1)
    print(board)
    print("nodes", G.number_of_nodes(), "edges", G.number_of_edges())
    print(Counter(d["kind"] for _, d in G.nodes(data=True)))
    print(Counter(d["rel"] for *_, d in G.edges(data=True)))
    for sc, cand, *_ in res:
        print(f"{sc:+7d}  {cand}")
    print("goal from:", list(G.predecessors("goal")))
    print("motifs:", sorted({d['label'] for _, d in G.nodes(data=True) if d['kind'] == 'motif'}))
