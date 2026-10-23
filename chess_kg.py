"""Build a typed chess knowledge graph for one position (search layer + concept layer).

Search layer : Position -> candidate Move -> best Reply -> Follow-up move -> Outcome -> Goal
Concept layer: Piece nodes with attacks / defends / pins edges, motif nodes (Check, Fork,
               Hanging, Capture, Threat) that link the moves to the pieces they act on.
Everything is computed from the board with python-chess; a small alpha-beta search
(material only) stands in for Stockfish to label outcomes.
"""
import chess
import networkx as nx

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
NAME = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
        chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}


def material(board, color):
    return sum(VAL[p.piece_type] * (1 if p.color == color else -1) for p in board.piece_map().values())


def alphabeta(board, depth, alpha, beta, color):
    """Negamax on material from `color`'s view; returns (score, principal variation)."""
    if board.is_checkmate():
        return (-1000 if board.turn == color else 1000), []
    if depth == 0 or board.is_game_over():
        return material(board, color), []
    maximizing = board.turn == color
    best, best_pv = (-10**6 if maximizing else 10**6), []
    moves = sorted(board.legal_moves, key=lambda m: (not board.is_capture(m), not board.gives_check(m)))
    for m in moves:
        board.push(m)
        score, pv = alphabeta(board, depth - 1, alpha, beta, color)
        board.pop()
        if (maximizing and score > best) or (not maximizing and score < best):
            best, best_pv = score, [m] + pv
        if maximizing:
            alpha = max(alpha, best)
        else:
            beta = min(beta, best)
        if beta <= alpha:
            break
    return best, best_pv


def pid(board, sq):
    p = board.piece_at(sq)
    return f"piece:{'W' if p.color else 'B'}_{chess.piece_symbol(p.piece_type).upper()}_{chess.square_name(sq)}"


def plabel(board, sq):
    p = board.piece_at(sq)
    return f"{'White' if p.color else 'Black'} {NAME[p.piece_type]} {chess.square_name(sq)}"


def build_kg(fen, depth=4, decisive=3):
    board = chess.Board(fen)
    me = board.turn
    G = nx.DiGraph()
    base = material(board, me)

    def node(n, kind, label, **kw):
        if n not in G:
            G.add_node(n, kind=kind, label=label, **kw)

    def edge(a, b, rel):
        G.add_edge(a, b, rel=rel)

    node("position", "position", f"Position ({'White' if me else 'Black'} to move)")
    node("goal", "goal", f"Goal: decisive advantage (≥ +{decisive} material)")

    # --- concept layer on the current position ---
    for sq, p in board.piece_map().items():
        node(pid(board, sq), "piece", plabel(board, sq), value=VAL[p.piece_type])
    for sq, p in board.piece_map().items():
        for t in board.attacks(sq):
            q = board.piece_at(t)
            if q is None:
                continue
            edge(pid(board, sq), pid(board, t), "attacks" if q.color != p.color else "defends")
        if p.piece_type != chess.KING and board.is_pinned(p.color, sq):
            pinner = [s for s in board.attackers(not p.color, sq)
                      if board.piece_at(s).piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN)]
            for s in pinner:
                edge(pid(board, s), pid(board, sq), "pins")
        # hanging = attacked by opponent, not defended
        if p.piece_type != chess.KING and board.attackers(not p.color, sq) and not board.attackers(p.color, sq):
            h = f"motif:hanging_{chess.square_name(sq)}"
            node(h, "motif", f"Hanging: {plabel(board, sq)}")
            edge(pid(board, sq), h, "is")
            for s in board.attackers(not p.color, sq):
                node(f"motif:threat_{chess.square_name(s)}_{chess.square_name(sq)}", "motif",
                     f"Threat: {plabel(board, s)} can take {plabel(board, sq)}")
                edge(pid(board, s), f"motif:threat_{chess.square_name(s)}_{chess.square_name(sq)}", "creates")
                edge(f"motif:threat_{chess.square_name(s)}_{chess.square_name(sq)}", pid(board, sq), "targets")

    # --- search layer: every legal move, labelled by a shallow search ---
    results = []
    for m in board.legal_moves:
        san = board.san(m)
        mv = f"move:{san}"
        node(mv, "move", san)
        edge("position", mv, "candidate")
        edge(mv, pid(board, m.from_square), "moves")
        if board.is_capture(m) and board.piece_at(m.to_square):
            c = f"motif:capture_{san}"
            node(c, "motif", f"Capture of {plabel(board, m.to_square)}")
            edge(mv, c, "creates"); edge(c, pid(board, m.to_square), "targets")
        board.push(m)
        if board.is_check():
            ck = f"motif:check_{san}"
            node(ck, "motif", f"Check on the {'Black' if me else 'White'} king")
            edge(mv, ck, "gives")
            edge(ck, pid(board, board.king(not me)), "targets")
        # fork: the moved piece attacks >= 2 enemy pieces worth >= 3 (king counts)
        tgts = [t for t in board.attacks(m.to_square)
                if board.piece_at(t) and board.piece_at(t).color != me
                and (board.piece_at(t).piece_type == chess.KING or VAL[board.piece_at(t).piece_type] >= 3)]
        if len(tgts) >= 2:
            fk = f"motif:fork_{san}"
            node(fk, "motif", f"Fork by {NAME[board.piece_at(m.to_square).piece_type]} on {chess.square_name(m.to_square)}")
            edge(mv, fk, "creates")
            board.pop()
            for t in tgts:  # targets refer to pieces of the original position
                edge(fk, pid(board, t), "targets")
            board.push(m)
        score, pv = alphabeta(board, depth - 1, -10**6, 10**6, me)
        board.pop()
        gain = score - base
        results.append((gain, san, [m] + pv))

        # principal variation as search-layer nodes: reply and follow-up
        b2 = board.copy(); b2.push(m)
        prev = mv
        for i, pm in enumerate(pv[:2]):
            psan = b2.san(pm)
            n = f"line:{san}|{i}|{psan}"
            node(n, "reply" if i == 0 else "move", psan + (" (best reply)" if i == 0 else " (follow-up)"))
            edge(prev, n, "best_reply" if i == 0 else "then")
            if i == 0 and f"motif:check_{san}" in G:
                edge(f"motif:check_{san}", n, "answered_by")
            if i == 1 and b2.is_capture(pm) and b2.piece_at(pm.to_square):
                cap_sq = pm.to_square
                # link the captured piece of the ORIGINAL position (if it never moved) to the follow-up
                orig = pid(board, cap_sq) if board.piece_at(cap_sq) else None
                if orig and orig in G:
                    edge(orig, n, "captured_by")
                    b3 = b2.copy(); b3.push(pm)
                    if not b3.attackers(not me, cap_sq):  # nobody can recapture
                        u = f"motif:undefended_{chess.square_name(cap_sq)}"
                        node(u, "motif", f"Undefended: {plabel(board, cap_sq)} (no recapture)")
                        edge(orig, u, "is"); edge(u, n, "allows")
            b2.push(pm)
            prev = n
        oc = f"outcome:{san}"
        node(oc, "outcome", f"Result after {san}: {gain:+d} material", gain=gain)
        edge(prev, oc, "results_in")
        if gain >= decisive:
            edge(oc, "goal", "achieves")
        if gain >= 1000 - 10:
            G.nodes[oc]["label"] = f"Result after {san}: checkmate"

    results.sort(reverse=True)
    return board, G, results


if __name__ == "__main__":
    import sys
    fen = sys.argv[1] if len(sys.argv) > 1 else "r3k2r/pp3ppp/8/1N1q4/8/3P4/PPP2PPP/R2QK2R w - - 0 1"
    board, G, res = build_kg(fen)
    print(board, "\n")
    print("nodes", G.number_of_nodes(), "edges", G.number_of_edges())
    from collections import Counter
    print(Counter(d["kind"] for _, d in G.nodes(data=True)))
    print(Counter(d["rel"] for *_, d in G.edges(data=True)))
    for g, san, pv in res[:6]:
        b = board.copy(); line = []
        for m in pv:
            line.append(b.san(m)); b.push(m)
        print(f"{g:+5d}  {san:7s} {' '.join(line)}")
    print("goal reachable from:", [n for n in G.predecessors("goal")])
