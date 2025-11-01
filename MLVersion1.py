# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple
import math
import numpy as np
import networkx as nx
import helpers

# ---- Optional ML backend (kept minimal) -------------------------------------
try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# =============================================================================
# Utilities & data helpers
# =============================================================================

def _q(x: float, nd: int = 6) -> float:
    """Quantize to avoid float fragmentation."""
    return round(float(x), nd)

def make_weights(miners: List[Dict]) -> Dict[int, float]:
    """Per-miner weights (defaults to 1.0)."""
    return {i: float(m.get("weight", 1.0)) for i, m in enumerate(miners)}

def evaluate_and_visualize(edges, miners, placements, ap_range_m, ap_link_range_m=None, layout="spring"):
    """Print coverage metrics and plot."""
    if ap_link_range_m is None:
        ap_link_range_m = ap_range_m

    covered_idxs, mesh_pairs = helpers.visualize(
        edges, miners, placements, ap_range_m, ap_link_range_m=ap_link_range_m, layout=layout
    )
    weights = make_weights(miners)
    covered_weight = sum(weights[i] for i in covered_idxs)
    total_weight = sum(weights.values())
    print(f"Covered miners: {len(covered_idxs)}/{len(miners)} ({len(covered_idxs)/len(miners)*100:.1f}%)")
    print(f"Weighted coverage: {covered_weight:.1f}/{total_weight:.1f} ({covered_weight/total_weight*100:.1f}%)")

    # Mesh connectivity report
    n_ap = len(placements)
    connected = helpers._ap_mesh_is_connected(mesh_pairs, n_ap)
    comp_text = "connected" if connected or n_ap <= 1 else "DISCONNECTED"
    print(f"AP mesh: {n_ap} nodes, {len(mesh_pairs)} links → {comp_text}")

# =============================================================================
# Graph building (split points at miners + regular grid) and mappings
# =============================================================================

def build_split_graph(
        edges: List[Dict],
        miners: List[Dict],
        candidate_step_m: float,
        include_vertices_as_candidates: bool = True
) -> Tuple[nx.Graph, Dict[Tuple, Tuple[str, float]], Dict[int, Tuple], Dict[str, Tuple], Dict[str, Tuple[str, str, float]]]:
    """
    Build a weighted graph split at miner offsets and regular step positions.
    Returns:
      G: weighted graph (nodes are ('v', vertex_id) or ('p', edge_id, offset))
      candidate_nodes: dict[node] -> (edge_id, offset_m)
      miner_nodes: dict[miner_idx] -> node in G
      vertex_map: dict[vertex_id] -> node
      edge_info: dict[edge_id] -> (u, v, L)
    """
    G = nx.Graph()
    vertex_node = lambda v: ('v', v)

    edge_info = {e["edge_id"]: (e["u"], e["v"], float(e["length_m"])) for e in edges}
    required_offsets = {e["edge_id"]: set() for e in edges}

    if include_vertices_as_candidates:
        for eid, (u, v, L) in edge_info.items():
            required_offsets[eid].add(_q(0.0))
            required_offsets[eid].add(_q(L))

    for m in miners:
        eid = m["edge_id"]; L = edge_info[eid][2]
        off = _q(max(0.0, min(L, float(m["offset_m"]))))
        required_offsets[eid].add(off)

    for eid, (u, v, L) in edge_info.items():
        step = float(candidate_step_m)
        if step > 0:
            n_steps = max(1, int(math.floor(L / step)))
            for i in range(1, n_steps):
                required_offsets[eid].add(_q(i * step))
        required_offsets[eid].add(_q(0.0))
        required_offsets[eid].add(_q(L))

    # add original vertices
    for eid, (u, v, L) in edge_info.items():
        G.add_node(vertex_node(u)); G.add_node(vertex_node(v))

    # build split chains for each edge
    point_nodes_on_edge = {}  # (eid, off) -> node
    for eid, (u, v, L) in edge_info.items():
        offs = sorted(required_offsets[eid])
        chain = []
        for off in offs:
            if off == 0.0:
                chain.append(vertex_node(u))
            elif off == _q(L):
                chain.append(vertex_node(v))
            else:
                node = ('p', eid, off)
                G.add_node(node)
                point_nodes_on_edge[(eid, off)] = node
                chain.append(node)
        for a, b, o1, o2 in zip(chain[:-1], chain[1:], offs[:-1], offs[1:]):
            seg = _q(o2 - o1)
            if seg > 0:
                G.add_edge(a, b, weight=seg)

    # candidates = all interior points (and optionally vertices)
    candidates: Dict[Tuple, Tuple[str, float]] = {}
    for (eid, off), node in point_nodes_on_edge.items():
        candidates[node] = (eid, off)
    if include_vertices_as_candidates:
        seen = set()
        for eid, (u, v, L) in edge_info.items():
            if u not in seen:
                candidates[('v', u)] = (eid, 0.0); seen.add(u)
            if v not in seen:
                candidates[('v', v)] = (eid, L);   seen.add(v)

    # miner -> graph node
    miner_nodes: Dict[int, Tuple] = {}
    for idx, m in enumerate(miners):
        eid, off = m["edge_id"], _q(m["offset_m"])
        u, v, L = edge_info[eid]
        if off <= 0.0: miner_nodes[idx] = ('v', u)
        elif off >= _q(L): miner_nodes[idx] = ('v', v)
        else: miner_nodes[idx] = point_nodes_on_edge[(eid, off)]

    vertex_map = {v: ('v', v) for v in set([e["u"] for e in edges] + [e["v"] for e in edges])}
    return G, candidates, miner_nodes, vertex_map, edge_info

# =============================================================================
# Coverage sets (within client range) and AP-link neighbor sets
# =============================================================================

def compute_coverage_sets(
        G: nx.Graph,
        candidate_nodes: Dict[Tuple, Tuple[str, float]],
        miner_nodes: Dict[int, Tuple],
        ap_range_m: float
) -> Dict[Tuple, set]:
    coverage: Dict[Tuple, set] = {}
    for c in candidate_nodes.keys():
        d = nx.single_source_dijkstra_path_length(G, c, cutoff=ap_range_m, weight='weight')
        coverage[c] = {i for i, mn in miner_nodes.items() if mn in d and d[mn] <= ap_range_m}
    return coverage

def compute_aplink_neighbors(
        G: nx.Graph,
        candidate_nodes: Dict[Tuple, Tuple[str, float]],
        ap_link_range_m: float
) -> Dict[Tuple, set]:
    neighbors: Dict[Tuple, set] = {}
    cand = set(candidate_nodes.keys())
    for c in cand:
        d = nx.single_source_dijkstra_path_length(G, c, cutoff=ap_link_range_m, weight='weight')
        neighbors[c] = {x for x in cand if (x != c and (x in d and d[x] <= ap_link_range_m))}
    return neighbors

# =============================================================================
# Candidate features for ML scoring
# =============================================================================

def candidate_features(
        G: nx.Graph,
        candidates: Dict[Tuple, Tuple[str, float]],
        miner_nodes: Dict[int, Tuple],
        miners: List[Dict],
        edge_info: Dict[str, Tuple[str, str, float]],
        ap_range_m: float
) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Build a small, generic feature vector per candidate.
    Features (per p):
      - Weighted miner density within r/2, r, 2r
      - Count of miners within r
      - min/mean distance to miners within r (fallback large if none)
      - Edge-relative: offset_norm, (1-offset_norm), edge_length
    """
    weights = make_weights(miners)
    idx_to_node = miner_nodes
    miner_nodes_list = list(idx_to_node.items())  # [(i, node), ...]
    miner_idx = [i for i, _ in miner_nodes_list]

    feats = []
    order = []
    for p, (eid, off) in candidates.items():
        L = edge_info[eid][2]
        off = float(off)
        # distances to miners (truncate at 2r to limit Dijkstra)
        d = nx.single_source_dijkstra_path_length(G, p, cutoff=2.0 * ap_range_m, weight='weight')

        # aggregations
        w_r2 = sum(weights[i] for i, n in miner_nodes_list if (n in d and d[n] <= 0.5 * ap_range_m))
        w_r  = sum(weights[i] for i, n in miner_nodes_list if (n in d and d[n] <= ap_range_m))
        w_2r = sum(weights[i] for i, n in miner_nodes_list if (n in d and d[n] <= 2.0 * ap_range_m))
        c_r  = sum(1 for _, n in miner_nodes_list if (n in d and d[n] <= ap_range_m))

        d_within = [d[n] for _, n in miner_nodes_list if n in d and d[n] <= ap_range_m]
        if d_within:
            dmin = float(min(d_within))
            dmean = float(sum(d_within) / len(d_within))
        else:
            dmin, dmean = ap_range_m, ap_range_m

        offset_norm = off / L if L > 0 else 0.0

        feats.append([
            w_r2, w_r, w_2r, c_r,
            dmin, dmean,
            L, offset_norm, 1.0 - offset_norm
        ])
        order.append(p)

    X = np.asarray(feats, dtype=float)
    return X, order

# =============================================================================
# Oracle (teacher) to produce labels for training (use your real labels if available)
# =============================================================================

def greedy_max_coverage_weighted_connected_teacher(
        coverage_sets: Dict[Tuple, set],
        weights: Dict[int, float],
        neighbors: Dict[Tuple, set],
        k: int,
        allow_connectors: bool = True
) -> List[Tuple]:
    """Connected greedy teacher used to label training snapshots."""
    all_miners = set().union(*coverage_sets.values()) if coverage_sets else set()
    uncovered = set(all_miners)
    remaining = set(coverage_sets.keys())
    chosen: List[Tuple] = []

    # seed: best standalone coverage
    best = max(remaining, key=lambda c: sum(weights[i] for i in coverage_sets[c]), default=None)
    if best is None:
        return []
    chosen.append(best); remaining.remove(best); uncovered -= coverage_sets[best]

    while len(chosen) < k:
        frontier = {c for s in chosen for c in neighbors[s]} & remaining
        if not frontier:
            break

        best_c, best_gain = None, 0.0
        for c in frontier:
            gain = sum(weights[i] for i in (coverage_sets[c] & uncovered))
            if gain > best_gain:
                best_gain, best_c = gain, c

        if best_c is not None and best_gain > 0.0:
            chosen.append(best_c); remaining.remove(best_c); uncovered -= coverage_sets[best_c]
            continue

        if not allow_connectors:
            break

        # one-step look-ahead connector
        best_connector, best_next_gain = None, 0.0
        for c in frontier:
            grow_frontier = set(neighbors[c])
            for s in chosen:
                grow_frontier |= neighbors[s]
            grow_frontier &= remaining
            next_gain = 0.0
            for n in grow_frontier:
                g = sum(weights[i] for i in (coverage_sets[n] & uncovered))
                if g > next_gain:
                    next_gain = g
            if next_gain > best_next_gain:
                best_next_gain, best_connector = next_gain, c

        if best_connector is None:
            break
        chosen.append(best_connector); remaining.remove(best_connector)
        # do not change uncovered (connector)

    return chosen

# =============================================================================
# Learned scorer + connected selection at inference
# =============================================================================

class LearnedScorer:
    """
    Simple wrapper around a scikit-learn model that scores candidates
    from hand-crafted features. Replace with any model you like.
    """
    def __init__(self):
        if not SKLEARN_OK:
            self.model = None  # fallback to a trivial heuristic
        else:
            # Easy, robust baseline; increase max_iter if you have many snapshots
            self.model = LogisticRegression(max_iter=200, solver="lbfgs")

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            # no-op fallback
            return
        self.model.fit(X, y)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            # fallback: simple heuristic = sum of first three features (w_r2 + w_r + w_2r)
            return X[:, 0] + X[:, 1] + X[:, 2]
        # probability of class 1 as score (shape: [N, 2])
        proba = self.model.predict_proba(X)
        return proba[:, 1]

def connected_selection_with_scores(
        scores: Dict[Tuple, float],
        coverage_sets: Dict[Tuple, set],
        weights: Dict[int, float],
        neighbors: Dict[Tuple, set],
        k: int,
        allow_connectors: bool = True
) -> List[Tuple]:
    """
    Connected greedy using learned scores:
    pick argmax over frontier of (score * marginal_coverage_weight).
    If frontier has no positive gain, place a one-step connector that unlocks
    the best next-step (also weighted by score).
    """
    all_miners = set().union(*coverage_sets.values()) if coverage_sets else set()
    uncovered = set(all_miners)
    remaining = set(coverage_sets.keys())
    chosen: List[Tuple] = []

    def marginal_gain(c):
        return sum(weights[i] for i in (coverage_sets[c] & uncovered))

    # seed by score * total coverage
    best = max(remaining, key=lambda c: (scores[c] * sum(weights[i] for i in coverage_sets[c])), default=None)
    if best is None:
        return []
    chosen.append(best); remaining.remove(best); uncovered -= coverage_sets[best]

    while len(chosen) < k:
        frontier = {c for s in chosen for c in neighbors[s]} & remaining
        if not frontier:
            break

        # best frontier (score * marginal gain)
        best_c, best_obj = None, 0.0
        for c in frontier:
            obj = scores[c] * marginal_gain(c)
            if obj > best_obj:
                best_obj, best_c = obj, c

        if best_c is not None and marginal_gain(best_c) > 0.0:
            chosen.append(best_c); remaining.remove(best_c); uncovered -= coverage_sets[best_c]
            continue

        if not allow_connectors:
            break

        # one-step connector that unlocks best next-step
        best_connector, best_next_obj = None, 0.0
        for c in frontier:
            grow_frontier = set(neighbors[c])
            for s in chosen:
                grow_frontier |= neighbors[s]
            grow_frontier &= remaining

            next_obj = 0.0
            for n in grow_frontier:
                obj = scores[n] * sum(weights[i] for i in (coverage_sets[n] & uncovered))
                if obj > next_obj:
                    next_obj = obj
            if next_obj > best_next_obj:
                best_next_obj, best_connector = next_obj, c

        if best_connector is None:
            break

        chosen.append(best_connector); remaining.remove(best_connector)
        # do not change 'uncovered' for pure connectors

    return chosen

# =============================================================================
# Training dataset builder (toy synthetic; replace with your real snapshots)
# =============================================================================

def synth_random_miners(edges, n_miners=12, seed=0):
    """Toy miner generator on given edges; replace with your real data loader."""
    rng = np.random.default_rng(seed)
    miners = []
    for _ in range(n_miners):
        e = edges[rng.integers(0, len(edges))]
        L = float(e["length_m"])
        offset = float(rng.uniform(0, L))
        weight = float(rng.uniform(0.5, 3.0)) * 1000.0  # toy "bandwidth-derived" weight
        miners.append({"edge_id": e["edge_id"], "offset_m": offset, "weight": weight})
    return miners

def build_training_matrix_from_snapshots(
        snapshots: List[Dict],
        ap_range_m: float,
        ap_link_range_m: float,
        k: int,
        candidate_step_m: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each snapshot:
      - build candidates, coverage sets, neighbors
      - label '1' the candidates chosen by the oracle teacher; others '0'
      - compute features per candidate
    Returns:
      X: [N_candidates_total, n_features], y: [N_candidates_total]
    """
    X_all, y_all = [], []
    for snap in snapshots:
        edges = snap["edges"]; miners = snap["miners"]
        G, candidates, miner_nodes, _, edge_info = build_split_graph(edges, miners, candidate_step_m, True)
        cov = compute_coverage_sets(G, candidates, miner_nodes, ap_range_m)
        nbr = compute_aplink_neighbors(G, candidates, ap_link_range_m)
        w = make_weights(miners)

        chosen = greedy_max_coverage_weighted_connected_teacher(cov, w, nbr, k, allow_connectors=True)
        chosen_set = set(chosen)

        X, order = candidate_features(G, candidates, miner_nodes, miners, edge_info, ap_range_m)
        y = np.array([1 if p in chosen_set else 0 for p in order], dtype=int)

        X_all.append(X); y_all.append(y)

    X_all = np.vstack(X_all) if X_all else np.zeros((0, 1))
    y_all = np.concatenate(y_all) if y_all else np.zeros((0,), dtype=int)
    return X_all, y_all

# =============================================================================
# Public API: train_once(...) and infer_once(...)
# =============================================================================

class APPlacementModel:
    def __init__(self, candidate_step_m: float, ap_range_m: float, ap_link_range_m: float, k: int):
        self.candidate_step_m = candidate_step_m
        self.ap_range_m = ap_range_m
        self.ap_link_range_m = ap_link_range_m
        self.k = k
        self.scorer = LearnedScorer()

    def train_once(self, training_snapshots: List[Dict]):
        """
        Train the scorer once (offline). Replace synthetic snapshots with real data.
        Each snapshot = {"edges": [...], "miners": [...]}
        """
        if not training_snapshots:
            raise ValueError("No training data provided.")
        X, y = build_training_matrix_from_snapshots(
            training_snapshots, self.ap_range_m, self.ap_link_range_m, self.k, self.candidate_step_m
        )
        if X.shape[0] == 0:
            raise ValueError("Empty feature matrix; check training snapshots.")
        self.scorer.fit(X, y)

    def infer_once(self, edges: List[Dict], miners: List[Dict]) -> List[Dict]:
        """
        Given a single 5-min snapshot (edges, miners), return up to k AP placements.
        """
        # Build candidates + precomputations
        G, candidates, miner_nodes, _, edge_info = build_split_graph(
            edges, miners, self.candidate_step_m, include_vertices_as_candidates=True
        )
        cov = compute_coverage_sets(G, candidates, miner_nodes, self.ap_range_m)
        nbr = compute_aplink_neighbors(G, candidates, self.ap_link_range_m)
        w = make_weights(miners)

        # Score candidates
        X, order = candidate_features(G, candidates, miner_nodes, miners, edge_info, self.ap_range_m)
        raw_scores = self.scorer.score(X)
        scores = {order[i]: float(raw_scores[i]) for i in range(len(order))}

        # Connected selection with learned scores
        chosen_nodes = connected_selection_with_scores(scores, cov, w, nbr, self.k, allow_connectors=True)

        # Convert chosen nodes to {edge_id, offset_m}
        placements = []
        for node in chosen_nodes:
            eid, off = candidates[node]
            placements.append({"edge_id": eid, "offset_m": float(off)})
        return placements

# =============================================================================
# Example usage (replace training with your historical dataset)
# =============================================================================
if __name__ == "__main__":
    # Mine layout and current 5-min snapshot (your provided example)
    edges = [
        {"edge_id": "E1", "u": "A", "v": "B", "length_m": 100.0},
        {"edge_id": "E2", "u": "B", "v": "C", "length_m": 60.0},
        {"edge_id": "E3", "u": "B", "v": "D", "length_m": 80.0},
    ]
    miners = [
        {"edge_id": "E1", "offset_m": 20.0, "weight": (2_389_450 + 5_210_390) / 300.0},
        {"edge_id": "E1", "offset_m": 70.0, "weight": (410_221 + 1_803_340) / 300.0},
        {"edge_id": "E2", "offset_m": 45.0, "weight": (0 + 32_611) / 300.0},
        {"edge_id": "E3", "offset_m": 20.0, "weight": (1_200_000 + 900_000) / 300.0},
    ]

    k = 3
    ap_range_m = 35.0
    ap_link_range_m = ap_range_m
    candidate_step_m = 1
    # candidate_step_m = max(1.0, ap_range_m / 3.0) # discretization granularity

    # -------- Train once (offline) -------------
    # Replace this synthetic generation with your real historical snapshots + oracle labels.
    rng = np.random.default_rng(123)
    training_snapshots = []
    for t in range(80):  # e.g., 80 synthetic windows
        miners_t = synth_random_miners(edges, n_miners=10 + rng.integers(0, 6), seed=int(rng.integers(1e9)))
        training_snapshots.append({"edges": edges, "miners": miners_t})

    model = APPlacementModel(candidate_step_m, ap_range_m, ap_link_range_m, k)
    if SKLEARN_OK:
        model.train_once(training_snapshots)
    else:
        print("[WARN] scikit-learn not available; using a simple heuristic scorer.")

    # -------- Inference every 5 min ------------
    placements = model.infer_once(edges, miners)
    print("Predicted AP placements:", placements)

    # Metrics + plot
    evaluate_and_visualize(edges, miners, placements, ap_range_m, ap_link_range_m=ap_link_range_m, layout="spring")
