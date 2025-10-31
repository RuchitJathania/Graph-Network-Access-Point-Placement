# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
# from matplotlib.patches import Circle

# ---------------------------
# Utilities
# ---------------------------
def _q(x: float, nd: int = 6) -> float:
    """Quantize to avoid float fragmentation when creating split points."""
    return round(float(x), nd)

def make_weights(miners: List[Dict]) -> Dict[int, float]:
    """
    Build weights per miner index. If 'weight' is missing, default to 1.0.
    e.g., weight = (bytes_up + bytes_down) / 300.0  # bytes/s over last 5 mins
    """
    return {i: float(m.get("weight", 1.0)) for i, m in enumerate(miners)}

# ---------------------------
# Split-graph builder (for along-edge distances)
# ---------------------------
def build_split_graph(
        edges: List[Dict],
        miners: List[Dict],
        candidate_step_m: float,
        include_vertices_as_candidates: bool = True
) -> Tuple[nx.Graph, Dict[Tuple, Tuple[str, float]], Dict[int, Tuple], Dict[str, Tuple]]:
    """
    Build a graph split at all miner positions and at regular candidate steps.
    Returns:
      G: weighted graph with nodes = vertices + inserted split points on edges
      candidate_nodes: dict[node] -> (edge_id, offset_m)
      miner_nodes: dict[miner_idx] -> node
      vertex_map: dict[vertex_id] -> ('v', vertex_id)
    Node encoding:
      ('v', vertex_id) for original vertices
      ('p', edge_id, offset_m) for interior points along an edge
    """
    G = nx.Graph()
    vertex_node = lambda v: ('v', v)

    # Edge info and required offsets
    edge_info = {e["edge_id"]: (e["u"], e["v"], float(e["length_m"])) for e in edges}
    required_offsets = {e["edge_id"]: set() for e in edges}

    # Optionally consider vertices as candidates
    if include_vertices_as_candidates:
        for eid, (u, v, L) in edge_info.items():
            required_offsets[eid].add(_q(0.0))
            required_offsets[eid].add(_q(L))

    # Add miner offsets
    for m in miners:
        eid = m["edge_id"]
        off = _q(m["offset_m"])
        L = edge_info[eid][2]
        required_offsets[eid].add(max(0.0, min(off, L)))

    # Regular candidate grid per edge (plus endpoints)
    for eid, (u, v, L) in edge_info.items():
        step = float(candidate_step_m)
        if step > 0:
            n_steps = max(1, int(math.floor(L / step)))
            for i in range(1, n_steps):
                required_offsets[eid].add(_q(i * step))
        required_offsets[eid].add(_q(0.0))
        required_offsets[eid].add(_q(L))

    # Add original vertices
    for eid, (u, v, L) in edge_info.items():
        G.add_node(vertex_node(u))
        G.add_node(vertex_node(v))

    # Build split chains for each edge
    point_nodes_on_edge = {}  # (eid, off) -> node
    for eid, (u, v, L) in edge_info.items():
        offs = sorted(required_offsets[eid])
        edge_nodes = []
        for off in offs:
            if off == 0.0:
                edge_nodes.append(vertex_node(u))
            elif off == _q(L):
                edge_nodes.append(vertex_node(v))
            else:
                node = ('p', eid, off)
                G.add_node(node)
                point_nodes_on_edge[(eid, off)] = node
                edge_nodes.append(node)
        for a, b, o1, o2 in zip(edge_nodes[:-1], edge_nodes[1:], offs[:-1], offs[1:]):
            seg_len = _q(o2 - o1)
            if seg_len > 0:
                G.add_edge(a, b, weight=seg_len)

    # Candidate positions = all interior points (and optionally vertices)
    candidate_nodes: Dict[Tuple, Tuple[str, float]] = {}
    for (eid, off), node in point_nodes_on_edge.items():
        candidate_nodes[node] = (eid, off)

    if include_vertices_as_candidates:
        # Map each vertex node to one adjacent edge representation (arbitrary if deg>1)
        seen_vertex = set()
        for eid, (u, v, L) in edge_info.items():
            if u not in seen_vertex:
                candidate_nodes[vertex_node(u)] = (eid, 0.0)
                seen_vertex.add(u)
            if v not in seen_vertex:
                candidate_nodes[vertex_node(v)] = (eid, L)
                seen_vertex.add(v)

    # Miner -> graph node
    miner_nodes: Dict[int, Tuple] = {}
    for idx, m in enumerate(miners):
        eid, off = m["edge_id"], _q(m["offset_m"])
        u, v, L = edge_info[eid]
        if off <= 0.0:
            miner_nodes[idx] = vertex_node(u)
        elif off >= _q(L):
            miner_nodes[idx] = vertex_node(v)
        else:
            miner_nodes[idx] = point_nodes_on_edge[(eid, off)]

    return G, candidate_nodes, miner_nodes, {v: vertex_node(v) for v in set([e["u"] for e in edges] + [e["v"] for e in edges])}

def compute_coverage_sets(
        G: nx.Graph,
        candidate_nodes: Dict[Tuple, Tuple[str, float]],
        miner_nodes: Dict[int, Tuple],
        ap_range_m: float
) -> Dict[Tuple, set]:
    """
    For each candidate node, compute the set of miner indices it covers within graph distance <= ap_range_m.
    """
    coverage: Dict[Tuple, set] = {}
    for cnode in candidate_nodes.keys():
        dists = nx.single_source_dijkstra_path_length(G, cnode, cutoff=ap_range_m, weight='weight')
        covered = {idx for idx, mnode in miner_nodes.items() if (mnode in dists and dists[mnode] <= ap_range_m)}
        coverage[cnode] = covered
    return coverage

def compute_aplink_neighbors(
        G: nx.Graph,
        candidate_nodes: Dict[Tuple, Tuple[str, float]],
        ap_link_range_m: float
) -> Dict[Tuple, set]:
    """
    For each candidate AP node, list other candidate AP nodes within AP-to-AP link range (graph distance).
    """
    neighbors: Dict[Tuple, set] = {}
    cand_set = set(candidate_nodes.keys())
    for c in cand_set:
        d = nx.single_source_dijkstra_path_length(G, c, cutoff=ap_link_range_m, weight='weight')
        nbrs = {x for x in cand_set if x is not c and (x in d and d[x] <= ap_link_range_m)}
        neighbors[c] = nbrs
    return neighbors

# ---------------------------
# Greedy weighted max-coverage *with connectivity constraint*
# ---------------------------
def greedy_max_coverage_weighted_connected(
        coverage_sets: Dict[Tuple, set],
        weights: Dict[int, float],
        neighbors: Dict[Tuple, set],
        k: int,
        allow_connectors: bool = True,
) -> List[Tuple]:
    """
    Maintain AP mesh connectivity: every added AP must be within 'ap_link_range_m' of the current mesh.
    If no positive-gain reachable candidate exists and 'allow_connectors' is True, add a connector AP
    (zero immediate gain) that maximizes best next-step potential gain.
    """
    all_miners = set().union(*coverage_sets.values()) if coverage_sets else set()
    uncovered = set(all_miners)
    remaining = set(coverage_sets.keys())
    chosen: List[Tuple] = []

    # pick the best seed (no connectivity yet)
    best = max(remaining, key=lambda c: sum(weights[i] for i in coverage_sets[c]), default=None)
    if best is None:
        return []
    chosen.append(best)
    remaining.remove(best)
    uncovered -= coverage_sets[best]

    # grow the connected mesh
    for _ in range(1, k):
        # candidates that keep connectivity to the current mesh
        frontier = {c for c in remaining if any(c in neighbors[s] for s in chosen)}
        if not frontier:
            break

        # pick best positive-gain candidate on the frontier
        best_c = None
        best_gain = 0.0
        for c in frontier:
            gain = sum(weights[i] for i in (coverage_sets[c] & uncovered))
            if gain > best_gain:
                best_gain, best_c = gain, c

        if best_c is not None and best_gain > 0.0:
            chosen.append(best_c)
            remaining.remove(best_c)
            uncovered -= coverage_sets[best_c]
            continue

        # No positive-gain reachable candidate. Optionally add a connector AP.
        if not allow_connectors:
            break

        # Look ahead one step: add connector 'c' then the best neighbor 'n' reachable via 'c'
        best_connector = None
        best_next_gain = 0.0
        for c in frontier:
            # after adding connector c, any neighbor of {chosen ∪ {c}} keeps mesh connected
            grow_frontier = set()
            grow_frontier |= neighbors[c]
            for s in chosen:
                grow_frontier |= neighbors[s]
            grow_frontier &= remaining
            # best immediate gain if we could add another AP after connector
            next_gain = 0.0
            for n in grow_frontier:
                g = sum(weights[i] for i in (coverage_sets[n] & uncovered))
                if g > next_gain:
                    next_gain = g
            if next_gain > best_next_gain:
                best_next_gain = next_gain
                best_connector = c

        if best_connector is None:
            break  # nowhere to go, stop early

        # add connector (immediate gain may be 0)
        chosen.append(best_connector)
        remaining.remove(best_connector)
        # (do not change 'uncovered' — connector has no/minor coverage)
        # Loop continues; next iteration should find positive-gain candidate via the extended mesh.

    return chosen

# ---------------------------
# Main solver (connected mesh)
# ---------------------------
def place_access_points(
        edges: List[Dict],
        miners: List[Dict],
        k: int,
        ap_range_m: float,
        candidate_step_m: float = None,
        ap_link_range_m: float = None,
        allow_connectors: bool = True,
) -> List[Dict]:
    """
    Return ≤k AP placements maximizing covered weight, with the constraint that all APs form
    ONE connected mesh using AP-to-AP link range 'ap_link_range_m' (defaults to ap_range_m).
    """
    if candidate_step_m is None:
        candidate_step_m = ap_range_m / 3.0
    if ap_link_range_m is None:
        ap_link_range_m = ap_range_m  # same radio/range used for backhaul unless specified

    G, candidates, miner_nodes, _ = build_split_graph(
        edges, miners, candidate_step_m, include_vertices_as_candidates=True
    )
    coverage_sets = compute_coverage_sets(G, candidates, miner_nodes, ap_range_m)
    neighbors = compute_aplink_neighbors(G, candidates, ap_link_range_m)
    weights = make_weights(miners)

    chosen_nodes = greedy_max_coverage_weighted_connected(
        coverage_sets, weights, neighbors, k, allow_connectors=allow_connectors
    )

    placements = []
    for node in chosen_nodes:
        eid, off = candidates[node]
        placements.append({"edge_id": eid, "offset_m": float(off)})
    return placements

# ---------------------------
# Visualization helpers
# ---------------------------
def build_graph_for_draw(edges):
    G = nx.Graph()
    edges_by_id = {}
    for e in edges:
        eid, u, v, L = e["edge_id"], e["u"], e["v"], float(e["length_m"])
        G.add_edge(u, v, weight=L, edge_id=eid, length_m=L)
        edges_by_id[eid] = (u, v, L)
    return G, edges_by_id

def _interp(pu, pv, t):  # t in [0,1]
    return (pu[0]*(1-t) + pv[0]*t, pu[1]*(1-t) + pv[1]*t)

def xy_on_edge(edges_by_id, pos, edge_id, offset_m):
    u, v, L = edges_by_id[edge_id]
    t = 0.0 if L <= 0 else max(0.0, min(1.0, offset_m / L))
    return _interp(pos[u], pos[v], t)

def _split_graph_for_dist(edges, miners, placements, step=1.0):
    """Split graph at miner/AP offsets so along-edge distances are correct for coverage/link checks."""
    def v_node(v): return ('v', v)
    edge_info = {e["edge_id"]: (e["u"], e["v"], float(e["length_m"])) for e in edges}
    req = {eid: {0.0, edge_info[eid][2]} for eid in edge_info}

    for m in miners:
        L = edge_info[m["edge_id"]][2]
        req[m["edge_id"]].add(round(max(0.0, min(L, float(m["offset_m"]))), 6))
    for p in placements:
        L = edge_info[p["edge_id"]][2]
        req[p["edge_id"]].add(round(max(0.0, min(L, float(p["offset_m"]))), 6))

    for eid, (_, _, L) in edge_info.items():
        if step > 0:
            n = int(L // step)
            for i in range(1, n):
                req[eid].add(round(i * step, 6))

    Gs = nx.Graph()
    for eid, (u, v, L) in edge_info.items():
        Gs.add_node(v_node(u)); Gs.add_node(v_node(v))

    pts = {}  # (eid, off) -> node
    for eid, (u, v, L) in edge_info.items():
        offs = sorted(req[eid])
        chain = []
        for off in offs:
            if off == 0.0:
                chain.append(v_node(u))
            elif abs(off - L) < 1e-6:
                chain.append(v_node(v))
            else:
                node = ('p', eid, off); Gs.add_node(node); pts[(eid, off)] = node
                chain.append(node)
        for a, b, o1, o2 in zip(chain[:-1], chain[1:], offs[:-1], offs[1:]):
            seg = round(o2 - o1, 6)
            if seg > 0: Gs.add_edge(a, b, weight=seg)

    def map_point(eid, off):
        off = round(float(off), 6)
        u, v, L = edge_info[eid]
        if off == 0.0: return ('v', u)
        if abs(off - L) < 1e-6: return ('v', v)
        return pts[(eid, off)]

    miner_nodes = {i: map_point(m["edge_id"], m["offset_m"]) for i, m in enumerate(miners)}
    ap_nodes = [map_point(p["edge_id"], p["offset_m"]) for p in placements]
    return Gs, miner_nodes, ap_nodes

def _covered_miners(Gs, miner_nodes, ap_nodes, ap_range_m):
    covered = set()
    for ap in ap_nodes:
        d = nx.single_source_dijkstra_path_length(Gs, ap, cutoff=ap_range_m, weight='weight')
        covered |= {i for i, n in miner_nodes.items() if n in d and d[n] <= ap_range_m}
    return covered

def _ap_mesh_edges(Gs, ap_nodes, ap_link_range_m):
    """Return pairs (i,j) of AP indices that are within AP link range along the graph."""
    edges = []
    for i in range(len(ap_nodes)):
        d = nx.single_source_dijkstra_path_length(Gs, ap_nodes[i], cutoff=ap_link_range_m, weight='weight')
        for j in range(i+1, len(ap_nodes)):
            if ap_nodes[j] in d and d[ap_nodes[j]] <= ap_link_range_m:
                edges.append((i, j))
    return edges

def visualize(edges, miners, placements, ap_range_m, ap_link_range_m=None, layout="spring"):
    """
    Draw side-by-side:
      (a) initial graph + miners,
      (b) graph with AP placements, covered miners, dashed AP range circles, and AP mesh links.
    """
    if ap_link_range_m is None:
        ap_link_range_m = ap_range_m

    Gdraw, edges_by_id = build_graph_for_draw(edges)
    pos = nx.spring_layout(Gdraw, seed=42) if layout == "spring" else nx.kamada_kawai_layout(Gdraw)

    miner_xy = [xy_on_edge(edges_by_id, pos, m["edge_id"], m["offset_m"]) for m in miners]
    ap_xy = [xy_on_edge(edges_by_id, pos, p["edge_id"], p["offset_m"]) for p in placements]

    # Coverage + mesh connectivity checks using along-edge distances
    Gs, miner_nodes, ap_nodes = _split_graph_for_dist(edges, miners, placements, step=max(1.0, ap_range_m/5))
    covered_idxs = _covered_miners(Gs, miner_nodes, ap_nodes, ap_range_m)
    mesh_pairs = _ap_mesh_edges(Gs, ap_nodes, ap_link_range_m)

    # Estimate layout scale to draw dashed range circles in ~meters
    ratios = []
    for u, v, d in Gdraw.edges(data=True):
        L = float(d.get("length_m", 0.0))
        if L > 0:
            x1, y1 = pos[u]; x2, y2 = pos[v]
            draw_len = math.hypot(x1 - x2, y1 - y2)
            if draw_len > 0:
                ratios.append(draw_len / L)
    scale = float(np.median(ratios)) if ratios else 1.0
    px_radius = ap_range_m * scale

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    # (a) Initial graph & miners
    ax = axes[0]
    nx.draw_networkx(Gdraw, pos, ax=ax, node_color="#f0f0f0", node_size=600, edge_color="#888")
    nx.draw_networkx_edge_labels(
        Gdraw, pos, ax=ax,
        edge_labels={(u, v): f'{d["length_m"]:.0f}m' for u, v, d in Gdraw.edges(data=True)}
    )
    if miner_xy:
        ax.scatter([x for x, y in miner_xy], [y for x, y in miner_xy], s=80, marker='o', label="Miner")
    ax.set_title("Graph & miners")
    ax.axis("off")
    ax.legend(loc="lower left")

    # (b) With AP placements + coverage + mesh links + dashed range
    ax = axes[1]
    nx.draw_networkx(Gdraw, pos, ax=ax, node_color="#f0f0f0", node_size=600, edge_color="#888")

    covered_pts = [miner_xy[i] for i in covered_idxs]
    uncovered_pts = [pt for i, pt in enumerate(miner_xy) if i not in covered_idxs]
    if uncovered_pts:
        ax.scatter([x for x, y in uncovered_pts], [y for x, y in uncovered_pts], s=80, marker='o', label="Uncovered")
    if covered_pts:
        ax.scatter([x for x, y in covered_pts], [y for x, y in covered_pts], s=90, marker='o', label="Covered")

    # AP markers
    if ap_xy:
        ax.scatter([x for x, y in ap_xy], [y for x, y in ap_xy], s=140, marker='^', label="AP")

        # # dashed range circles
        # for (x, y) in ap_xy:
        #     circ = Circle((x, y), px_radius, fill=False, linestyle='--', linewidth=1.5)
        #     ax.add_patch(circ)

        # draw AP mesh links (solid lines between AP markers)
        for (i, j) in mesh_pairs:
            x1, y1 = ap_xy[i]; x2, y2 = ap_xy[j]
            ax.plot([x1, x2], [y1, y2], linewidth=2.0, alpha=0.7, label="_nolegend_")

    ax.set_aspect('equal', adjustable='datalim')
    # Add legend proxies
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], linestyle='--', color='black', label=f"AP range ≈ {ap_range_m} m"))
    labels.append(f"AP range ≈ {ap_range_m} m")
    handles.append(plt.Line2D([0], [0], color='black', linewidth=2.0, label="AP mesh link"))
    labels.append("AP mesh link")
    ax.legend(handles, labels, loc="lower right")

    ax.set_title("AP placements: coverage & connected mesh")
    ax.axis("off")
    plt.show()

# ---------------------------
# Demo / Example
# ---------------------------
if __name__ == "__main__":
    edges = [
        {"edge_id": "E1", "u": "A", "v": "B", "length_m": 100.0},
        {"edge_id": "E2", "u": "B", "v": "C", "length_m": 60.0},
        {"edge_id": "E3", "u": "B", "v": "D", "length_m": 80.0},
    ]
    # Offsets measured from u -> v; add 'weight' for weighted coverage (defaults to 1.0).
    miners = [
        {"edge_id": "E1", "offset_m": 20.0, "weight": (2_389_450 + 5_210_390) / 300.0},
        {"edge_id": "E1", "offset_m": 70.0, "weight": (410_221 + 1_803_340) / 300.0},
        {"edge_id": "E2", "offset_m": 45.0, "weight": (0 + 32_611) / 300.0},
        {"edge_id": "E3", "offset_m": 20.0, "weight": (1_200_000 + 900_000) / 300.0},
    ]

    k = 3
    ap_range_m = 35.0
    ap_link_range_m = ap_range_m  # set different if AP backhaul range differs from client coverage

    placements = place_access_points(
        edges, miners, k, ap_range_m,
        candidate_step_m=1,               # defaults to r/3
        ap_link_range_m=ap_link_range_m,     # mesh link range
        allow_connectors=True                # allow 0-gain APs to bridge components
    )
    print("Chosen AP placements:", placements)

    # Coverage & mesh after placement (for reporting)
    Gs, miner_nodes, ap_nodes = _split_graph_for_dist(edges, miners, placements, step=max(1.0, ap_range_m/5))
    covered_idxs = _covered_miners(Gs, miner_nodes, ap_nodes, ap_range_m)
    weights = make_weights(miners)
    covered_weight = sum(weights[i] for i in covered_idxs)
    total_weight = sum(weights.values())
    print(f"Covered miners: {len(covered_idxs)}/{len(miners)} "
          f"({len(covered_idxs)/len(miners)*100:.1f}%)")
    print(f"Weighted coverage: {covered_weight:.1f}/{total_weight:.1f} "
          f"({covered_weight/total_weight*100:.1f}%)")

    visualize(edges, miners, placements, ap_range_m, ap_link_range_m=ap_link_range_m, layout="spring")
