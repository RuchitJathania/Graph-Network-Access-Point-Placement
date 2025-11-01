import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import numpy as np
import networkx as nx

def build_graph_for_draw(edges):
    G = nx.Graph()
    edges_by_id = {}
    for e in edges:
        eid, u, v, L = e["edge_id"], e["u"], e["v"], float(e["length_m"])
        G.add_edge(u, v, weight=L, edge_id=eid, length_m=L)
        edges_by_id[eid] = (u, v, L)
    return G, edges_by_id

def _interp(pu, pv, t):
    return (pu[0]*(1-t) + pv[0]*t, pu[1]*(1-t) + pv[1]*t)

def xy_on_edge(edges_by_id, pos, edge_id, offset_m):
    u, v, L = edges_by_id[edge_id]
    t = 0.0 if L <= 0 else max(0.0, min(1.0, offset_m / L))
    return _interp(pos[u], pos[v], t)

def _split_graph_for_dist(edges, miners, placements, step=1.0):
    """Split edges at miner/AP offsets (and a regular grid) for along-edge distances."""
    def v_node(v): return ('v', v)
    edge_info = {e["edge_id"]: (e["u"], e["v"], float(e["length_m"])) for e in edges}
    req = {eid: {0.0, edge_info[eid][2]} for eid in edge_info}

    for m in miners:
        eid = m["edge_id"]; L = edge_info[eid][2]
        off = float(m["offset_m"]); off = max(0.0, min(L, off))
        req[eid].add(round(off, 6))
    for p in placements:
        eid = p["edge_id"]; L = edge_info[eid][2]
        off = float(p["offset_m"]); off = max(0.0, min(L, off))
        req[eid].add(round(off, 6))

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
            if seg > 0:
                Gs.add_edge(a, b, weight=seg)

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
    pairs = []
    for i in range(len(ap_nodes)):
        d = nx.single_source_dijkstra_path_length(Gs, ap_nodes[i], cutoff=ap_link_range_m, weight='weight')
        for j in range(i+1, len(ap_nodes)):
            if ap_nodes[j] in d and d[ap_nodes[j]] <= ap_link_range_m:
                pairs.append((i, j))
    return pairs

def _ap_mesh_is_connected(mesh_pairs, n):
    if n <= 1: return True
    H = nx.Graph(); H.add_nodes_from(range(n)); H.add_edges_from(mesh_pairs)
    return nx.number_connected_components(H) == 1

def visualize(edges, miners, placements, ap_range_m, ap_link_range_m=None, layout="spring"):
    """Side-by-side: (a) graph+miners, (b) AP placements + covered miners + mesh links + dashed AP range."""
    if ap_link_range_m is None:
        ap_link_range_m = ap_range_m

    Gdraw, edges_by_id = build_graph_for_draw(edges)
    pos = nx.spring_layout(Gdraw, seed=42) if layout == "spring" else nx.kamada_kawai_layout(Gdraw)

    miner_xy = [xy_on_edge(edges_by_id, pos, m["edge_id"], m["offset_m"]) for m in miners]
    ap_xy = [xy_on_edge(edges_by_id, pos, p["edge_id"], p["offset_m"]) for p in placements]

    # Coverage + mesh using along-edge distances
    Gs, miner_nodes, ap_nodes = _split_graph_for_dist(edges, miners, placements, step=max(1.0, ap_range_m/5))
    covered_idxs = _covered_miners(Gs, miner_nodes, ap_nodes, ap_range_m)
    mesh_pairs = _ap_mesh_edges(Gs, ap_nodes, ap_link_range_m)

    # scale for drawing dashed circles
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

    # (a) Graph & miners
    ax = axes[0]
    nx.draw_networkx(Gdraw, pos, ax=ax, node_color="#f0f0f0", node_size=600, edge_color="#888")
    nx.draw_networkx_edge_labels(
        Gdraw, pos, ax=ax,
        edge_labels={(u, v): f'{d["length_m"]:.0f}m' for u, v, d in Gdraw.edges(data=True)}
    )
    if miner_xy:
        ax.scatter([x for x, y in miner_xy], [y for x, y in miner_xy], s=80, marker='o', label="Miner")
    ax.set_title("Graph & miners")
    ax.axis("off"); ax.legend(loc="lower left")

    # (b) Placements, coverage, mesh
    ax = axes[1]
    nx.draw_networkx(Gdraw, pos, ax=ax, node_color="#f0f0f0", node_size=600, edge_color="#888")

    covered_pts = [miner_xy[i] for i in covered_idxs]
    uncovered_pts = [pt for i, pt in enumerate(miner_xy) if i not in covered_idxs]
    if uncovered_pts:
        ax.scatter([x for x, y in uncovered_pts], [y for x, y in uncovered_pts], s=80, marker='o', label="Uncovered")
    if covered_pts:
        ax.scatter([x for x, y in covered_pts], [y for x, y in covered_pts], s=90, marker='o', label="Covered")

    if ap_xy:
        ax.scatter([x for x, y in ap_xy], [y for x, y in ap_xy], s=140, marker='^', label="AP")
        # dashed range circles
        for (x, y) in ap_xy:
            circ = Circle((x, y), px_radius, fill=False, linestyle='--', linewidth=1.5)
            ax.add_patch(circ)
        # mesh links
        for (i, j) in mesh_pairs:
            x1, y1 = ap_xy[i]; x2, y2 = ap_xy[j]
            ax.plot([x1, x2], [y1, y2], linewidth=2.0, alpha=0.7, label="_nolegend_")

    ax.set_aspect('equal', adjustable='datalim')
    # Legend proxies
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], linestyle='--', color='black', label=f"AP range ≈ {ap_range_m} m"))
    labels.append(f"AP range ≈ {ap_range_m} m")
    handles.append(plt.Line2D([0], [0], color='black', linewidth=2.0, label="AP mesh link"))
    labels.append("AP mesh link")
    ax.legend(handles, labels, loc="lower right")

    ax.set_title("AP placements: coverage & connected mesh")
    ax.axis("off")
    plt.show()

    return covered_idxs, mesh_pairs

