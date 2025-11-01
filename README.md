# Graph-Constrained AP Placement in Mines

This repository contains a **graph-aware access point (AP) placement** toolkit for underground mines (or any tunnel-like environment). Communication is restricted to shafts that form a weighted graph; APs have limited client range and must also form a **connected backhaul mesh**. We provide:

- A **connectivity-aware greedy baseline** for (weighted) maximum coverage on graphs  
- A **train-once, infer-often ML scorer** that ranks candidate sites and selects a connected set  
- Utilities for **along-edge distances**, **candidate discretization**, **coverage/mesh computation**, and **visualization** (miners, APs, dashed coverage circles, mesh links)  
- Reproducible **toy examples** and **metrics** (covered count/weight, mesh connectivity)

---

## Problem (Brief)

- Mine is an **undirected, edge-weighted graph** `G=(V,E)`, with edge length `l_e` (meters).  
- Miners/devices lie anywhere along edges at offsets `(e, x)`.  
- Each AP covers miners within along-graph distance `r`.  
- APs must form a **connected mesh** with AP-to-AP link range `r_link`.  
- Goal (per 5-minute window): place up to `k` APs to **maximize covered (weighted) miners** while keeping the AP mesh **connected**.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

Minimal `requirements.txt`:

```
networkx
numpy
matplotlib
scikit-learn
```

---

## Data Format

**Edges** (meters; offsets measured from `u` toward `v`):

```python
edges = [
  {"edge_id": "E1", "u": "A", "v": "B", "length_m": 100.0},
  {"edge_id": "E2", "u": "B", "v": "C", "length_m": 60.0},
  {"edge_id": "E3", "u": "B", "v": "D", "length_m": 80.0},
]
```

**Miners** (optional `weight` from traffic over last 5 minutes):

```python
miners = [
  {"edge_id": "E1", "offset_m": 20.0, "weight": (2_389_450 + 5_210_390) / 300.0},
  {"edge_id": "E1", "offset_m": 70.0, "weight": (410_221   + 1_803_340) / 300.0},
  {"edge_id": "E2", "offset_m": 45.0, "weight": (0         +   32_611) / 300.0},
  {"edge_id": "E3", "offset_m": 20.0, "weight": (1_200_000 +  900_000) / 300.0},
]
```

**Parameters**:

```python
k = 3
ap_range_m = 35.0
ap_link_range_m = ap_range_m   # or set differently if backhaul range differs
candidate_step_m = 1.0         # grid resolution along edges (~ r/3 recommended)
```

---

## Quick Start

### 1) Greedy (Connected Max-Coverage)

```python
from src.placement_greedy import place_access_points, visualize, make_weights,                                   _split_graph_for_dist, _covered_miners

placements = place_access_points(
    edges, miners, k,
    ap_range_m,
    candidate_step_m=candidate_step_m,
    ap_link_range_m=ap_link_range_m,
    allow_connectors=True,
)
print("Chosen AP placements:", placements)

# Metrics
Gs, miner_nodes, ap_nodes = _split_graph_for_dist(edges, miners, placements, step=max(1.0, ap_range_m/5))
covered_idxs = _covered_miners(Gs, miner_nodes, ap_nodes, ap_range_m)
weights = make_weights(miners)
covered_weight = sum(weights[i] for i in covered_idxs)
total_weight   = sum(weights.values())
print(f"Covered miners: {len(covered_idxs)}/{len(miners)} ({len(covered_idxs)/len(miners)*100:.1f}%)")
print(f"Weighted coverage: {covered_weight:.1f}/{total_weight:.1f} ({covered_weight/total_weight*100:.1f}%)")

# Visualization
visualize(edges, miners, placements, ap_range_m, ap_link_range_m=ap_link_range_m, layout="spring")
```

Example toy output:

```
Chosen AP placements: [{'edge_id': 'E1', 'offset_m': 49.0}, {'edge_id': 'E1', 'offset_m': 79.0}, {'edge_id': 'E2', 'offset_m': 11.0}]
Covered miners: 4/4 (100.0%)
Weighted coverage: 39820.0/39820.0 (100.0%)
```

### 2) Learning-Based (Train-Once, Infer-Often)

- **Training** (offline): generate labeled windows using a strong oracle (e.g., connected greedy with lookahead), compute features, and fit a scorer (e.g., `LogisticRegression`, `RandomForest`, `HistGradientBoosting`).
- **Inference** (online): score each candidate site and run the connected selection that blends model score and marginal coverage.

Toy example (with a tiny synthetic fit):

```python
from src.placement_ml import train_toy_scorer, place_access_points_ml,                               _split_graph_for_dist, _covered_miners, visualize, make_weights

clf, scaler, feature_keys = train_toy_scorer()  # demo utility for toy data

placements = place_access_points_ml(
    edges, miners, k,
    ap_range_m,
    ap_link_range_m=ap_link_range_m,
    candidate_step_m=candidate_step_m,
    clf=clf, scaler=scaler, feature_keys=feature_keys,
    alpha=1.0, beta=1.0, allow_connectors=True
)
print("Predicted AP placements:", placements)

# Metrics + viz same as greedy
```

Example output (toy):

```
Predicted AP placements: [{'edge_id': 'E1', 'offset_m': 55.0}, {'edge_id': 'E1', 'offset_m': 85.0}, {'edge_id': 'E2', 'offset_m': 10.0}]
Covered miners: 4/4 (100.0%)
Weighted coverage: 39820.0/39820.0 (100.0%)
AP mesh: 3 nodes, 2 links → connected
```

---

## Visuals

The `visualize(...)` helper renders:

- Initial graph + miners  
- Placements with **triangles**, dashed **coverage circles** (radius `r`), **covered vs. uncovered** miners, and **AP mesh links** (solid)

> Distances are along the graph; circle radii are scaled to the 2D layout for approximate readability.

---

## References

- Maximal covering on networks with edge/continuous demand:  
  Blanquero, Carrizosa, G.-Tóth. *Maximal covering location problems on networks with regional demand.* Optimization Online (2014).  
  https://optimization-online.org/wp-content/uploads/2014/09/4553.pdf

- Minmax regret MCLP with edge demands (discretization on networks):  
  Baldomero-Naranjo, Kalcsics, Rodríguez-Chía. *arXiv:2409.11872* (2024).  
  https://arxiv.org/abs/2409.11872

- ML inspiration for AP placement with GNN-style ideas:  
  Papanikolaou-Ntais et al. *Optimizing access point placement using GNN for enhancing indoors UE localization.* EuCNC/6G Summit (2025).  
  https://ieeexplore.ieee.org/document/11036902
