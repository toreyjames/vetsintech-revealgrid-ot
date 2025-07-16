import networkx as nx, random, io, base64, json, numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# -----------------------------
# 1.  Data models & helpers
# -----------------------------
SiteDict = Dict[str, Dict[str, float]]

SAMPLE_SITES: SiteDict = {
    "Detroit Plant": {"lat": 42.33, "lon": -83.04, "eff": 85, "down": 5, "risk": 20},
    "Chicago Plant": {"lat": 41.88, "lon": -87.63, "eff": 92, "down": 2, "risk": 10},
    "Austin Plant":  {"lat": 30.26, "lon": -97.74, "eff": 78, "down": 10, "risk": 35},
    "Seattle Plant": {"lat": 47.60, "lon": -122.33, "eff": 88, "down": 4, "risk": 15},
}

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# -----------------------------
# 2.  Visualisations
# -----------------------------
def global_site_map(sites: SiteDict = SAMPLE_SITES):
    fig, ax = plt.subplots(figsize=(8,5))
    lats, lons, effs = zip(*[(v["lat"], v["lon"], v["eff"]) for v in sites.values()])
    scatter = ax.scatter(lons, lats, c=effs, cmap="RdYlGn", s=200)
    for name, d in sites.items():
        ax.text(d["lon"]+0.3, d["lat"], f"{name}\nEff {d['eff']}%", fontsize=8)
    ax.set_title("RevealGrid – Global Uptime View")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    plt.colorbar(scatter, label="Efficiency %")
    return fig

# -----------------------------
# 3.  Graph‑based process map
# -----------------------------
def sample_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = ["Sensor1","PLC1","Actuator1","ManualCheck","Bottling","Quality","OptimusBot"]
    edges = [("Sensor1","PLC1"),("PLC1","Actuator1"),("Actuator1","ManualCheck"),
             ("ManualCheck","Bottling"),("Bottling","Quality"),("OptimusBot","ManualCheck")]
    G.add_edges_from(edges)
    for n in nodes:
        G.nodes[n]["latency"] = random.randint(1,10)
        G.nodes[n]["type"] = ("manual" if "Manual" in n else
                              "robot" if "Bot" in n else "automated")
    return G

def plot_process(G: nx.DiGraph):
    pos = nx.spring_layout(G, seed=3)
    colors = {"automated":"cornflowerblue","manual":"tomato","robot":"limegreen"}
    node_colors = [colors[G.nodes[n]["type"]] for n in G.nodes]
    fig, ax = plt.subplots(figsize=(6,4))
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
            edge_color="gray", node_size=900, font_size=9)
    ax.set_title("Process Flow (latency‑coded)")
    return fig

# -----------------------------
# 4.  Hybrid discovery stub
# -----------------------------
def hybrid_discover(G: nx.DiGraph, cmdb_assets: List[str], uploaded_assets: List[str]):
    # CMDB
    for a in cmdb_assets:
        G.add_node(a, type="synced", latency=random.randint(1,8))
        G.add_edge(random.choice(list(G.nodes)), a)
    # Uploaded / CV
    for a in uploaded_assets:
        G.add_node(a, type="offline", latency=random.randint(1,8))
        G.add_edge(a, random.choice(list(G.nodes)))
    return G

# -----------------------------
# 5.  Simple downtime heat‑map
# -----------------------------
def downtime_recommendations(G: nx.DiGraph) -> List[str]:
    lat = np.array([G.nodes[n]["latency"] for n in G.nodes])
    thr = lat.mean() + 2*lat.std()
    recs = [f"High latency on {n} ({G.nodes[n]['latency']} ms) – investigate"
            for n in G.nodes if G.nodes[n]["latency"] > thr]
    return recs 