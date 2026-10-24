"""MORPHE Dashboard - Enterprise AI Memory Management Interface

Memory-Oriented Reconfigurable Plasticity and Hierarchical Evolution.
Professional, high-contrast, local-first memory topology visualization,
routed retrieval tracing, autonomous sandbox adaptation, and audit version control.
"""

import io
import json
import sys
from pathlib import Path
import time
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Setup path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from morphe.core import MorpheSystem
from morphe.adaptation.triggers import detect_adaptation_candidates
from morphe.adaptation.operations import SplitOperation, MergeOperation, ShortcutOperation, AdaptationResult
from morphe.policy.controller import BenchmarkMode


st.set_page_config(
    page_title="MORPHE | Reconfigurable Plastic Memory",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------- DESIGN SYSTEM: REFERENCE UI STYLING -----------------
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #F8FAF8;
        color: #111827;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E5EAE5;
        padding-top: 1rem;
    }
    
    /* Headers & Typography - High Contrast & Visibility */
    h1, h2, h3, h4, h5, h6 {
        color: #111827 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    p, span, div, label {
        color: #1F2937;
    }
    
    /* Navigation Radio Items in Sidebar */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        gap: 6px;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        padding: 9px 14px;
        border-radius: 8px;
        transition: all 0.15s ease;
        cursor: pointer;
        font-weight: 500;
        font-size: 0.90rem;
        color: #374151;
        background-color: transparent;
        display: flex;
        align-items: center;
        border: 1px solid transparent;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: #F0F5F1;
        color: #1B4D3E;
        border-color: #E2E8F0;
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #EAF2EC !important;
        color: #2D5A3D !important;
        font-weight: 700 !important;
        border-color: #D6E3D8 !important;
        box-shadow: 0 1px 2px rgba(45, 90, 61, 0.05);
    }
    div[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] {
        accent-color: #2D5A3D;
    }

    /* Reference UI Hero Banner */
    .morphe-hero-banner {
        background-color: #EAF2EC;
        border: 1px solid #D6E3D8;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 22px;
    }
    .morphe-hero-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        color: #2D5A3D;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .morphe-hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #17231B;
        margin: 0 0 10px 0;
        letter-spacing: -0.03em;
        line-height: 1.2;
    }
    .morphe-hero-desc {
        font-size: 0.95rem;
        color: #374A3D;
        max-width: 780px;
        line-height: 1.5;
        margin-bottom: 16px;
    }
    .morphe-tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .morphe-hero-tag {
        display: inline-flex;
        align-items: center;
        background-color: #FFFFFF;
        color: #2D5A3D;
        border: 1px solid #CFDFD2;
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Metric Cards in Row */
    .morphe-metric-card {
        background: #FFFFFF;
        border: 1px solid #E5EAE5;
        border-radius: 14px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        margin-bottom: 18px;
        min-height: 104px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .metric-card-label {
        font-size: 0.80rem;
        font-weight: 600;
        color: #55695C;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card-icon {
        width: 28px;
        height: 28px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 11px;
    }
    .icon-green { background: #DCFCE7; color: #166534; }
    .icon-purple { background: #F3E8FF; color: #6B21A8; }
    .icon-amber { background: #FEF3C7; color: #92400E; }
    .icon-blue { background: #E0F2FE; color: #075985; }

    .metric-card-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.1;
    }
    .metric-card-sub {
        font-size: 0.76rem;
        color: #6B7280;
        margin-top: 4px;
    }

    /* General Step / Panel Cards */
    .morphe-panel {
        background: #FFFFFF;
        border: 1px solid #E5EAE5;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }
    .morphe-panel-step {
        font-size: 0.72rem;
        font-weight: 700;
        color: #2D5A3D;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .morphe-panel-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #111827;
        margin: 0 0 6px 0;
    }
    .morphe-panel-desc {
        font-size: 0.85rem;
        color: #4B5563;
        margin-bottom: 16px;
    }

    /* Badges */
    .morphe-badge {
        display: inline-block;
        padding: 3px 10px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-right: 6px;
    }
    .morphe-badge-green { background-color: #EAF2EC; color: #2D5A3D; border: 1px solid #D6E3D8; }
    .morphe-badge-blue { background-color: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
    .morphe-badge-purple { background-color: #FAF5FF; color: #7E22CE; border: 1px solid #E9D5FF; }
    .morphe-badge-amber { background-color: #FFFBEB; color: #B45309; border: 1px solid #FDE68A; }
    .morphe-badge-red { background-color: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }
    .morphe-badge-slate { background-color: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; }

    /* Buttons Styling */
    button[kind="primary"], .stButton > button[kind="primary"] {
        background-color: #2D5A3D !important;
        border-color: #2D5A3D !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        box-shadow: 0 1px 2px rgba(45, 90, 61, 0.12) !important;
    }
    button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        background-color: #224830 !important;
        border-color: #224830 !important;
    }
    .stButton > button {
        border-radius: 8px !important;
        border: 1px solid #CBD5E1 !important;
        font-weight: 600 !important;
        color: #1F2937 !important;
        background-color: #FFFFFF !important;
    }
    .stButton > button:hover {
        border-color: #2D5A3D !important;
        color: #2D5A3D !important;
        background-color: #F7FAF8 !important;
    }

    /* Inputs */
    input, textarea, select {
        border-radius: 8px !important;
        border: 1px solid #CBD5E1 !important;
        color: #111827 !important;
        background-color: #FFFFFF !important;
    }
    input:focus, textarea:focus {
        border-color: #2D5A3D !important;
        box-shadow: 0 0 0 1px #2D5A3D !important;
    }

    /* Metrics Override */
    div[data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        color: #111827 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        color: #4B5563 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Sidebar Footer Card */
    .sidebar-footer-card {
        background-color: #EAF2EC;
        border: 1px solid #D6E3D8;
        border-radius: 12px;
        padding: 14px;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- SYSTEM INITIALIZATION -----------------
@st.cache_resource
def get_morphe_system() -> MorpheSystem:
    system = MorpheSystem()
    if not system.is_initialized:
        system.initialize_from_benchmark()
    return system


system = get_morphe_system()

# ----------------- COLOR PALETTE FOR DISTINCT MODULES -----------------
MOD_COLORS = [
    "#2D5A3D",  # Forest Green (Brand)
    "#2563EB",  # Royal Blue
    "#7C3AED",  # Violet
    "#D97706",  # Amber
    "#0891B2",  # Cyan
    "#DB2777",  # Rose
    "#4F46E5",  # Indigo
    "#0D9488",  # Persian Teal
    "#EA580C",  # Sunset
    "#DC2626",  # Crimson
    "#65A30D",  # Olive Lime
    "#475569",  # Slate
]


def get_color_for_module(module_id: str, is_active: bool = False) -> str:
    if is_active:
        return "#F59E0B"  # Gold highlight for active traversal
    try:
        num = int("".join([c for c in module_id if c.isdigit()]))
        return MOD_COLORS[num % len(MOD_COLORS)]
    except Exception:
        h = sum(ord(c) for c in module_id)
        return MOD_COLORS[h % len(MOD_COLORS)]


# ----------------- GRAPH BUILDER HELPER -----------------
def generate_plotly_network(graph, layout_type="spring", highlighted_nodes=None):
    highlighted_nodes = highlighted_nodes or set()
    G = graph.graph
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
        return fig

    if layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=1.4)

    edge_x_struct, edge_y_struct, edge_hover_struct = [], [], []
    edge_x_short, edge_y_short, edge_hover_short = [], [], []

    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        e_type = data.get("edge_type", "structural")
        weight = data.get("weight", 1.0)
        label = f"{u} <-> {v} ({e_type}, w={weight})"

        if e_type == "shortcut":
            edge_x_short.extend([x0, x1, None])
            edge_y_short.extend([y0, y1, None])
            edge_hover_short.append(label)
        else:
            edge_x_struct.extend([x0, x1, None])
            edge_y_struct.extend([y0, y1, None])
            edge_hover_struct.append(label)

    traces = []
    if edge_x_struct:
        traces.append(
            go.Scatter(
                x=edge_x_struct,
                y=edge_y_struct,
                line=dict(width=1.8, color="#CBD5E1"),
                hoverinfo="none",
                mode="lines",
                name="Structural Link",
            )
        )
    if edge_x_short:
        traces.append(
            go.Scatter(
                x=edge_x_short,
                y=edge_y_short,
                line=dict(width=3.4, color="#DC2626", dash="dash"),
                hoverinfo="none",
                mode="lines",
                name="Learned Shortcut",
            )
        )

    node_x, node_y, node_sizes, node_colors, node_hover = [], [], [], [], []
    for node_id in G.nodes():
        if node_id not in pos:
            continue
        x, y = pos[node_id]
        mod = graph.get_module(node_id)
        size = len(mod.local_index) if mod else 0
        node_x.append(x)
        node_y.append(y)
        node_sizes.append(max(26, size * 5.6))

        is_act = node_id in highlighted_nodes
        node_colors.append(get_color_for_module(node_id, is_active=is_act))

        var = mod.local_index.compute_variance() if mod else 0.0
        degree = G.degree(node_id)
        node_hover.append(
            f"<b>Module:</b> {node_id}<br>"
            f"<b>Total Records:</b> {size}<br>"
            f"<b>Internal Variance:</b> {var:.3f}<br>"
            f"<b>Graph Degree:</b> {degree}<br>"
            f"<b>Status:</b> {'Active Route Traversal' if is_act else 'Committed'}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[f"{n} ({len(graph.get_module(n).local_index) if graph.get_module(n) else 0})" for n in G.nodes()],
        textposition="top center",
        hovertext=node_hover,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2.5, color="#FFFFFF"),
            opacity=0.96,
        ),
        name="Memory Modules",
    )
    traces.append(node_trace)

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=dict(
                text="Topological Memory Graph (Nodes = Semantic Clusters, Red Dashed = Synthesized Shortcuts)",
                font=dict(size=13, color="#111827", family="sans-serif"),
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="closest",
            margin=dict(b=15, l=15, r=15, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            height=460,
        ),
    )
    return fig


# ----------------- SIDEBAR: NAVIGATION & CONTROLS -----------------
with st.sidebar:
    st.markdown("""
    <div style="padding-bottom: 12px; margin-bottom: 14px; border-bottom: 1px solid #E5EAE5;">
        <div style="display: flex; align-items: center; gap: 9px;">
            <div style="width: 26px; height: 26px; border-radius: 7px; background-color: #2D5A3D; display: flex; align-items: center; justify-content: center; color: #FFFFFF; font-weight: 800; font-size: 14px;">
                M
            </div>
            <div>
                <div style="font-size: 1.22rem; font-weight: 800; color: #17231B; letter-spacing: -0.02em; line-height: 1.1;">
                    MORPHE
                </div>
                <div style="font-size: 0.70rem; color: #5C7063; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
                    Local AI Organizer
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    active_version = system.version_manager.active_version_id or "v0_initial"
    st.markdown(f"""
    <div style="background: #F8FAF8; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 12px; margin-bottom: 16px;">
        <div style="font-size: 0.68rem; color: #5C7063; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Committed State</div>
        <div style="font-size: 0.88rem; font-weight: 700; color: #2D5A3D; word-break: break-all;">{active_version}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 0.70rem; font-weight: 700; color: #6B7280; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
        Workspace
    </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "Workspace Navigation",
        [
            "Overview & Topology",
            "Semantic Retrieval",
            "Knowledge Ingestion",
            "Adaptation Cockpit",
            "Version History & Restore",
            "Architecture Comparison",
            "Multi-Policy Benchmark",
            "System Walkthrough",
        ],
        label_visibility="collapsed",
        key="main_navigation_selection",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.70rem; font-weight: 700; color: #6B7280; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">
        Autonomous Controls
    </div>
    """, unsafe_allow_html=True)

    quick_mode = st.selectbox(
        "Adaptation Policy",
        ["rl_guided", "heuristic_only"],
        key="sb_quick_policy_select",
    )

    if st.button("Trigger Self-Adaptation", key="sb_btn_quick_adapt", use_container_width=True):
        with st.spinner("Evaluating candidates in sandbox trial copy..."):
            step_log = system.adapt_step(mode=quick_mode)
            if step_log:
                if step_log.decision == "ACCEPTED":
                    st.success(f"{step_log.action_type.upper()}: ACCEPTED")
                else:
                    st.warning(f"{step_log.action_type.upper()}: REJECTED")
                st.caption(step_log.reason)
            else:
                st.info("Memory structure is in equilibrium.")
        st.rerun()

    if st.button("Reset Memory to Benchmark", key="sb_btn_reset_all", use_container_width=True):
        system.graph = type(system.graph)(dimension=system.embedder.dimension)
        system.initialize_from_benchmark()
        st.success("Memory reset completed.")
        st.rerun()

    st.markdown(f"""
    <div class="sidebar-footer-card">
        <div style="font-weight: 700; color: #17231B; font-size: 0.85rem; margin-bottom: 4px;">
            Deterministic Safety Gate
        </div>
        <div style="color: #3C4E42; font-size: 0.76rem; line-height: 1.4; margin-bottom: 8px;">
            AI inference runs locally in isolated sandbox copies before committing changes. Zero-risk rollback guaranteed.
        </div>
        <div style="display: flex; align-items: center; gap: 6px; font-size: 0.74rem; font-weight: 700; color: #2D5A3D;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #2D5A3D;"></span>
            LOCAL-FIRST MODE ACTIVE
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: #94A3B8; padding: 10px 4px 0 4px;">
        <span>MORPHE Architecture</span>
        <span>v1.0</span>
    </div>
    """, unsafe_allow_html=True)


# ----------------- MAIN TOP BAR (BREADCRUMB & STATUS) -----------------
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; padding-bottom: 12px; margin-bottom: 12px; border-bottom: 1px solid #E5EAE5;">
    <div style="font-size: 0.86rem; color: #6B7280;">
        Workspace &nbsp;/&nbsp; <span style="color: #111827; font-weight: 700;">{nav_selection}</span>
    </div>
    <div style="display: inline-flex; align-items: center; gap: 7px; background-color: #EAF2EC; border: 1px solid #D6E3D8; padding: 4px 14px; border-radius: 9999px; font-size: 0.78rem; font-weight: 700; color: #2D5A3D;">
        <span style="display: inline-block; width: 7px; height: 7px; border-radius: 50%; background-color: #2D5A3D;"></span>
        Local AI Ready
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------- REFERENCE UI HERO BANNER -----------------
st.markdown("""
<div class="morphe-hero-banner">
    <div class="morphe-hero-eyebrow">Hierarchical Vector Memory System</div>
    <h1 class="morphe-hero-title">Autonomous Plastic Memory Architecture</h1>
    <div class="morphe-hero-desc">
        Self-optimizing topological vector memory that reconfigures dynamically through learned structural adaptations
        while deterministic validation gates prevent regressions.
    </div>
    <div class="morphe-tag-row">
        <span class="morphe-hero-tag">Private by Design</span>
        <span class="morphe-hero-tag">Runs on Your Machine</span>
        <span class="morphe-hero-tag">Deterministic Sandbox Validation</span>
        <span class="morphe-hero-tag">Zero-Risk Rollbacks</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------- TOP METRIC CARDS ROW -----------------
eval_metrics = system.get_evaluation_metrics(use_heldout=False)
total_modules = len(system.graph.modules)
total_shortcuts = len(system.graph.get_shortcuts())

col_c1, col_c2, col_c3, col_c4 = st.columns(4)

with col_c1:
    st.markdown(f"""
    <div class="morphe-metric-card">
        <div class="metric-card-header">
            <span class="metric-card-label">Active Modules</span>
            <div class="metric-card-icon icon-green">M</div>
        </div>
        <div class="metric-card-value">{total_modules}</div>
        <div class="metric-card-sub">In memory topology</div>
    </div>
    """, unsafe_allow_html=True)

with col_c2:
    st.markdown(f"""
    <div class="morphe-metric-card">
        <div class="metric-card-header">
            <span class="metric-card-label">Recall@3</span>
            <div class="metric-card-icon icon-purple">R</div>
        </div>
        <div class="metric-card-value">{eval_metrics.recall_at_3:.4f}</div>
        <div class="metric-card-sub">Tuning set accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col_c3:
    st.markdown(f"""
    <div class="morphe-metric-card">
        <div class="metric-card-header">
            <span class="metric-card-label">Shortcuts</span>
            <div class="metric-card-icon icon-amber">S</div>
        </div>
        <div class="metric-card-value">{total_shortcuts}</div>
        <div class="metric-card-sub">Learned graph bridges</div>
    </div>
    """, unsafe_allow_html=True)

with col_c4:
    st.markdown(f"""
    <div class="morphe-metric-card">
        <div class="metric-card-header">
            <span class="metric-card-label">Avg Latency</span>
            <div class="metric-card-icon icon-blue">T</div>
        </div>
        <div class="metric-card-value">{eval_metrics.avg_latency_ms:.2f} ms</div>
        <div class="metric-card-sub">Centroid routing speed</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# VIEW 1: OVERVIEW & TOPOLOGY
# =========================================================================
if nav_selection == "Overview & Topology":
    st.markdown("""
    <div class="morphe-panel">
        <div class="morphe-panel-step">Module 01</div>
        <h3 class="morphe-panel-title">Interactive Memory Graph Topology</h3>
        <div class="morphe-panel-desc">
            Visual representation of autonomous modules, structural links, and synthesized shortcuts across the vector space.
        </div>
    """, unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        layout_choice = st.radio(
            "Graph Layout Algorithm",
            ["spring", "kamada_kawai", "circular"],
            horizontal=True,
            key="graph_layout_choice_radio",
        )
    with col_ctrl2:
        mod_inspect_select = st.selectbox(
            "Inspect Specific Module Contents",
            ["None"] + [m.module_id for m in system.graph.get_all_modules()],
            key="graph_inspect_module_select",
        )

    highlight_set = {mod_inspect_select} if mod_inspect_select != "None" else set()
    col_chart, col_side = st.columns([3, 1])

    with col_chart:
        fig_net = generate_plotly_network(system.graph, layout_type=layout_choice, highlighted_nodes=highlight_set)
        st.plotly_chart(fig_net, use_container_width=True, key="plotly_network_view")

    with col_side:
        st.markdown("#### Module Metrics")
        mod_rows = []
        for m in system.graph.get_all_modules():
            mod_rows.append({
                "Module": m.module_id,
                "Records": len(m.local_index),
                "Variance": round(m.local_index.compute_variance(), 3),
            })
        st.dataframe(pd.DataFrame(mod_rows), use_container_width=True, hide_index=True)

        shortcuts = system.graph.get_shortcuts()
        st.markdown(f"#### Active Shortcuts ({len(shortcuts)})")
        if shortcuts:
            for u, v, d in shortcuts:
                st.markdown(f"`{u}` <-> `{v}` <span class='morphe-badge morphe-badge-red'>w={d.get('weight', 1.0)}</span>", unsafe_allow_html=True)
        else:
            st.caption("No shortcuts synthesized yet. Frequently co-accessed multi-thematic queries trigger shortcut creation.")

    if mod_inspect_select != "None":
        st.markdown("---")
        st.markdown(f"#### Deep Inspection: Module `{mod_inspect_select}`")
        target_mod = system.graph.get_module(mod_inspect_select)
        if target_mod:
            c1, c2, c3 = st.columns(3)
            c1.metric("Record Count", len(target_mod.local_index))
            c2.metric("Cluster Variance", f"{target_mod.local_index.compute_variance():.4f}")
            c3.metric("Capacity Limit", target_mod.capacity)

            rec_ids, vectors = target_mod.local_index.get_all_vectors()
            all_records = {r.id: r for r in system.storage.get_all_records()}

            inspect_rows = []
            for rid in rec_ids:
                rec_item = all_records.get(rid)
                inspect_rows.append({
                    "Record ID": rid,
                    "Title": rec_item.metadata.get("title", "") if rec_item else "",
                    "Category": rec_item.metadata.get("category", "") if rec_item else "",
                    "Snippet": rec_item.text[:100] + "..." if rec_item and len(rec_item.text) > 100 else (rec_item.text if rec_item else ""),
                })
            st.dataframe(pd.DataFrame(inspect_rows), use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# VIEW 2: SEMANTIC RETRIEVAL
# =========================================================================
elif nav_selection == "Semantic Retrieval":
    st.markdown("""
    <div class="morphe-panel">
        <div class="morphe-panel-step">Module 02</div>
        <h3 class="morphe-panel-title">Routed Semantic Retrieval</h3>
        <div class="morphe-panel-desc">
            Queries strictly traverse top candidate centroids and 1-hop shortcuts. Global brute-force search is prohibited.
        </div>
    """, unsafe_allow_html=True)

    preset_queries = [
        "How does Raft elect a leader and maintain log replication in distributed state machines?",
        "Explain event-driven spike processing on neuromorphic silicon chips for neural computation.",
        "How does quantum teleportation transfer quantum state using entangled Bell pairs?",
        "How do Paxos prepare and accept phases prevent conflicting updates?",
        "What is hippocampal memory indexing and consolidation into the cortex?",
    ]
    query_mode = st.selectbox(
        "Select Example Query or Choose Custom:",
        ["Custom"] + preset_queries,
        key="retrieval_query_mode_select",
    )
    if query_mode == "Custom":
        search_query = st.text_input(
            "Query Prompt",
            value="How does synaptic long-term potentiation strengthen neuronal connections in Hebbian learning?",
            key="retrieval_custom_query_input",
        )
    else:
        search_query = st.text_input("Query Prompt", value=query_mode, key="retrieval_preset_query_input")

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        top_k_records = st.slider("Top Documents to Retrieve", 1, 10, 5, key="retrieval_top_k_slider")
    with col_q2:
        st.write("")
        st.write("")
        execute_button = st.button("Execute Routed Query", type="primary", use_container_width=True, key="btn_run_retrieval")

    if execute_button or search_query:
        with st.spinner("Routing query through centroids and shortcut hops..."):
            res = system.query(search_query, top_k=top_k_records)

        st.markdown("#### Execution Telemetry")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Execution Latency", f"{res.latency_ms:.2f} ms")
        m2.metric("Queried Modules", str(len(res.queried_modules)), ", ".join(res.queried_modules))
        m3.metric("Shortcuts Traversed", str(len(res.shortcut_hops)))
        m4.metric("Top Relevance Score", f"{res.records[0]['score']:.4f}" if res.records else "0.0")

        st.markdown("#### Routing Path on Memory Topology")
        st.plotly_chart(
            generate_plotly_network(system.graph, layout_type="spring", highlighted_nodes=set(res.queried_modules)),
            use_container_width=True,
            key="plotly_traversed_retrieval_network",
        )

        col_cent, col_records = st.columns([1, 2])
        with col_cent:
            st.markdown("#### Centroid Cosine Proximity")
            cent_data = []
            for m_id, score in res.centroid_ranked_modules:
                was_queried = m_id in res.queried_modules
                cent_data.append({
                    "Module": m_id,
                    "Cosine Similarity": round(score, 4),
                    "Status": "Traversed" if was_queried else "Filtered Out",
                })
            df_cent = pd.DataFrame(cent_data)
            fig_cent = px.bar(
                df_cent,
                x="Cosine Similarity",
                y="Module",
                orientation="h",
                color="Status",
                color_discrete_map={"Traversed": "#2D5A3D", "Filtered Out": "#CBD5E1"},
                height=260,
            )
            fig_cent.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
            )
            st.plotly_chart(fig_cent, use_container_width=True, key="plotly_centroid_bars")

        with col_records:
            st.markdown("#### Retrieved Ranked Records")
            if res.records:
                all_recs = {r.id: r for r in system.storage.get_all_records()}
                for rank, hit in enumerate(res.records, start=1):
                    rec = all_recs.get(hit["id"])
                    title = rec.metadata.get("title", hit["id"]) if rec else hit["id"]
                    text = rec.text if rec else "Text not found"
                    mod_id = hit["module_id"]
                    score = hit["score"]

                    with st.expander(f"Rank {rank}: [{mod_id}] {title} - Score: {score:.4f}", expanded=(rank == 1)):
                        st.markdown(f"<span class='morphe-badge morphe-badge-green'>{mod_id}</span> <span class='morphe-badge morphe-badge-blue'>Score: {score:.4f}</span>", unsafe_allow_html=True)
                        st.write(text)
            else:
                st.warning("No records matched within the traversed candidate modules.")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# VIEW 3: KNOWLEDGE INGESTION
# =========================================================================
elif nav_selection == "Knowledge Ingestion":
    st.markdown("""
    <div class="morphe-panel">
        <div class="morphe-panel-step">Module 03</div>
        <h3 class="morphe-panel-title">Knowledge Ingestion & Upload</h3>
        <div class="morphe-panel-desc">
            Upload files or compose custom documents to dynamically integrate new knowledge into the hierarchical memory graph.
        </div>
    """, unsafe_allow_html=True)

    col_up_mode, col_up_stats = st.columns([2, 1])
    with col_up_mode:
        upload_type = st.radio(
            "Ingestion Mode",
            ["Multi-File Upload (.txt, .json, .csv, .md)", "Direct Text Composer"],
            horizontal=True,
            key="ingestion_type_radio",
        )

    with col_up_stats:
        total_recs = len(system.storage.get_all_records())
        st.metric("Total Knowledge Records", total_recs, f"{len(system.graph.modules)} Modules Active")

    if upload_type == "Multi-File Upload (.txt, .json, .csv, .md)":
        files = st.file_uploader(
            "Select text, JSON, or CSV files to ingest",
            type=["txt", "json", "csv", "md"],
            accept_multiple_files=True,
            key="file_uploader_ingest",
        )
        custom_cat = st.text_input("Category Tag", value="custom_knowledge", key="upload_file_cat_tag")

        if files and st.button("Ingest Uploaded Files", type="primary", key="btn_execute_file_upload"):
            parsed_docs = []
            for f in files:
                fname = f.name
                content = f.read().decode("utf-8", errors="replace")
                if fname.endswith(".json"):
                    try:
                        data = json.loads(content)
                        if isinstance(data, list):
                            for idx, item in enumerate(data):
                                if isinstance(item, dict):
                                    parsed_docs.append({
                                        "id": item.get("id", f"{fname}_{idx}"),
                                        "title": item.get("title", f"{fname} #{idx}"),
                                        "text": item.get("text", str(item)),
                                        "category": item.get("category", custom_cat),
                                    })
                        elif isinstance(data, dict):
                            parsed_docs.append({
                                "id": data.get("id", fname),
                                "title": data.get("title", fname),
                                "text": data.get("text", str(data)),
                                "category": data.get("category", custom_cat),
                            })
                    except Exception as e:
                        st.error(f"Error parsing JSON file {fname}: {e}")
                elif fname.endswith(".csv"):
                    try:
                        df = pd.read_csv(io.StringIO(content))
                        text_col = df.columns[0]
                        for c in df.columns:
                            if "text" in c.lower() or "content" in c.lower():
                                text_col = c
                                break
                        for idx, row in df.iterrows():
                            parsed_docs.append({
                                "id": f"{fname}_{idx}",
                                "title": f"{fname} Row {idx}",
                                "text": str(row[text_col]),
                                "category": custom_cat,
                            })
                    except Exception as e:
                        st.error(f"Error parsing CSV file {fname}: {e}")
                else:
                    parsed_docs.append({
                        "id": Path(fname).stem,
                        "title": Path(fname).stem,
                        "text": content,
                        "category": custom_cat,
                    })

            if parsed_docs:
                with st.spinner(f"Chunking, embedding, and integrating {len(parsed_docs)} documents..."):
                    added_chunks = system.ingest_custom_documents(parsed_docs)
                st.success(f"Successfully integrated {len(parsed_docs)} documents ({len(added_chunks)} chunks) into memory graph.")
                st.rerun()
            else:
                st.warning("No readable text content identified in uploaded files.")

    else:
        doc_title = st.text_input("Document Title", value="Reinforcement Learning for Memory Networks", key="composer_doc_title")
        doc_cat = st.text_input("Category Tag", value="machine_learning", key="composer_doc_category")
        doc_text = st.text_area(
            "Document Text Content",
            height=160,
            value="Reinforcement learning can dynamically reconfigure neural memory graphs by treating structural changes as discrete actions. Multi-armed bandits learn which modules to split or shortcut to maximize recall while penalizing query latency.",
            key="composer_doc_text_area",
        )

        if st.button("Ingest Composed Text", type="primary", key="btn_execute_text_composer"):
            if doc_text.strip():
                with st.spinner("Generating embeddings and routing to closest module..."):
                    payload = [{
                        "id": f"doc_{int(time.time())}",
                        "title": doc_title,
                        "category": doc_cat,
                        "text": doc_text.strip(),
                    }]
                    added_chunks = system.ingest_custom_documents(payload)
                st.success(f"Successfully ingested '{doc_title}' ({len(added_chunks)} chunks).")
                st.rerun()
            else:
                st.warning("Please enter text content before ingesting.")

    st.markdown("---")
    st.markdown("#### Live Memory Database Records")
    all_recs = system.storage.get_all_records()
    if all_recs:
        table_recs = []
        for r in all_recs[-12:]:
            table_recs.append({
                "Record ID": r.id,
                "Module": r.module_id,
                "Title": r.metadata.get("title", ""),
                "Category": r.metadata.get("category", ""),
                "Snippet": r.text[:80] + "..." if len(r.text) > 80 else r.text,
            })
        st.dataframe(pd.DataFrame(table_recs), use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# VIEW 4: ADAPTATION COCKPIT
# =========================================================================
elif nav_selection == "Adaptation Cockpit":
    st.markdown("""
    <div class="morphe-panel">
        <div class="morphe-panel-step">Module 04</div>
        <h3 class="morphe-panel-title">Autonomous Adaptation Cockpit & Sandbox Simulator</h3>
        <div class="morphe-panel-desc">
            Inspect pending structural mutations, simulate operations on isolated trial graphs, and verify deterministic acceptance gates.
        </div>
    """, unsafe_allow_html=True)

    candidates = detect_adaptation_candidates(
        graph=system.graph,
        thresholds=system.config.thresholds,
        storage=system.storage,
    )

    col_cand, col_sim = st.columns([3, 2])
    with col_cand:
        st.markdown(f"#### Active Adaptation Candidates ({len(candidates)})")
        if candidates:
            c_rows = []
            for c in candidates:
                c_rows.append({
                    "Candidate ID": c.candidate_id,
                    "Action": c.operation_type.upper(),
                    "Targets": ", ".join(c.target_modules),
                    "Priority": round(c.priority_score, 2),
                    "Trigger Rationale": c.reason,
                })
            st.dataframe(pd.DataFrame(c_rows), use_container_width=True, hide_index=True)
        else:
            st.info("All modules are within variance and size capacity bounds.")

    with col_sim:
        st.markdown("#### Interactive Sandbox Simulator")
        st.caption("Simulate an operation on a cloned trial graph copy.")
        sim_action = st.selectbox("Action to Simulate", ["SPLIT", "MERGE", "SHORTCUT"], key="sim_action_select")

        available_mods = list(system.graph.modules.keys())
        selected_op = None

        if sim_action == "SPLIT" and available_mods:
            target_split = st.selectbox("Module to Split", available_mods, key="sim_split_target_select")
            selected_op = SplitOperation(module_id=target_split)

        elif sim_action == "MERGE" and len(available_mods) >= 2:
            c_m1, c_m2 = st.columns(2)
            mod_a = c_m1.selectbox("Module A", available_mods, index=0, key="sim_merge_a_select")
            mod_b = c_m2.selectbox("Module B", available_mods, index=1, key="sim_merge_b_select")
            selected_op = MergeOperation(module_a=mod_a, module_b=mod_b)

        elif sim_action == "SHORTCUT" and len(available_mods) >= 2:
            c_s1, c_s2 = st.columns(2)
            mod_a = c_s1.selectbox("Source Module", available_mods, index=0, key="sim_short_a_select")
            mod_b = c_s2.selectbox("Target Module", available_mods, index=len(available_mods)-1, key="sim_short_b_select")
            selected_op = ShortcutOperation(module_a=mod_a, module_b=mod_b, weight=0.9)

        if selected_op and st.button("Run Sandbox Trial", type="primary", use_container_width=True, key="btn_run_sandbox_sim"):
            with st.spinner("Cloning memory sandbox and evaluating tuning suite..."):
                val_res = system.validator.validate_adaptation(
                    prod_graph=system.graph,
                    operation=selected_op,
                    tuning_queries=system.benchmark_dataset.tuning_queries,
                )

            if val_res.accepted:
                st.success(f"Trial ACCEPTED: {val_res.reason}")
                if st.button("Commit This Adaptation to Production", key="btn_commit_sandbox_trial"):
                    system.graph.modules = val_res.trial_graph.modules
                    system.graph.graph = val_res.trial_graph.graph
                    system.version_manager.commit_version(
                        graph=system.graph,
                        operation_name=val_res.operation_type,
                        metrics_dict=val_res.post_metrics.to_dict(),
                        reassignments=val_res.record_reassignments,
                        target_modules=val_res.target_modules,
                        reason=val_res.reason,
                    )
                    st.success("Committed to production!")
                    st.rerun()
            else:
                st.error(f"Trial REJECTED: {val_res.reason}")
                st.caption("Production memory remained intact. Trial copy was discarded.")

    st.markdown("---")
    st.markdown("#### Audit & Rollback Event Log")
    audits = system.version_manager.get_audit_trail(limit=10)
    if audits:
        st.dataframe(pd.DataFrame(audits)[["timestamp", "action_type", "target_modules", "decision", "reason", "delta_recall", "delta_latency"]], use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# VIEW 5: VERSION HISTORY & RESTORE
# =========================================================================
elif nav_selection == "Version History & Restore":
    st.markdown("""
    <div class="morphe-panel">
        <div class="morphe-panel-step">Module 05</div>
        <h3 class="morphe-panel-title">Point-in-Time Architecture Revisions</h3>
        <div class="morphe-panel-desc">
            Inspect past committed topology snapshots, monitor retrieval progression, and execute single-click rollback.
        </div>
    """, unsafe_allow_html=True)

    snapshots = system.version_manager.get_history()
    if snapshots:
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            snap_rows = []
            for s in snapshots:
                snap_rows.append({
                    "Version ID": s["version_id"],
                    "Operation": s["operation"].upper(),
                    "Recall@3": s["metrics"].get("recall_at_3", 0.0),
                    "MRR": s["metrics"].get("mrr", 0.0),
                    "Latency (ms)": s["metrics"].get("avg_latency_ms", 0.0),
                    "Modules": s["metrics"].get("module_count", 0),
                })
            st.dataframe(pd.DataFrame(snap_rows), use_container_width=True, hide_index=True)

        with col_t2:
            st.markdown("#### Restore State")
            version_names = [s["version_id"] for s in snapshots]
            target_restore = st.selectbox(
                "Choose Version Snapshot",
                version_names,
                index=len(version_names)-1,
                key="history_restore_select",
            )
            if st.button("Restore Selected Version", type="primary", use_container_width=True, key="btn_execute_restore"):
                with st.spinner(f"Reverting memory graph to snapshot {target_restore}..."):
                    system.restore_version(target_restore)
                st.success(f"Restored to '{target_restore}'!")
                st.rerun()

        st.markdown("#### Metric Progression Across Snapshot History")
        vers = [s["version_id"] for s in snapshots]
        r3_vals = [s["metrics"].get("recall_at_3", 0.0) for s in snapshots]
        lat_vals = [s["metrics"].get("avg_latency_ms", 0.0) for s in snapshots]

        fig_prog = make_subplots(rows=1, cols=2, subplot_titles=("Recall@3 Progression", "Average Latency (ms)"))
        fig_prog.add_trace(go.Scatter(x=vers, y=r3_vals, mode="lines+markers", name="Recall@3", line=dict(color="#2D5A3D", width=2.5)), row=1, col=1)
        fig_prog.add_trace(go.Scatter(x=vers, y=lat_vals, mode="lines+markers", name="Latency (ms)", line=dict(color="#2563EB", width=2.5)), row=1, col=2)
        fig_prog.update_layout(
            height=320,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            margin=dict(l=15, r=15, t=35, b=15),
        )
        st.plotly_chart(fig_prog, use_container_width=True, key="plotly_history_progression")
    else:
        st.info("No snapshots recorded yet.")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# VIEW 6: ARCHITECTURE COMPARISON
# =========================================================================
elif nav_selection == "Architecture Comparison":
    st.markdown("""
    <div class="morphe-panel">
        <div class="morphe-panel-step">Module 06</div>
        <h3 class="morphe-panel-title">Before and After Structural Audit</h3>
        <div class="morphe-panel-desc">
            Quantitative audit of structural changes comparing the original memory graph against the current production state.
        </div>
    """, unsafe_allow_html=True)

    snapshots = system.version_manager.get_history()
    if len(snapshots) >= 2:
        init_s = snapshots[0]
        curr_s = snapshots[-1]

        m_init = init_s["metrics"]
        m_curr = curr_s["metrics"]

        d_r3 = m_curr.get("recall_at_3", 0) - m_init.get("recall_at_3", 0)
        d_mrr = m_curr.get("mrr", 0) - m_init.get("mrr", 0)
        d_lat = m_curr.get("avg_latency_ms", 0) - m_init.get("avg_latency_ms", 0)
        d_mods = m_curr.get("module_count", 0) - m_init.get("module_count", 0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recall@3", f"{m_curr.get('recall_at_3', 0):.4f}", f"{d_r3:+.4f}")
        c2.metric("MRR", f"{m_curr.get('mrr', 0):.4f}", f"{d_mrr:+.4f}")
        c3.metric("Avg Latency", f"{m_curr.get('avg_latency_ms', 0):.2f} ms", f"{d_lat:+.2f} ms")
        c4.metric("Modules", f"{m_curr.get('module_count', 0)}", f"{d_mods:+d}")

        st.markdown("#### Side-by-Side Metric Details")
        metrics_comp = [
            {"Metric": "Recall@1", "Original State": m_init.get("recall_at_1", 0), "Current State": m_curr.get("recall_at_1", 0)},
            {"Metric": "Recall@3", "Original State": m_init.get("recall_at_3", 0), "Current State": m_curr.get("recall_at_3", 0)},
            {"Metric": "Recall@5", "Original State": m_init.get("recall_at_5", 0), "Current State": m_curr.get("recall_at_5", 0)},
            {"Metric": "MRR", "Original State": m_init.get("mrr", 0), "Current State": m_curr.get("mrr", 0)},
            {"Metric": "Latency (ms)", "Original State": m_init.get("avg_latency_ms", 0), "Current State": m_curr.get("avg_latency_ms", 0)},
            {"Metric": "Shortcuts", "Original State": m_init.get("shortcut_count", 0), "Current State": m_curr.get("shortcut_count", 0)},
            {"Metric": "Modules", "Original State": m_init.get("module_count", 0), "Current State": m_curr.get("module_count", 0)},
        ]
        st.dataframe(pd.DataFrame(metrics_comp), use_container_width=True, hide_index=True)
    else:
        st.info("Execute at least one adaptation cycle or ingest data to generate a comparison.")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# VIEW 7: MULTI-POLICY BENCHMARK
# =========================================================================
elif nav_selection == "Multi-Policy Benchmark":
    st.markdown("""
    <div class="morphe-panel">
        <div class="morphe-panel-step">Module 07</div>
        <h3 class="morphe-panel-title">Multi-Policy Benchmark: Static vs Heuristic vs RL-Guided</h3>
        <div class="morphe-panel-desc">
            Compares static memory against greedy heuristics and LinUCB contextual bandit policy on the identical held-out test suite.
        </div>
    """, unsafe_allow_html=True)

    cycles_slider = st.slider("Adaptation Cycles per Mode", 2, 8, 4, key="arena_cycles_slider")

    if st.button("Run Benchmark Arena", type="primary", key="btn_run_arena_benchmark"):
        with st.spinner("Executing three-way evaluation across independent replicated memory graphs..."):
            arena_results = system.run_three_way_comparison(num_cycles=cycles_slider)

        st.success("Three-way benchmark completed successfully.")

        arena_rows = []
        for mode_key, exp in arena_results.items():
            h = exp.final_metrics_heldout
            arena_rows.append({
                "Operational Mode": mode_key.upper(),
                "Held-Out Recall@3": h.recall_at_3,
                "Held-Out MRR": h.mrr,
                "Latency (ms)": h.avg_latency_ms,
                "Accepted": exp.accepted_count,
                "Rejected": exp.rejected_count,
                "Final Modules": h.module_count,
                "Shortcuts": h.shortcut_count,
            })

        df_arena = pd.DataFrame(arena_rows)
        st.dataframe(df_arena, use_container_width=True, hide_index=True)

        fig_arena_bar = go.Figure(data=[
            go.Bar(name='Held-Out Recall@3', x=df_arena["Operational Mode"], y=df_arena["Held-Out Recall@3"], marker_color='#2D5A3D'),
            go.Bar(name='Held-Out MRR', x=df_arena["Operational Mode"], y=df_arena["Held-Out MRR"], marker_color='#2563EB'),
        ])
        fig_arena_bar.update_layout(
            barmode='group',
            title=dict(text="Held-Out Retrieval Quality Across Operational Modes", font=dict(color="#111827", size=13)),
            height=340,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
            yaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
        )
        st.plotly_chart(fig_arena_bar, use_container_width=True, key="plotly_arena_bar_chart")

        st.markdown("#### LinUCB Contextual Bandit Decision Trace")
        rl_data = arena_results.get("rl_guided")
        if rl_data and rl_data.logs:
            trace_items = []
            for l in rl_data.logs:
                trace_items.append({
                    "Step": l.step,
                    "Action": l.action_type,
                    "Targets": ", ".join(l.target_modules),
                    "Decision": l.decision,
                    "Reward": l.reward,
                    "Delta Recall": l.delta_recall,
                    "Reason": l.reason,
                })
            st.dataframe(pd.DataFrame(trace_items), use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# VIEW 8: SYSTEM WALKTHROUGH
# =========================================================================
elif nav_selection == "System Walkthrough":
    st.markdown("""
    <div class="morphe-panel">
        <div class="morphe-panel-step">Verification Suite</div>
        <h3 class="morphe-panel-title">Interactive System Walkthrough</h3>
        <div class="morphe-panel-desc">
            Step-by-step verification demonstrating structural plasticity, routed retrieval, sandbox safety gates, and point-in-time restore.
        </div>
    """, unsafe_allow_html=True)

    demo_stage = st.radio(
        "Demonstration Stage:",
        [
            "Stage 1: Baseline Architecture & Weakness Discovery",
            "Stage 2: Routed Semantic Retrieval (No Global Scans)",
            "Stage 3: Autonomous Split Adaptation",
            "Stage 4: Synthesizing Learned Shortcuts",
            "Stage 5: Safety Gate & Zero-Risk Rollback",
            "Stage 6: Historical Snapshots & 1-Click Restore",
            "Stage 7: Head-to-Head 3-Way Benchmark Arena",
        ],
        horizontal=True,
        key="judge_demo_stage_radio",
    )

    if demo_stage == "Stage 1: Baseline Architecture & Weakness Discovery":
        st.markdown("### Stage 1: Baseline Architecture & Weakness Discovery")
        st.caption("Inspect the initial memory graph topology and observe structural bottlenecks identified by telemetry.")

        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.plotly_chart(generate_plotly_network(system.graph), use_container_width=True, key="showcase_net_s1")

        with c_right:
            st.markdown("#### Baseline Telemetry")
            m = system.get_evaluation_metrics(use_heldout=False)
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Recall@3", f"{m.recall_at_3:.4f}")
            col_m2.metric("MRR", f"{m.mrr:.4f}")
            col_m1.metric("Avg Latency", f"{m.avg_latency_ms:.2f} ms")
            col_m2.metric("Active Modules", len(system.graph.modules))

            st.markdown("---")
            st.markdown("#### Detected Weaknesses (Triggers)")
            cands = detect_adaptation_candidates(system.graph, system.config.thresholds, system.storage)
            if cands:
                for c in cands[:3]:
                    st.markdown(f"""
                    <div style="background: #FFFBEB; border-left: 4px solid #D97706; padding: 10px 14px; border-radius: 4px; margin-bottom: 8px;">
                        <span class="morphe-badge morphe-badge-amber">{c.operation_type.upper()}</span>
                        <b>Target:</b> {', '.join(c.target_modules)}<br>
                        <small style="color: #78350F;">{c.reason}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("System currently balanced.")

        st.info("**Architectural Principle:** Fixed memory topologies degrade as knowledge accumulates. The autonomous monitor continuously identifies oversized clusters, high intra-module variance, and co-accessed retrieval paths.")

    elif demo_stage == "Stage 2: Routed Semantic Retrieval (No Global Scans)":
        st.markdown("### Stage 2: Routed Semantic Retrieval (No Global Scans)")
        st.caption("Queries strictly traverse candidate centroids and 1-hop shortcuts. Global brute-force search is prohibited.")

        preset_q = "How does Raft elect a leader and maintain log replication in distributed state machines?"
        query_input = st.text_input("Test Query", value=preset_q, key="showcase_query_input")
        top_k = st.slider("Top Documents", 1, 5, 3, key="showcase_top_k_slider")

        if st.button("Execute Routed Query", type="primary", key="btn_showcase_query"):
            res = system.query(query_input, top_k=top_k)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Execution Latency", f"{res.latency_ms:.2f} ms")
            c2.metric("Queried Modules", f"{len(res.queried_modules)}", ", ".join(res.queried_modules))
            c3.metric("Shortcut Hops", f"{len(res.shortcut_hops)}")
            c4.metric("Top Score", f"{res.records[0]['score']:.4f}" if res.records else "0.0")

            c_chart, c_res = st.columns([1, 1])
            with c_chart:
                st.markdown("#### Routing Path on Topology")
                st.plotly_chart(
                    generate_plotly_network(system.graph, highlighted_nodes=set(res.queried_modules)),
                    use_container_width=True,
                    key="showcase_traversed_net",
                )

            with c_res:
                st.markdown("#### Centroid Proximity & Retrieved Hits")
                for mod_id, sc in res.centroid_ranked_modules:
                    is_hit = mod_id in res.queried_modules
                    st.markdown(f"`{mod_id}`: Cosine = **{sc:.3f}** {'[Visited]' if is_hit else '[Pruned]'}")

                st.markdown("##### Top Record Content")
                if res.records:
                    top_rec_id = res.records[0]["id"]
                    all_recs = {r.id: r for r in system.storage.get_all_records()}
                    r_obj = all_recs.get(top_rec_id)
                    st.write(r_obj.text if r_obj else top_rec_id)

    elif demo_stage == "Stage 3: Autonomous Split Adaptation":
        st.markdown("### Stage 3: Autonomous Split Adaptation")
        st.caption("Propose, sandbox-validate, and apply a 2-Means split to an overloaded, high-variance module.")

        largest_mod = max(system.graph.get_all_modules(), key=lambda m: len(m.local_index))
        st.write(f"**Target Candidate:** Module `{largest_mod.module_id}` ({len(largest_mod.local_index)} records, variance={largest_mod.local_index.compute_variance():.3f})")

        if st.button("Propose, Sandbox-Validate & Apply Split", type="primary", key="btn_showcase_split"):
            with st.spinner("Cloning memory sandbox and evaluating tuning suite..."):
                split_op = SplitOperation(module_id=largest_mod.module_id)
                val_res = system.validator.validate_adaptation(
                    prod_graph=system.graph,
                    operation=split_op,
                    tuning_queries=system.benchmark_dataset.tuning_queries,
                )

            if val_res.accepted and val_res.trial_graph:
                system.graph.modules = val_res.trial_graph.modules
                system.graph.graph = val_res.trial_graph.graph
                new_v = system.version_manager.commit_version(
                    graph=system.graph,
                    operation_name="split",
                    metrics_dict=val_res.post_metrics.to_dict(),
                    reassignments=val_res.record_reassignments,
                    target_modules=val_res.target_modules,
                    reason=val_res.reason,
                )
                st.success(f"Trial ACCEPTED & Committed as `{new_v}`.")
                st.write(f"**Rationale:** {val_res.reason}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Delta Recall", f"{val_res.delta_recall:+.4f}")
                col2.metric("Delta MRR", f"{val_res.delta_mrr:+.4f}")
                col3.metric("Delta Latency", f"{val_res.delta_latency:+.2f} ms")

                st.plotly_chart(generate_plotly_network(system.graph), use_container_width=True, key="showcase_net_split_done")
            else:
                st.warning(f"Trial REJECTED: {val_res.reason}")

    elif demo_stage == "Stage 4: Synthesizing Learned Shortcuts":
        st.markdown("### Stage 4: Synthesizing Learned Topological Shortcuts")
        st.caption("Creates an edge between frequently co-accessed modules so multi-thematic queries hop across clusters.")

        all_mods = list(system.graph.modules.keys())
        u = all_mods[0]
        v = all_mods[-1]
        st.write(f"**Target Modules for Shortcut:** `{u}` <-> `{v}`")

        if st.button("Synthesize Topological Shortcut", type="primary", key="btn_showcase_shortcut"):
            with st.spinner("Evaluating shortcut impact in sandbox trial copy..."):
                short_op = ShortcutOperation(module_a=u, module_b=v, weight=0.85)
                val_res = system.validator.validate_adaptation(
                    prod_graph=system.graph,
                    operation=short_op,
                    tuning_queries=system.benchmark_dataset.tuning_queries,
                )

            if val_res.accepted and val_res.trial_graph:
                system.graph.modules = val_res.trial_graph.modules
                system.graph.graph = val_res.trial_graph.graph
                new_v = system.version_manager.commit_version(
                    graph=system.graph,
                    operation_name="shortcut",
                    metrics_dict=val_res.post_metrics.to_dict(),
                    reason="Shortcut synthesis for co-accessed pathway",
                )
                st.success(f"Shortcut ACCEPTED & Committed as `{new_v}`.")
                st.plotly_chart(generate_plotly_network(system.graph), use_container_width=True, key="showcase_net_shortcut_done")
            else:
                st.warning(f"Shortcut trial rejected: {val_res.reason}")

    elif demo_stage == "Stage 5: Safety Gate & Zero-Risk Rollback":
        st.markdown("### Stage 5: Safety Gate & Zero-Risk Rollback")
        st.caption("Demonstrating that destructive or degrading structural mutations are rejected before production commit.")

        st.markdown("""
        <div style="background: #FEF2F2; border-left: 4px solid #DC2626; padding: 14px; border-radius: 6px; margin-bottom: 16px;">
            <b style="color: #991B1B;">Mandatory Safety Verification:</b>
            <div style="color: #7F1D1D; font-size: 0.88rem;">
                Inject an intentionally destructive mutation (purging index vectors) inside the sandbox copy.
                The deterministic acceptance gate detects the recall collapse, rejects the candidate, discards the sandbox, and preserves production state.
            </div>
        </div>
        """, unsafe_allow_html=True)

        pre_mods = len(system.graph.modules)

        if st.button("Inject Harmful Mutation (Simulate Corruption)", type="primary", key="btn_showcase_harmful"):
            class DestructiveHarmfulMutation:
                def apply(self, graph, seed=42):
                    for m in graph.get_all_modules():
                        m.local_index.record_ids.clear()
                        m.local_index.vectors = np.empty((0, graph.dimension), dtype=np.float32)
                        m.local_index.id_to_idx.clear()
                    return AdaptationResult(
                        operation_type="destructive_purge",
                        target_modules=["all"],
                        new_modules=[],
                        description="Simulated recall-collapsing mutation",
                        record_reassignments={},
                    )

            with st.spinner("Testing candidate in sandbox copy..."):
                val_harm = system.validator.validate_adaptation(
                    prod_graph=system.graph,
                    operation=DestructiveHarmfulMutation(),
                    tuning_queries=system.benchmark_dataset.tuning_queries,
                )

            if not val_harm.accepted:
                system.version_manager.log_rollback(val_harm)
                st.error(f"Trial REJECTED by Acceptance Gate: {val_harm.reason}")
                st.markdown(f"""
                - **Delta Recall Detected:** <span class="morphe-badge morphe-badge-red">{val_harm.delta_recall:+.4f}</span>
                - **Sandbox Status:** Discarded (Garbage collected)
                - **Production Memory Intact:** Active module count preserved at **{len(system.graph.modules)}** (Pre: {pre_mods})
                - **Audit Trail:** Rollback event permanently logged in SQLite
                """, unsafe_allow_html=True)

    elif demo_stage == "Stage 6: Historical Snapshots & 1-Click Restore":
        st.markdown("### Stage 6: Architecture Versioning & Point-in-Time Restore")
        st.caption("Every architectural commit is versioned in SQLite with metrics, parent pointers, and record assignments.")

        snapshots = system.version_manager.get_history()
        snap_list = [s["version_id"] for s in snapshots]

        c_table, c_restore = st.columns([2, 1])
        with c_table:
            st.markdown("#### Version Revision Timeline")
            t_rows = []
            for s in snapshots:
                t_rows.append({
                    "Version": s["version_id"],
                    "Operation": s["operation"].upper(),
                    "Recall@3": s["metrics"].get("recall_at_3", 0.0),
                    "MRR": s["metrics"].get("mrr", 0.0),
                    "Latency (ms)": s["metrics"].get("avg_latency_ms", 0.0),
                    "Modules": s["metrics"].get("module_count", 0),
                })
            st.dataframe(pd.DataFrame(t_rows), use_container_width=True, hide_index=True)

        with c_restore:
            st.markdown("#### Point-in-Time Restore")
            target_v = st.selectbox("Select Target Snapshot", snap_list, key="showcase_restore_select")
            if st.button("Restore Selected Snapshot", type="primary", key="btn_showcase_restore"):
                system.restore_version(target_v)
                st.success(f"Successfully reverted memory state to `{target_v}`.")
                st.rerun()

    elif demo_stage == "Stage 7: Head-to-Head 3-Way Benchmark Arena":
        st.markdown("### Stage 7: Head-to-Head 3-Way Benchmark Arena")
        st.caption("Compares Static Memory vs Heuristic-Only vs LinUCB RL-Guided on the identical held-out test suite.")

        arena_cycles = st.slider("Adaptation Cycles per Mode", 2, 6, 3, key="showcase_arena_cycles")
        if st.button("Run Three-Way Benchmark Arena", type="primary", key="btn_showcase_run_arena"):
            with st.spinner("Executing three-way benchmark across independent memory graph replicas..."):
                arena_res = system.run_three_way_comparison(num_cycles=arena_cycles)

            rows = []
            for mode_k, exp in arena_res.items():
                h = exp.final_metrics_heldout
                rows.append({
                    "Mode": mode_k.upper(),
                    "Held-Out Recall@3": h.recall_at_3,
                    "Held-Out MRR": h.mrr,
                    "Avg Latency (ms)": h.avg_latency_ms,
                    "Accepted": exp.accepted_count,
                    "Rejected": exp.rejected_count,
                    "Modules": h.module_count,
                })
            df_ar = pd.DataFrame(rows)
            st.dataframe(df_ar, use_container_width=True, hide_index=True)

            fig_bar = go.Figure(data=[
                go.Bar(name='Held-Out Recall@3', x=df_ar["Mode"], y=df_ar["Held-Out Recall@3"], marker_color='#2D5A3D'),
                go.Bar(name='Held-Out MRR', x=df_ar["Mode"], y=df_ar["Held-Out MRR"], marker_color='#2563EB'),
            ])
            fig_bar.update_layout(
                barmode='group',
                title="Held-Out Retrieval Quality Across Operational Modes",
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                height=340,
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="showcase_arena_plot")

    st.markdown("</div>", unsafe_allow_html=True)
