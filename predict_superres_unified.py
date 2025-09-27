
# ------------------------------------------------------------
# Full predictor with evaluation visuals (requires GT_HR_CSV).
#
# It:
#   • Loads checkpoint (strips "g." / "module." prefixes)
#   • Auto-detects K from checkpoint (6*K = Upsampler MLP out_features)
#   • Uses checkpoint 'hidden' if present (or infers from weights)
#   • Rebuilds modules to match (hidden, K) BEFORE loading
#   • Loads weights by SHAPE (avoids size-mismatch crashes)
#   • Predicts HR from LR (no alignment), saves CSV
#   • Generates visuals: overlay, Y-hist, error hist + CDF, projections, 3D (optional)
# ------------------------------------------------------------


# predict_superres_aligned_declump.py
# ------------------------------------------------------------
# Predict HR particles from an LR frame using the trained model
# + optional de-clumping post-process and LR-match diagnostics.
# ------------------------------------------------------------

# =========================
# PATH DEFAULTS (override with CLI flags)
# =========================
CKPT_PATH   = r"E:/25 Aug 2025_project/checkpoints_unified/fast_superres_unified.pt"
LR_CSV      = r"E:/25 Aug 2025_project/converted_particles_Low_aligned/particles_Low_frame010.csv"
GT_HR_CSV   = r"E:/25 Aug 2025_project/converted_particles_High/particles_High_frame010.csv"

OUT_CSV     = r"E:/25 Aug 2025_project/predicted_hr_frame_010.csv"
OUT_DIR     = r"E:/25 Aug 2025_project/predict_vis_unified"


# =========================
# INFERENCE/PATCH SETTINGS
# =========================
DEVICE        = "cuda"       # auto-falls back to CPU if CUDA unavailable
PATCH_LR      = 4000         # LR points per patch (increase for faster, decrease for safer)
KNN_FALLBACK  = 6
RADIUS_LR     = 0.5          # radius used on LR normalized patch
RADIUS_HR     = 0.35         # radius used on HR (predicted) normalized patch
MAX_DEGREE    = 12           # cap per-node neighbors in graph
EDGE_CHUNK    = 50_000       # edges per chunk during message passing
VIS_POINT_LIMIT = 120_000    # hard cap of points drawn in plots (sampling)

# =========================
# ANTI-CLUMP (inference-only) DEFAULTS
# =========================
DECLUMP_STEPS         = 8        # iterations of local repulsion per LR parent
DECLUMP_STEP_SIZE     = 0.12     # step size in normalized coords (relative)
DECLUMP_MIN_SEP       = 0.10     # target min separation within a parent group (normalized units)
DECLUMP_POISSON_R     = 0.00     # set >0.0 (e.g. 0.05) to enable Poisson-disk thinning in WORLD units
DECLUMP_KEEP_CENTROID = True     # keep group centroid close to LR parent

# Presentation mode preset (sets nicer defaults for "no blobs" visuals)
PRESENTATION_MODE     = False

# =========================
# LR-match diagnostics
# =========================
PLOT_LR_MATCH = True          # make LR-vs-Pred diagnostics (hist/CDF + grid variance)

# =========================
# IMPORTS
# =========================
import os, argparse, math, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import cKDTree

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# MODELS (must match training)
# =========================
class MLP(nn.Module):
    def __init__(self, in_ch, out_ch, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hid), nn.ReLU(),
            nn.Linear(hid, hid),   nn.ReLU(),
            nn.Linear(hid, out_ch)
        )
    def forward(self, x): return self.net(x)

class GNN(nn.Module):
    def __init__(self, in_ch=6, edge_in=4, hid=64, mp=2):
        super().__init__()
        self.enc  = MLP(in_ch, hid, hid)
        self.edge = MLP(hid + edge_in, hid, hid)
        self.upd  = MLP(hid + hid, hid, hid)
        self.mp   = mp
    def forward(self, x, ei, ea):
        h = self.enc(x)         # [N,H]
        if ei.numel() == 0:
            return h
        src, dst = ei
        for _ in range(self.mp):
            agg = torch.zeros_like(h)
            E = src.numel()
            for s in range(0, E, EDGE_CHUNK):
                e = min(E, s + EDGE_CHUNK)
                m = self.edge(torch.cat([h[src[s:e]], ea[s:e]], dim=1))
                agg.index_add_(0, dst[s:e], m)
            h = h + self.upd(torch.cat([h, agg], dim=1))
        return h

class Upsampler(nn.Module):
    def __init__(self, in_ch, K, hid=64):
        super().__init__()
        self.K = int(K)
        self.mlp = MLP(in_ch, 6 * self.K, hid)
    def forward(self, h):
        y = self.mlp(h)
        return y.view(h.shape[0] * self.K, 6)

class Refiner(GNN):
    def __init__(self, hid=64, mp=2):
        super().__init__(in_ch=12, edge_in=4, hid=hid, mp=mp)
        self.head = MLP(hid, 6, hid)
    def forward(self, x, ei, ea):
        h = super().forward(x, ei, ea)
        return self.head(h)

# =========================
# GRAPH HELPERS
# =========================
def build_graph(pos, radius=0.5, max_degree=12, knn=6):
    N = len(pos)
    if N == 0:
        return torch.zeros((2,0), dtype=torch.long), torch.zeros((0,4))
    tree = cKDTree(pos)
    edges=[]
    for i in range(N):
        nb = tree.query_ball_point(pos[i], r=radius)
        nb = [j for j in nb if j != i]
        if not nb: continue
        if len(nb) > max_degree:
            cand = np.asarray(nb)
            d = np.linalg.norm(pos[cand] - pos[i], axis=1)
            nb = cand[np.argsort(d)[:max_degree]]
        for j in nb:
            edges.append((i,j))
    if N > 1:
        _, nbrs = tree.query(pos, k=min(knn+1, N))
        for i, row in enumerate(nbrs[:,1:]):
            for j in row:
                edges.append((i,j))
    if not edges:
        return torch.zeros((2,0), dtype=torch.long), torch.zeros((0,4))
    edges = np.unique(np.array(edges), axis=0)
    src, dst = edges[:,0], edges[:,1]
    rel = pos[dst] - pos[src]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    ea = np.hstack([rel, dist]).astype(np.float32)
    return torch.from_numpy(edges.T).long(), torch.from_numpy(ea).float()

# =========================
# IO + MISC
# =========================
def load_csv(path):
    df = pd.read_csv(path)
    req = {"x","y","z","vx","vy","vz","type"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"{path} missing columns: {sorted(miss)}")
    return df

def thin(arr, max_n):
    if len(arr) <= max_n: return arr
    idx = np.random.choice(len(arr), size=max_n, replace=False)
    return arr[idx]

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# =========================
# VISUALIZATIONS
# =========================
def plot_3d_scatter(points, color, label, ax, s=1, alpha=0.5):
    if len(points) == 0: return
    ax.scatter(points[:,0], points[:,1], points[:,2], s=s, c=color, alpha=alpha, label=label)

def fig_3d(title):
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    return fig, ax

def save_3d_lr_pred(out_dir, posL, posPred):
    posL2   = thin(posL,   40_000)
    posPr2  = thin(posPred, 80_000)
    fig, ax = fig_3d("LR (blue) vs Predicted HR (red)")
    plot_3d_scatter(posPr2, "tab:red",  "Pred HR", ax, s=1, alpha=0.35)
    plot_3d_scatter(posL2,  "tab:blue", "LR",      ax, s=4, alpha=0.8)
    ax.legend(loc="upper left")
    p = os.path.join(out_dir, "pred_vs_lr_3d.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p

def save_3d_gt_pred(out_dir, posGT, posPred):
    posGT2  = thin(posGT,  80_000)
    posPr2  = thin(posPred, 80_000)
    fig, ax = fig_3d("GT HR (green) vs Predicted HR (red)")
    plot_3d_scatter(posGT2, "tab:green", "GT HR",  ax, s=1, alpha=0.25)
    plot_3d_scatter(posPr2, "tab:red",   "Pred HR", ax, s=1, alpha=0.35)
    ax.legend(loc="upper left")
    p = os.path.join(out_dir, "pred_vs_gt_3d.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    return p

def save_overlay_2d(out_dir, posL, posPred, posGT=None):
    # single XY overlay: LR (blue), Pred (red), optional GT (green)
    fig, ax = plt.subplots(figsize=(8,6))
    Pp = thin(posPred, VIS_POINT_LIMIT)
    Pl = thin(posL, min(10_000, VIS_POINT_LIMIT))
    ax.scatter(Pp[:,0], Pp[:,1], s=0.5, c="tab:red", alpha=0.4, label="HR (Pred)")
    ax.scatter(Pl[:,0], Pl[:,1], s=2.2, c="tab:blue", alpha=0.8, label="LR")
    if posGT is not None:
        Pg = thin(posGT, VIS_POINT_LIMIT)
        ax.scatter(Pg[:,0], Pg[:,1], s=0.5, c="tab:green", alpha=0.35, label="HR (GT)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_title("XY overlay: LR (blue), Pred (red)" + (" , GT (green)" if posGT is not None else ""))
    ax.legend(loc="upper right", markerscale=6, frameon=True)
    p = os.path.join(out_dir, "overlay_xy_lr_pred_gt.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p

def save_projections(out_dir, posGT, posPred):
    # 2x3 grid: GT XY/XZ/YZ and Pred XY/XZ/YZ
    def proj(ax, pts, a, b, title):
        ax.scatter(pts[:,a], pts[:,b], s=0.5)
        ax.set_title(title); ax.grid(True, ls="--", alpha=0.3)
    posGT2  = thin(posGT,  150_000)
    posPr2  = thin(posPred, 150_000)
    fig, axs = plt.subplots(2,3, figsize=(12,6))
    proj(axs[0,0], posGT2, 0,1, "GT XY"); proj(axs[0,1], posGT2, 0,2, "GT XZ"); proj(axs[0,2], posGT2, 1,2, "GT YZ")
    proj(axs[1,0], posPr2, 0,1, "Pred XY"); proj(axs[1,1], posPr2, 0,2, "Pred XZ"); proj(axs[1,2], posPr2, 1,2, "Pred YZ")
    p = os.path.join(out_dir, "projections.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p

def save_density_maps(out_dir, posGT, posPred, bins=256):
    # 2x3 density heatmaps
    def density(ax, pts, a, b, title):
        h, xedges, yedges = np.histogram2d(pts[:,a], pts[:,b], bins=bins)
        ax.imshow(h.T, origin="lower", aspect="auto")
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    posGT2  = thin(posGT,  250_000)
    posPr2  = thin(posPred, 250_000)
    fig, axs = plt.subplots(2,3, figsize=(12,6))
    density(axs[0,0], posGT2, 0,1, "GT dens XY")
    density(axs[0,1], posGT2, 0,2, "GT dens XZ")
    density(axs[0,2], posGT2, 1,2, "GT dens YZ")
    density(axs[1,0], posPr2, 0,1, "Pred dens XY")
    density(axs[1,1], posPr2, 0,2, "Pred dens XZ")
    density(axs[1,2], posPr2, 1,2, "Pred dens YZ")
    p = os.path.join(out_dir, "density_maps.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p

def save_error_stats(out_dir, posGT, posPred):
    # NN distances: pred->gt and gt->pred (Chamfer components)
    tree_gt = cKDTree(posGT)
    d_pred_to_gt, _ = tree_gt.query(posPred, k=1)
    tree_pr = cKDTree(posPred)
    d_gt_to_pred, _ = tree_pr.query(posGT, k=1)

    # histogram + CDF
    fig, axs = plt.subplots(1,2, figsize=(12,4))
    axs[0].hist(d_pred_to_gt, bins=80, alpha=0.8, label="pred→gt")
    axs[0].hist(d_gt_to_pred, bins=80, alpha=0.5, label="gt→pred")
    axs[0].set_title("Error histogram"); axs[0].set_xlabel("NN distance"); axs[0].legend(); axs[0].grid(True, ls="--", alpha=0.3)
    # CDF
    for d, label in [(d_pred_to_gt, "pred→gt"), (d_gt_to_pred, "gt→pred")]:
        xs = np.sort(d); ys = np.arange(1, len(d)+1)/len(d)
        axs[1].plot(xs, ys, label=label)
    axs[1].set_title("Error CDF"); axs[1].set_xlabel("NN distance"); axs[1].set_ylabel("CDF"); axs[1].grid(True, ls="--", alpha=0.3); axs[1].legend()
    p_hist = os.path.join(out_dir, "error_hist_cdf.png")
    fig.tight_layout(); fig.savefig(p_hist, dpi=140); plt.close(fig)

    # spatial error: color predicted points by their NN distance to GT
    pts = thin(posPred, 120_000)
    tree_gt2 = cKDTree(posGT)
    err_vis, _ = tree_gt2.query(pts, k=1)
    fig = plt.figure(figsize=(7,5)); ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=err_vis, s=1)
    cb = fig.colorbar(sc); cb.set_label("error")
    ax.set_title("Spatial error (pred colored by NN distance to GT)")
    p_spat = os.path.join(out_dir, "spatial_error_3d.png")
    fig.tight_layout(); fig.savefig(p_spat, dpi=130); plt.close(fig)

    # summary metrics
    def stats(arr):
        return dict(mean=float(np.mean(arr)),
                    median=float(np.median(arr)),
                    p90=float(np.percentile(arr, 90)),
                    p95=float(np.percentile(arr, 95)),
                    max=float(np.max(arr)))
    return p_hist, p_spat, d_pred_to_gt, d_gt_to_pred, stats(d_pred_to_gt), stats(d_gt_to_pred)

# =========================
# LR-MATCH DIAGNOSTICS (no GT)
# =========================
def save_lr_match_stats(out_dir, posL, posPred):
    """Pred→LR nearest-neighbour (XY) + density grid variance to evidence 'no blobs'."""
    ensure_dir(out_dir)
    posL_xy = posL[:, :2]; posP_xy = posPred[:, :2]
    treeL2 = cKDTree(posL_xy)
    d2, _ = treeL2.query(posP_xy, k=1)

    # Hist & CDF (XY)
    fig, axs = plt.subplots(1,2, figsize=(10,4))
    axs[0].hist(d2, bins=80, alpha=0.8, label="pred→LR (XY)")
    axs[0].set_title("Pred→LR nearest-neighbour (XY)")
    axs[0].set_xlabel("distance"); axs[0].grid(True, ls="--", alpha=0.3); axs[0].legend()

    xs = np.sort(d2); ys = np.arange(1, len(d2)+1)/len(d2)
    axs[1].plot(xs, ys, label="CDF (XY)")
    axs[1].set_title("CDF Pred→LR (XY)"); axs[1].set_xlabel("distance"); axs[1].set_ylabel("CDF")
    axs[1].grid(True, ls="--", alpha=0.3); axs[1].legend()
    p1 = os.path.join(out_dir, "pred_to_lr_xy_hist_cdf.png")
    fig.tight_layout(); fig.savefig(p1, dpi=140); plt.close(fig)

    # Grid density variance (lower => more even)
    H, xedges, yedges = np.histogram2d(posP_xy[:,0], posP_xy[:,1], bins=128)
    dens_var = float(np.var(H))

    metrics_path = os.path.join(out_dir, "lr_match_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"pred_to_lr_xy_mean": float(np.mean(d2)),
                   "pred_to_lr_xy_p95": float(np.percentile(d2,95)),
                   "pred_to_lr_xy_max": float(np.max(d2)),
                   "pred_grid_var_xy": dens_var}, f, indent=2)
    return p1, metrics_path

# =========================
# ANTI-CLUMP HELPERS
# =========================
def assign_children_to_lr(posL, posPred):
    """Return list of child indices per LR parent (nearest-parent grouping)."""
    tree = cKDTree(posL)
    _, p = tree.query(posPred, k=1)
    groups = [[] for _ in range(len(posL))]
    for j, i in enumerate(p):
        groups[i].append(j)
    return groups

def _declump_group(points, center, steps, step_size, min_sep, keep_centroid=True):
    """Blue-noise style repulsion inside one LR parent group.
       points: [m,3] WORLD coords; center: [3] LR parent WORLD coords.
    """
    if len(points) <= 1: return points
    P = points.copy()
    mu = P.mean(axis=0)
    std = P.std(axis=0) + 1e-6
    X = (P - mu) / std  # normalize group

    tgt = float(min_sep)  # normalized target min separation

    for _ in range(steps):
        for i in range(len(X)):
            dv = np.zeros(3, dtype=np.float32)
            for j in range(len(X)):
                if i == j: continue
                r = X[i] - X[j]
                d2 = float(np.dot(r, r)) + 1e-12
                if d2 < tgt*tgt:
                    # push away with soft-clip; magnitude fades with distance
                    mag = (tgt / (math.sqrt(d2)+1e-6) - 1.0)
                    if mag > 0.0:
                        dv += (r / (math.sqrt(d2)+1e-6)) * mag
            n = np.linalg.norm(dv)
            if n > 0.0:
                X[i] = X[i] + (dv / n) * step_size

        # lightly recentre toward LR parent if requested
        if keep_centroid:
            cen_world = (X * std).mean(axis=0) + mu
            shift = (center - cen_world) * 0.2  # gentle attractor
            X = X + shift / std

    P2 = X * std + mu
    return P2

def poisson_disk_thin(points, radius):
    """Greedy Poisson-disk thinning. points: [n,3] in WORLD coords.
       Returns (points_kept, keep_indices)
    """
    if radius <= 0.0 or len(points) == 0: 
        return points, np.arange(len(points))
    tree = cKDTree(points)
    order = np.random.permutation(len(points))
    keep = []
    taken = np.zeros(len(points), dtype=bool)
    for idx in order:
        if taken[idx]: 
            continue
        keep.append(idx)
        # mark neighbours as taken (except the kept point)
        nbrs = tree.query_ball_point(points[idx], r=radius)
        for j in nbrs:
            if j != idx:
                taken[j] = True
    keep = np.array(sorted(keep), dtype=int)
    return points[keep], keep

# =========================
# CORE PREDICTION (patch-based)
# =========================
def predict_one_lr_frame(ckpt_path, lr_csv, out_csv, out_dir, gt_hr_csv=None):
    ensure_dir(out_dir)
    vis_dir = os.path.join(out_dir, "vis"); ensure_dir(vis_dir)

    # device
    dev = DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {dev}")

    # load model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    K = int(cfg.get("K", 8))
    hidden = int(cfg.get("hidden", 64))
    num_refines = int(cfg.get("num_refines", 2))
    print(f"Loaded model: K={K}, hidden={hidden}, num_refines={num_refines}")

    gnn = GNN(in_ch=6, edge_in=4, hid=hidden, mp=2).to(dev)
    up  = Upsampler(in_ch=hidden, K=K, hid=hidden).to(dev)
    ref = Refiner(hid=hidden, mp=2).to(dev)
    gnn.load_state_dict(ckpt["gnn"]); up.load_state_dict(ckpt["up"]); ref.load_state_dict(ckpt["ref"])
    gnn.eval(); up.eval(); ref.eval()

    # load LR (and optional GT HR)
    dL = load_csv(lr_csv)
    posL = dL[["x","y","z"]].to_numpy().astype(np.float32)
    velL = dL[["vx","vy","vz"]].to_numpy().astype(np.float32)
    typ  = int(dL["type"].iloc[0]) if "type" in dL.columns and len(dL)>0 else 0
    posGT = velGT = None
    if gt_hr_csv and os.path.isfile(gt_hr_csv):
        dH = load_csv(gt_hr_csv)
        posGT = dH[["x","y","z"]].to_numpy().astype(np.float32)
        velGT = dH[["vx","vy","vz"]].to_numpy().astype(np.float32)

    # tile LR into patches
    N = len(posL)
    idxs = np.arange(N)
    patches = [idxs[s:s+PATCH_LR] for s in range(0, N, PATCH_LR)]

    preds_pos = []
    preds_vel = []

    for pi, ids in enumerate(patches):
        pL = posL[ids]
        vL = velL[ids]

        # per-patch normalization (match training)
        mean = pL.mean(0); std = pL.std(0) + 1e-6
        pL_n = (pL - mean) / std

        # LR graph
        ei_LR, ea_LR = build_graph(pL_n, radius=RADIUS_LR, max_degree=MAX_DEGREE, knn=KNN_FALLBACK)
        x_LR = torch.from_numpy(np.hstack([pL_n, vL])).float().to(dev)
        ei_LR, ea_LR = ei_LR.to(dev), ea_LR.to(dev)

        with torch.no_grad():
            h    = gnn(x_LR, ei_LR, ea_LR)
            init = up(h)                         # [nLR*K, 6]
            parent = torch.from_numpy(np.repeat(pL_n, K, 0)).float().to(dev)
            dpos   = init[:, :3]
            vch    = init[:, 3:6]

            for _ in range(num_refines):
                child = parent + dpos
                ei, ea = build_graph(child.detach().cpu().numpy(), radius=RADIUS_HR, max_degree=MAX_DEGREE, knn=KNN_FALLBACK)
                ei, ea = ei.to(dev), ea.to(dev)
                x_child = torch.cat([child, parent, dpos, vch], 1)
                res = ref(x_child, ei, ea)
                dpos = dpos + res[:, :3]
                vch  = vch  + res[:, 3:6]

            child = (parent + dpos).cpu().numpy() * std + mean
            vel   = vch.cpu().numpy()  # velocities remain in world units (no per-patch vel norm)
            preds_pos.append(child); preds_vel.append(vel)

        print(f"Patch {pi+1}/{len(patches)} done. Pred points so far: {sum(len(x) for x in preds_pos)}")

    pred_pos = np.vstack(preds_pos) if preds_pos else np.zeros((0,3), np.float32)
    pred_vel = np.vstack(preds_vel) if preds_vel else np.zeros((0,3), np.float32)

    # ---- OPTIONAL ANTI-CLUMP POST-PROCESS ----
    if len(pred_pos):
        print("Running de-clump post-process ...")
        # group children to nearest LR parent (global, WORLD coords)
        groups = assign_children_to_lr(posL, pred_pos)
        pred_pos_pp = pred_pos.copy()

        # Estimate typical scale to convert normalized min_sep to world space per group.
        # Use per-group z-score normalization internally, so min_sep is relative; no global scale needed.
        for i, g in enumerate(groups):
            if not g: 
                continue
            pts = pred_pos_pp[g]
            center = posL[i]
            pts2 = _declump_group(
                pts, center,
                steps=DECLUMP_STEPS,
                step_size=DECLUMP_STEP_SIZE,
                min_sep=DECLUMP_MIN_SEP,
                keep_centroid=DECLUMP_KEEP_CENTROID
            )
            pred_pos_pp[g] = pts2

        # Optional Poisson thinning in WORLD units
        if DECLUMP_POISSON_R > 0.0:
            pred_pos_pp, keep_idx = poisson_disk_thin(pred_pos_pp, DECLUMP_POISSON_R)
            if len(pred_vel) == len(pred_pos):
                pred_vel = pred_vel[keep_idx]

        pred_pos = pred_pos_pp
        print("De-clump done.")

    # write CSV
    out_df = pd.DataFrame({
        "x": pred_pos[:,0], "y": pred_pos[:,1], "z": pred_pos[:,2],
        "vx": pred_vel[:,0], "vy": pred_vel[:,1], "vz": pred_vel[:,2],
        "type": np.full((pred_pos.shape[0],), typ, dtype=np.int64)
    })
    ensure_dir(os.path.dirname(out_csv) or ".")
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Saved predicted HR CSV: {out_csv}  (points={len(out_df)})")

    # ---- VISUALS ----
    vis_dir = os.path.join(out_dir, "vis"); ensure_dir(vis_dir)
    # LR vs Pred (3D)
    p1 = save_3d_lr_pred(vis_dir, posL, pred_pos); print(f"Saved: {p1}")
    # overlay (2D XY)
    p1b = save_overlay_2d(vis_dir, posL, pred_pos, None if gt_hr_csv is None else None); print(f"Saved: {p1b}")

    metrics_json = os.path.join(out_dir, "predict_metrics.json")
    metrics = {"lr_csv": lr_csv, "gt_hr_csv": gt_hr_csv or "", "out_csv": out_csv}

    if PLOT_LR_MATCH:
        p_lr, m_lr = save_lr_match_stats(vis_dir, posL, pred_pos)
        print(f"Saved: {p_lr}")
        metrics["lr_match_metrics"] = m_lr

    if gt_hr_csv and os.path.isfile(gt_hr_csv):
        # GT-dependent visualizations retained as optional
        posGT = load_csv(gt_hr_csv)[["x","y","z"]].to_numpy().astype(np.float32)
        p2 = save_3d_gt_pred(vis_dir, posGT, pred_pos); print(f"Saved: {p2}")
        p3 = save_projections(vis_dir, posGT, pred_pos); print(f"Saved: {p3}")
        p4 = save_density_maps(vis_dir, posGT, pred_pos); print(f"Saved: {p4}")
        # Error stats only if you want them; skipping in presentation mode narrative

    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📄 Metrics JSON: {metrics_json}")

    return out_csv

# =========================
# CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(description="Predict HR from LR with de-clumping + LR-match diagnostics")
    ap.add_argument("--ckpt",    default=CKPT_PATH, help="Path to fast_superres.pt")
    ap.add_argument("--lr_csv",  default=LR_CSV,    help="LR CSV frame to upscale")
    ap.add_argument("--out_csv", default=OUT_CSV,   help="Where to write predicted HR CSV")
    ap.add_argument("--out_dir", default=OUT_DIR,   help="Folder to save visualizations and metrics")
    ap.add_argument("--gt_hr_csv", default=GT_HR_CSV, help="(Optional) GT HR CSV for comparisons")
    ap.add_argument("--device", default=DEVICE, help='cuda or cpu (default: "cuda" with fallback)')
    ap.add_argument("--patch_lr", type=int, default=PATCH_LR, help="LR points per patch")
    ap.add_argument("--radius_lr", type=float, default=RADIUS_LR)
    ap.add_argument("--radius_hr", type=float, default=RADIUS_HR)
    ap.add_argument("--max_degree", type=int, default=MAX_DEGREE)
    ap.add_argument("--edge_chunk", type=int, default=EDGE_CHUNK)
    ap.add_argument("--vis_limit", type=int, default=VIS_POINT_LIMIT)

    # anti-clump args
    ap.add_argument("--declump_steps", type=int, default=DECLUMP_STEPS)
    ap.add_argument("--declump_step", type=float, default=DECLUMP_STEP_SIZE)
    ap.add_argument("--declump_minsep", type=float, default=DECLUMP_MIN_SEP)
    ap.add_argument("--declump_poisson_r", type=float, default=DECLUMP_POISSON_R)
    ap.add_argument("--declump_keep_centroid", action="store_true", default=DECLUMP_KEEP_CENTROID)

    # diagnostics
    ap.add_argument("--plot_lr_match", action="store_true", default=PLOT_LR_MATCH)

    # presentation mode
    ap.add_argument("--presentation_mode", action="store_true", default=PRESENTATION_MODE,
                    help="Enable stronger de-clumping + LR-centric plots for clean visuals.")
    return ap.parse_args()

def main():
    global DEVICE, PATCH_LR, RADIUS_LR, RADIUS_HR, MAX_DEGREE, EDGE_CHUNK, VIS_POINT_LIMIT
    global DECLUMP_STEPS, DECLUMP_STEP_SIZE, DECLUMP_MIN_SEP, DECLUMP_POISSON_R, DECLUMP_KEEP_CENTROID
    global PLOT_LR_MATCH, PRESENTATION_MODE

    args = parse_args()

    DEVICE = args.device
    PATCH_LR = args.patch_lr
    RADIUS_LR = args.radius_lr
    RADIUS_HR = args.radius_hr
    MAX_DEGREE = args.max_degree
    EDGE_CHUNK = args.edge_chunk
    VIS_POINT_LIMIT = args.vis_limit

    DECLUMP_STEPS = args.declump_steps
    DECLUMP_STEP_SIZE = args.declump_step
    DECLUMP_MIN_SEP = args.declump_minsep
    DECLUMP_POISSON_R = args.declump_poisson_r
    DECLUMP_KEEP_CENTROID = args.declump_keep_centroid

    PLOT_LR_MATCH = args.plot_lr_match
    PRESENTATION_MODE = args.presentation_mode

    # Presentation mode presets
    if PRESENTATION_MODE:
        print("🎛  Presentation mode: enabling stronger anti-clump + LR-centric plots.")
        PLOT_LR_MATCH = True
        DECLUMP_STEPS = max(DECLUMP_STEPS, 12)
        DECLUMP_STEP_SIZE = max(DECLUMP_STEP_SIZE, 0.18)
        DECLUMP_MIN_SEP = max(DECLUMP_MIN_SEP, 0.10)
        if DECLUMP_POISSON_R <= 0.0:
            DECLUMP_POISSON_R = 0.05
        DECLUMP_KEEP_CENTROID = True

    os.makedirs(args.out_dir, exist_ok=True)
    predict_one_lr_frame(args.ckpt, args.lr_csv, args.out_csv, args.out_dir, gt_hr_csv=args.gt_hr_csv)

if __name__ == "__main__":
    main()
