
# train_superres_unified.py
# ------------------------------------------------------------
# Training script (no alignment) with anti-blob shaping and
# even-distribution (blue-noise) term. Saves a checkpoint that
# includes config (K, hidden, etc.) so the paired predictor
# loads cleanly without shape/key errors.
# ------------------------------------------------------------

# ==== EDIT THESE PATHS IF NEEDED ====
LR_DIR   = r"E:/25 Aug 2025_project/converted_particles_Low"
HR_DIR   = r"E:/25 Aug 2025_project/converted_particles_High"
OUT_DIR  = r"E:/25 Aug 2025_project/checkpoints_unified"
CKPT_NAME = "fast_superres_unified.pt"

# ==== TRAINING PARAMS ====
EPOCHS        = 100
BATCHES       = 220
DEVICE        = "cuda"
SEED          = 42

# Model / graph
HIDDEN        = 96
NUM_REFINES   = 2
MAX_DEGREE    = 12
EDGE_CHUNK    = 80_000

# Expansion (K)
AUTO_K        = True
K_FALLBACK    = 32
K_CAP         = 64

# Patch sizes
PATCH_LR      = 400
PATCH_HR      = 40_000

# Base loss weights
W_POS         = 1.00
W_VEL         = 0.30
W_CHAMFER     = 0.40
CHAMFER_SAMP  = 25_000

# Anti-blob & band shaping
W_VAR_Y_INIT   = 0.16
W_REPULSE_INIT = 0.030
REP_SIGMA      = 0.020
REP_SUBSAMPLE  = 6000

# Y-band shaping (robust)
W_Y_MEAN_INIT  = 0.06
W_Y_BAND_INIT  = 0.08
Y_BAND_K       = 1.5

# Small XY cohesion
W_XY_COHERE_INIT = 0.02

# Even-distribution (blue-noise)
W_EVEN   = 0.05
R_MIN    = 0.03
R_MAX    = 0.08

# Visuals
VIS_EVERY_BATCH = 50
VIS_POINT_LIMIT = 20_000

import os, re, glob, random, numpy as np, pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def ensure_dir(d): os.makedirs(d, exist_ok=True); return d

def frame_index(path: str) -> int:
    m = re.search(r'(\d+)(?=\.csv$)', os.path.basename(path)); return int(m.group(1)) if m else -1

def map_by_frame(folder: str):
    paths = glob.glob(os.path.join(folder, "*.csv"))
    return {frame_index(p): p for p in paths if frame_index(p) >= 0}

def pair_lr_hr(lr_dir, hr_dir):
    lr_map = map_by_frame(lr_dir); hr_map = map_by_frame(hr_dir)
    common = sorted(set(lr_map) & set(hr_map))
    if not common: raise RuntimeError("No paired LR/HR frames found.")
    return [lr_map[i] for i in common], [hr_map[i] for i in common]

def load_csv(path):
    df = pd.read_csv(path)
    req = {"x","y","z","vx","vy","vz","type"}
    miss = req - set(df.columns)
    if miss: raise RuntimeError(f"{os.path.basename(path)} missing columns: {sorted(miss)}")
    return df

def build_graph(pos, radius=0.5, max_degree=12, knn=6):
    N = len(pos)
    if N == 0: return torch.zeros((2,0), dtype=torch.long), torch.zeros((0,4))
    tree = cKDTree(pos); edges=[]
    for i in range(N):
        nb = tree.query_ball_point(pos[i], r=radius); nb = [j for j in nb if j != i]
        if not nb: continue
        if len(nb) > max_degree:
            cand = np.asarray(nb); d = np.linalg.norm(pos[cand] - pos[i], axis=1)
            nb = cand[np.argsort(d)[:max_degree]]
        edges.extend((i,j) for j in nb)
    if N > 1:
        _, nbrs = tree.query(pos, k=min(knn+1, N))
        for i, row in enumerate(nbrs[:,1:]):
            edges.extend((i,int(j)) for j in row)
    if not edges: return torch.zeros((2,0), dtype=torch.long), torch.zeros((0,4))
    edges = np.unique(np.array(edges), axis=0)
    src, dst = edges[:,0], edges[:,1]
    rel = pos[dst] - pos[src]; dist = np.linalg.norm(rel, axis=1, keepdims=True)
    ea = np.hstack([rel, dist]).astype(np.float32)
    return torch.from_numpy(edges.T).long(), torch.from_numpy(ea).float()

def assign_hr_to_lr_parents(posL, posH, K):
    tree = cKDTree(posL); _, p = tree.query(posH, k=1)
    groups = [[] for _ in range(len(posL))]
    for j, i in enumerate(p): groups[i].append(j)
    out=[]
    for i,g in enumerate(groups):
        if not g: out.append([]); continue
        d = np.linalg.norm(posH[g] - posL[i], axis=1); order = np.argsort(d)
        out.append([g[k] for k in order[:K]])
    return out

def thin(arr, nmax):
    if len(arr) <= nmax: return arr
    idx = np.random.choice(len(arr), size=nmax, replace=False); return arr[idx]

def estimate_K95(lr_dir, hr_dir):
    lr_map, hr_map = map_by_frame(lr_dir), map_by_frame(hr_dir)
    common = sorted(set(lr_map) & set(hr_map))
    ratios=[]
    for i in common:
        try:
            nL = len(pd.read_csv(lr_map[i])); nH = len(pd.read_csv(hr_map[i]))
            if nL>0 and nH>0: ratios.append(nH/nL)
        except Exception: pass
    if not ratios: return None
    q95 = float(np.percentile(np.array(ratios), 95))
    return int(np.clip(np.ceil(q95), 1, K_CAP))

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
        h = self.enc(x)
        if ei.numel() == 0: return h
        src, dst = ei; E = src.numel()
        for _ in range(self.mp):
            agg = torch.zeros_like(h)
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
        self.mlp = MLP(in_ch, 6*self.K, hid)
    def forward(self, h): return self.mlp(h).view(h.shape[0]*self.K, 6)

class Refiner(GNN):
    def __init__(self, hid=64, mp=2):
        super().__init__(in_ch=12, edge_in=4, hid=hid, mp=mp)
        self.head = MLP(hid, 6, hid)
    def forward(self, x, ei, ea): return self.head(super().forward(x, ei, ea))

def save_patch_visual(vis_dir, epoch, batch_i, posL, posH, posPred, title_suffix=""):
    posL2 = thin(posL, min(5_000, VIS_POINT_LIMIT))
    posH2 = thin(posH, VIS_POINT_LIMIT)
    posPr = thin(posPred, VIS_POINT_LIMIT)
    fig, ax = plt.subplots(figsize=(10,3.2))
    ax.scatter(posH2[:,0],posH2[:,1],s=1,c="tab:green",alpha=0.25,label="HR (GT)")
    ax.scatter(posPr[:,0],posPr[:,1], s=3,c="tab:red",  alpha=0.55,label="HR (Pred)")
    ax.scatter(posL2[:,0],posL2[:,1],s=8,c="tab:blue",  alpha=0.80,label="LR")
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, ls="--", alpha=0.3)
    ax.set_title(f"Epoch {epoch}  Batch {batch_i}  {title_suffix}")
    ax.legend(loc="upper right", markerscale=6, frameon=True)
    out = os.path.join(vis_dir, f"epoch_{epoch:03d}_batch_{batch_i:04d}.png")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig); return out

def save_y_hist(vis_dir, epoch, child_world, gt_world):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(child_world[:,1], bins=80, alpha=0.6, label="Pred y")
    ax.hist(gt_world[:,1],    bins=80, alpha=0.4, label="GT y")
    ax.set_title(f"Y-distribution epoch {epoch}")
    ax.legend(); ax.grid(True, ls="--", alpha=0.3)
    p = os.path.join(vis_dir, f"epoch_{epoch:03d}_y_hist.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)

def save_lr_hr_overlay(vis_dir, epoch, posL, posH):
    fig, ax = plt.subplots(figsize=(10,3.2))
    ax.scatter(thin(posH,40_000)[:,0], thin(posH,40_000)[:,1], s=1, c="tab:green", alpha=0.35, label="HR")
    ax.scatter(thin(posL,10_000)[:,0], thin(posL,10_000)[:,1], s=4, c="tab:blue",  alpha=0.80, label="LR")
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="upper right"); ax.set_title(f"Epoch {epoch}  LR vs HR sanity (XY)")
    p = os.path.join(vis_dir, f"epoch_{epoch:03d}_lr_vs_hr_xy.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)

def even_distribution_loss(points, r_min=R_MIN, r_max=R_MAX, k=8):
    if points.shape[0] < 2:
        return torch.tensor(0.0, device=points.device)
    d = torch.cdist(points, points, p=2)
    d = d + torch.eye(points.shape[0], device=points.device) * 1e9
    k_eff = min(k, max(1, points.shape[0]-1))
    knn_d, _ = torch.topk(d, k=k_eff, largest=False)
    loss_close = F.relu(r_min - knn_d).pow(2).mean()
    loss_far   = F.relu(knn_d - r_max).pow(2).mean()
    return loss_close + 0.1 * loss_far

def main():
    set_seed(SEED)
    assert os.path.isdir(LR_DIR) and os.path.isdir(HR_DIR), "Check LR_DIR/HR_DIR"
    ensure_dir(OUT_DIR); vis_dir = ensure_dir(os.path.join(OUT_DIR, "vis"))

    # Auto K from LR/HR ratio distribution
    K = K_FALLBACK
    if AUTO_K:
        est = estimate_K95(LR_DIR, HR_DIR)
        if est is not None: K = int(est)
    K = int(np.clip(K, 1, K_CAP))
    print(f"Using K = {K}")

    lr_files, hr_files = pair_lr_hr(LR_DIR, HR_DIR)
    print(f"Paired frames: {len(lr_files)}")

    dev = DEVICE if (DEVICE=="cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Device: {dev}")
    amp_enable = (dev == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enable)

    gnn = GNN(in_ch=6, edge_in=4, hid=HIDDEN, mp=2).to(dev)
    up  = Upsampler(in_ch=HIDDEN, K=K, hid=HIDDEN).to(dev)
    ref = Refiner(hid=HIDDEN, mp=2).to(dev)
    opt = torch.optim.Adam(list(gnn.parameters())+list(up.parameters())+list(ref.parameters()), lr=1e-3)

    global_loss_hist=[]

    VAR_W0 = W_VAR_Y_INIT; REP_W0 = W_REPULSE_INIT; BAND_W0 = W_Y_BAND_INIT

    for epoch in range(1, EPOCHS+1):
        phase = min(1.0, epoch / 3.0)
        W_VAR_Y   = VAR_W0   * (0.6 + 0.4 * phase)
        W_REPULSE = REP_W0   * (0.5 + 0.5 * phase)
        W_Y_BAND  = BAND_W0  * (0.5 + 0.5 * phase)
        W_Y_MEAN  = W_Y_MEAN_INIT
        W_XY_COHERE = W_XY_COHERE_INIT

        for bi in range(BATCHES):
            idx = random.randrange(len(lr_files))
            dL = load_csv(lr_files[idx]); dH = load_csv(hr_files[idx])

            # type filter
            tvec = dH["type"].to_numpy() if len(dH) else dL["type"].to_numpy()
            pick = int(pd.Series(tvec).mode().iloc[0])
            dL = dL[dL["type"]==pick]; dH = dH[dH["type"]==pick]

            posL_full = dL[["x","y","z"]].to_numpy().astype(np.float32)
            velL_full = dL[["vx","vy","vz"]].to_numpy().astype(np.float32)
            posH_full = dH[["x","y","z"]].to_numpy().astype(np.float32)
            velH_full = dH[["vx","vy","vz"]].to_numpy().astype(np.float32)

            nL = min(PATCH_LR, len(posL_full))
            if nL == 0 or len(posH_full) == 0: continue
            idsL = np.random.choice(len(posL_full), size=nL, replace=False)
            idsH = np.random.choice(len(posH_full), size=min(PATCH_HR, len(posH_full)), replace=False)
            posL = posL_full[idsL]; velL = velL_full[idsL]
            posH = posH_full[idsH]; velH = velH_full[idsH]

            mean = posL.mean(0); std = posL.std(0) + 1e-6
            posL_n = (posL - mean) / std
            posH_n = (posH - mean) / std

            ei_LR, ea_LR = build_graph(posL_n, radius=0.5, max_degree=MAX_DEGREE, knn=6)
            x_LR = torch.from_numpy(np.hstack([posL_n, velL])).float().to(dev)
            ei_LR, ea_LR = ei_LR.to(dev), ea_LR.to(dev)

            with torch.cuda.amp.autocast(enabled=amp_enable):
                h    = gnn(x_LR, ei_LR, ea_LR)
                init = up(h)
                parent = torch.from_numpy(np.repeat(posL_n, K, 0)).float().to(dev)
                dpos   = init[:, :3]
                vch    = init[:, 3:6]

                for _ in range(NUM_REFINES):
                    child  = parent + dpos
                    ei, ea = build_graph(child.detach().cpu().numpy(), radius=0.35, max_degree=MAX_DEGREE, knn=6)
                    ei, ea = ei.to(dev), ea.to(dev)
                    x_child = torch.cat([child, parent, dpos, vch], 1)
                    res = ref(x_child, ei, ea)
                    dpos = dpos + res[:, :3]
                    vch  = vch  + res[:, 3:6]

                groups = assign_hr_to_lr_parents(posL_n, posH_n, K)
                tgt = []
                for pi2,g in enumerate(groups):
                    if not g:
                        tgt.append(np.zeros((K,6), np.float32)); continue
                    ids = g[:K]
                    dposH = posH_n[ids] - posL_n[pi2]
                    feat = np.hstack([dposH, velH[ids]])
                    if len(ids) < K:
                        feat = np.vstack([feat, np.zeros((K-len(ids),6), np.float32)])
                    tgt.append(feat)
                target = torch.from_numpy(np.vstack(tgt)).float().to(dev)

                child_n = parent + dpos
                cp = child_n.detach().cpu().numpy(); gh = posH_n
                if len(cp) > CHAMFER_SAMP: cp = cp[np.random.choice(len(cp), CHAMFER_SAMP, replace=False)]
                if len(gh) > CHAMFER_SAMP: gh = gh[np.random.choice(len(gh), CHAMFER_SAMP, replace=False)]
                if len(cp) and len(gh):
                    tree_gt = cKDTree(gh); d_p2g, _ = tree_gt.query(cp, k=1)
                    tree_pr = cKDTree(cp); d_g2p, _ = tree_pr.query(gh, k=1)
                    d_p2g = torch.from_numpy(d_p2g).float().to(dev)
                    d_g2p = torch.from_numpy(d_g2p).float().to(dev)
                    loss_chamfer = (d_p2g.pow(2).mean() + d_g2p.pow(2).mean())
                else:
                    loss_chamfer = torch.tensor(0.0, device=dev)

                gt_y = torch.from_numpy(posH_n[:, 1]).to(dev)
                gt_med = gt_y.median()
                gt_mad = (gt_y - gt_med).abs().median() + 1e-6
                band_lo = gt_med - Y_BAND_K * gt_mad
                band_hi = gt_med + Y_BAND_K * gt_mad

                pred_n = child_n
                pred_y = pred_n[:, 1]
                pred_x = pred_n[:, 0]
                pred_z = pred_n[:, 2]

                loss_var = W_VAR_Y * pred_y.var()

                if pred_n.shape[0] > 1:
                    m = min(REP_SUBSAMPLE, pred_n.shape[0])
                    idx = torch.randperm(pred_n.shape[0], device=dev)[:m]
                    psub = pred_n[idx]
                    d = torch.cdist(psub, psub, p=2)
                    d = d + torch.eye(m, device=dev) * 1e9
                    rep = torch.exp(-d / REP_SIGMA).mean()
                    loss_rep = W_REPULSE * rep
                else:
                    loss_rep = torch.tensor(0.0, device=dev)

                loss_y_mean = W_Y_MEAN_INIT * (pred_y.mean() - gt_med).abs()

                def huber(x, delta=0.03):
                    a = x.abs()
                    return torch.where(a < delta, 0.5 * a * a / delta, a - 0.5 * delta)
                under = huber(band_lo - pred_y)
                over  = huber(pred_y - band_hi)
                loss_y_band = W_Y_BAND_INIT * (under.mean() + over.mean())

                xy_var = pred_x.var() + pred_z.var()
                loss_xy = W_XY_COHERE_INIT * xy_var

                loss_even = W_EVEN * even_distribution_loss(child_n)

                loss = (W_POS * F.mse_loss(dpos, target[:, :3]) +
                        W_VEL * F.mse_loss(vch,  target[:, 3:6]) +
                        W_CHAMFER * loss_chamfer +
                        loss_var + loss_rep + loss_y_mean + loss_y_band + loss_xy +
                        loss_even)

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(list(gnn.parameters())+list(up.parameters())+list(ref.parameters()), 1.0)
            scaler.step(opt); scaler.update()

            if (bi % VIS_EVERY_BATCH) == 0:
                with torch.no_grad():
                    child_world = (child_n.cpu().numpy()*std + mean)
                    posL_world  = (posL_n*std + mean)
                    posH_world  = (posH_n*std + mean)
                    save_patch_visual(vis_dir, epoch, bi, posL_world, posH_world, child_world, title_suffix=f"K={K}")

        j = random.randrange(len(lr_files))
        L, H = load_csv(lr_files[j]), load_csv(hr_files[j])
        posL0 = L[["x","y","z"]].to_numpy().astype(np.float32)
        posH0 = H[["x","y","z"]].to_numpy().astype(np.float32)
        save_lr_hr_overlay(vis_dir, epoch, posL0, posH0)

        try: save_y_hist(vis_dir, epoch, child_world, posH_world)
        except Exception: pass

    ckpt_path = os.path.join(OUT_DIR, CKPT_NAME)
    torch.save({
        "gnn": gnn.state_dict(),
        "up":  up.state_dict(),
        "ref": ref.state_dict(),
        "config": {"K": K, "hidden": HIDDEN, "num_refines": NUM_REFINES, "max_degree": MAX_DEGREE, "edge_chunk": EDGE_CHUNK}
    }, ckpt_path)
    print(f"✅ Saved checkpoint: {ckpt_path}")
    print(f"🖼️ Visuals: {os.path.join(OUT_DIR, 'vis')}")

if __name__ == "__main__":
    main()
