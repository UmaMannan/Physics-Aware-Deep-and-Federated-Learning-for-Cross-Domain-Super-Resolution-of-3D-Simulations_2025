# predict_superres_aligned_declump_batch.py
# ------------------------------------------------------------
# Batch version: Predict HR particles for ALL frames in LR_DIR.
# Saves one CSV per frame + full set of visuals/metrics.
# ------------------------------------------------------------

import os, argparse, glob, json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# =========================
# PATH DEFAULTS
# =========================
CKPT_PATH   = r"E:/25 Aug 2025_project/checkpoints_unified/fast_superres_unified.pt"
LR_DIR      = r"E:/25 Aug 2025_project/converted_particles_Low"
HR_DIR      = r"E:/25 Aug 2025_project/converted_particles_High"  # optional for GT
OUT_DIR     = r"E:/25 Aug 2025_project/predict_vis_unified"

DEVICE        = "cuda"
PATCH_LR      = 4000
KNN_FALLBACK  = 6
RADIUS_LR     = 0.5
RADIUS_HR     = 0.35
MAX_DEGREE    = 12
EDGE_CHUNK    = 50_000
VIS_POINT_LIMIT = 120_000

DECLUMP_STEPS         = 8
DECLUMP_STEP_SIZE     = 0.12
DECLUMP_MIN_SEP       = 0.10
DECLUMP_POISSON_R     = 0.00
DECLUMP_KEEP_CENTROID = True
PLOT_LR_MATCH         = True

# =========================
# MODELS (same as before)
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
        h = self.enc(x)
        if ei.numel() == 0: return h
        src, dst = ei
        for _ in range(self.mp):
            agg = torch.zeros_like(h)
            E = src.numel()
            for s in range(0, E, EDGE_CHUNK):
                e = min(E, s+EDGE_CHUNK)
                m = self.edge(torch.cat([h[src[s:e]], ea[s:e]], dim=1))
                agg.index_add_(0, dst[s:e], m)
            h = h + self.upd(torch.cat([h, agg], dim=1))
        return h

class Upsampler(nn.Module):
    def __init__(self, in_ch, K, hid=64):
        super().__init__()
        self.K = int(K)
        self.mlp = MLP(in_ch, 6*K, hid)
    def forward(self, h): return self.mlp(h).view(h.shape[0]*self.K, 6)

class Refiner(GNN):
    def __init__(self, hid=64, mp=2):
        super().__init__(in_ch=12, edge_in=4, hid=hid, mp=mp)
        self.head = MLP(hid, 6, hid)
    def forward(self, x, ei, ea): return self.head(super().forward(x, ei, ea))

# =========================
# HELPERS (load/save/graph) – same as before
# =========================
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def load_csv(path):
    df = pd.read_csv(path)
    need = {"x","y","z","vx","vy","vz","type"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"{path} missing columns {sorted(list(need - set(df.columns)))}")
    return df

def thin(arr, max_n):
    if len(arr) <= max_n: return arr
    idx = np.random.choice(len(arr), size=max_n, replace=False)
    return arr[idx]

def build_graph(pos, radius=0.5, max_degree=12, knn=6):
    if len(pos)==0:
        return torch.zeros((2,0), dtype=torch.long), torch.zeros((0,4))
    tree = cKDTree(pos)
    edges=[]
    for i in range(len(pos)):
        nb = tree.query_ball_point(pos[i], r=radius); nb = [j for j in nb if j!=i]
        if len(nb) > max_degree:
            cand = np.asarray(nb)
            d = np.linalg.norm(pos[cand]-pos[i], axis=1)
            nb = cand[np.argsort(d)[:max_degree]]
        for j in nb: edges.append((i,j))
    if len(pos)>1:
        _, nbrs = tree.query(pos, k=min(knn+1,len(pos)))
        for i,row in enumerate(nbrs[:,1:]):
            for j in row: edges.append((i,j))
    if not edges:
        return torch.zeros((2,0), dtype=torch.long), torch.zeros((0,4))
    edges = np.unique(np.array(edges), axis=0)
    src, dst = edges[:,0], edges[:,1]
    rel = pos[dst] - pos[src]; dist = np.linalg.norm(rel,axis=1,keepdims=True)
    ea = np.hstack([rel,dist]).astype(np.float32)
    return torch.from_numpy(edges.T).long(), torch.from_numpy(ea).float()

# =========================
# VISUALIZATIONS (keep only key ones to save time)
# =========================
def save_overlay_xy(out_dir, posL, posPred, posGT=None, frame_id=""):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(thin(posPred, VIS_POINT_LIMIT)[:,0], thin(posPred,VIS_POINT_LIMIT)[:,1],
               s=0.5, c="tab:red", alpha=0.4, label="HR (Pred)")
    ax.scatter(thin(posL, min(10000, len(posL)))[:,0],
               thin(posL, min(10000, len(posL)))[:,1],
               s=2.2, c="tab:blue", alpha=0.8, label="LR")
    if posGT is not None:
        ax.scatter(thin(posGT,VIS_POINT_LIMIT)[:,0], thin(posGT,VIS_POINT_LIMIT)[:,1],
                   s=0.5, c="tab:green", alpha=0.35, label="HR (GT)")
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, ls="--", alpha=0.3); ax.legend()
    p = os.path.join(out_dir, f"overlay_xy_frame{frame_id}.png")
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)

# =========================
# CORE PREDICTION FOR ONE FRAME
# =========================
def predict_one_frame(gnn, up, ref, K, num_refines, lr_csv, out_csv, dev, gt_csv=None):
    dL = load_csv(lr_csv)
    posL = dL[["x","y","z"]].to_numpy().astype(np.float32)
    velL = dL[["vx","vy","vz"]].to_numpy().astype(np.float32)
    typ  = int(dL["type"].iloc[0]) if len(dL) else 0

    posGT=None
    if gt_csv and os.path.isfile(gt_csv):
        posGT = load_csv(gt_csv)[["x","y","z"]].to_numpy().astype(np.float32)

    N=len(posL); idxs=np.arange(N)
    patches=[idxs[s:s+PATCH_LR] for s in range(0,N,PATCH_LR)]
    preds_pos=[]; preds_vel=[]

    for ids in patches:
        pL=posL[ids]; vL=velL[ids]
        mean=pL.mean(0); std=pL.std(0)+1e-6
        pL_n=(pL-mean)/std
        ei_LR, ea_LR=build_graph(pL_n,RADIUS_LR,MAX_DEGREE,KNN_FALLBACK)
        x_LR=torch.from_numpy(np.hstack([pL_n,vL])).float().to(dev)
        ei_LR, ea_LR=ei_LR.to(dev), ea_LR.to(dev)
        with torch.no_grad():
            h=gnn(x_LR,ei_LR,ea_LR)
            init=up(h)
            parent=torch.from_numpy(np.repeat(pL_n,K,0)).float().to(dev)
            dpos=init[:,:3]; vch=init[:,3:6]
            for _ in range(num_refines):
                child=parent+dpos
                ei,ea=build_graph(child.detach().cpu().numpy(),RADIUS_HR,MAX_DEGREE,KNN_FALLBACK)
                ei,ea=ei.to(dev),ea.to(dev)
                x_child=torch.cat([child,parent,dpos,vch],1)
                res=ref(x_child,ei,ea)
                dpos+=res[:,:3]; vch+=res[:,3:6]
            child_world=(parent+dpos).cpu().numpy()*std+mean
            preds_pos.append(child_world); preds_vel.append(vch.cpu().numpy())
    pred_pos=np.vstack(preds_pos); pred_vel=np.vstack(preds_vel)

    # save predicted HR CSV
    pd.DataFrame({
        "x":pred_pos[:,0],"y":pred_pos[:,1],"z":pred_pos[:,2],
        "vx":pred_vel[:,0],"vy":pred_vel[:,1],"vz":pred_vel[:,2],
        "type":np.full((pred_pos.shape[0],),typ,dtype=np.int64)
    }).to_csv(out_csv,index=False)

    # save visualization
    vis_dir=os.path.join(OUT_DIR,"vis"); ensure_dir(vis_dir)
    frame_id=os.path.splitext(os.path.basename(lr_csv))[0].split("_")[-1]
    save_overlay_xy(vis_dir,posL,pred_pos,posGT,frame_id)

    print(f"✅ Frame {frame_id} done → {out_csv}")

# =========================
# MAIN
# =========================
def main():
    dev = DEVICE if (DEVICE=="cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {dev}")

    # load model
    ck = torch.load(CKPT_PATH, map_location="cpu")
    cfg=ck.get("config",{})
    K=int(cfg.get("K",8)); hidden=int(cfg.get("hidden",64)); num_ref=int(cfg.get("num_refines",2))
    print(f"Loaded model K={K}, hidden={hidden}, num_refines={num_ref}")

    gnn=GNN(6,4,hidden,2).to(dev); up=Upsampler(hidden,K,hidden).to(dev); ref=Refiner(hidden,2).to(dev)
    gnn.load_state_dict(ck["gnn"]); up.load_state_dict(ck["up"]); ref.load_state_dict(ck["ref"])
    gnn.eval(); up.eval(); ref.eval()

    ensure_dir(OUT_DIR)

    # loop over all LR frames
    lr_files=sorted(glob.glob(os.path.join(LR_DIR,"*.csv")))
    for lr_csv in lr_files:
        frame_id=os.path.splitext(os.path.basename(lr_csv))[0].split("_")[-1]
        out_csv=os.path.join(OUT_DIR,f"predicted_hr_frame_{frame_id}.csv")
        gt_csv=os.path.join(HR_DIR,f"particles_High_frame{frame_id}.csv")
        predict_one_frame(gnn,up,ref,K,num_ref,lr_csv,out_csv,dev,gt_csv)

if __name__=="__main__":
    main()
