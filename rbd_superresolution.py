# =======================================
# rbd_superresolution_full_v2_fixed.py
# Train super-resolution model for XYZ + attributes
# =======================================
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -------------------------
# CONFIG
# -------------------------
LR_DIR   = r"E:\RBD_MACHINE_LEARNING_MODEL\LR"
HR_DIR   = r"E:\RBD_MACHINE_LEARNING_MODEL\HR"
RUNS_DIR = r"E:\RBD_MACHINE_LEARNING_MODEL\Runs"
os.makedirs(RUNS_DIR, exist_ok=True)

CKPT_PATH       = os.path.join(RUNS_DIR, "superres_rbd_full_best.pt")
LABELMAP_JSON   = os.path.join(RUNS_DIR, "label_map.json")
STATS_JSON      = os.path.join(RUNS_DIR, "normalization_stats.json")

EPOCHS   = 120
LR_RATE  = 1e-3
HIDDEN   = 256
K        = 8              # upsampling factor

# Attributes to learn
CONT_ATTRS   = ["wx", "wy", "wz"]  # continuous (regressed)
CATEG_ATTRS  = ["name"]            # categorical (classified)

# Loss weights / schedule
W_CHAMFER    = 1.0
W_CENTROID   = 0.1
TARGET_W_CONT  = 1.0
TARGET_W_CATEG = 1.0
POS_WARMUP_EPOCHS = 10      # geometry-only warmup
LABEL_SMOOTHING   = 0.05
USE_HUBER         = True
CLIP_NORM         = 1.0

# Data scan
MAX_FRAMES = 999  # upper bound to scan for rbd_*_frameXXX.csv

# -------------------------
# Helpers
# -------------------------
def frame_paths(idx: int):
    return (os.path.join(LR_DIR, f"rbd_Low_frame{idx:03d}.csv"),
            os.path.join(HR_DIR, f"rbd_High_frame{idx:03d}.csv"))

def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def columns_exist(df, cols):
    return all(c in df.columns for c in cols)

def find_available_frames():
    ids = []
    for i in range(1, MAX_FRAMES+1):
        lr, hr = frame_paths(i)
        if os.path.exists(lr) and os.path.exists(hr):
            ids.append(i)
    return ids

# -------------------------
# Label map (categorical)
# -------------------------
def build_label_map(frame_ids):
    vocab = {c: set() for c in CATEG_ATTRS}
    for i in frame_ids:
        _, hr_fp = frame_paths(i)
        dH = safe_read_csv(hr_fp)
        for c in CATEG_ATTRS:
            if c in dH.columns:
                vals = dH[c].astype(str).unique().tolist()
                vocab[c].update(vals)
    label_map = {}
    for c in CATEG_ATTRS:
        sorted_vals = sorted(list(vocab[c]))
        label_map[c] = {v: j for j, v in enumerate(sorted_vals)}
    return label_map

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# -------------------------
# Normalization stats (fit on HR)
# -------------------------
def compute_norm_stats(frame_ids):
    xs, conts = [], []
    for i in frame_ids:
        _, hr_fp = frame_paths(i)
        dH = safe_read_csv(hr_fp)
        if columns_exist(dH, ["x","y","z"]):
            xs.append(dH[["x","y","z"]].to_numpy().astype(np.float32))
        if CONT_ATTRS and columns_exist(dH, CONT_ATTRS):
            conts.append(dH[CONT_ATTRS].to_numpy().astype(np.float32))
    X = np.concatenate(xs, axis=0)
    x_mean = X.mean(0).tolist()
    x_std  = (X.std(0) + 1e-8).tolist()
    if conts:
        C = np.concatenate(conts, axis=0)
        c_mean = C.mean(0).tolist()
        c_std  = (C.std(0) + 1e-8).tolist()
    else:
        c_mean, c_std = None, None
    return {"x_mean": x_mean, "x_std": x_std, "c_mean": c_mean, "c_std": c_std}

def norm(arr, mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std  = np.asarray(std, dtype=np.float32)
    return (arr - mean) / std

# -------------------------
# Dataset
# -------------------------
def load_frame(i, label_map):
    lr_fp, hr_fp = frame_paths(i)
    dL = safe_read_csv(lr_fp)
    dH = safe_read_csv(hr_fp)

    if not columns_exist(dL, ["x","y","z"]) or not columns_exist(dH, ["x","y","z"]):
        raise ValueError(f"[frame {i}] Missing x,y,z in LR/HR")

    posL = dL[["x","y","z"]].to_numpy().astype(np.float32)
    posH = dH[["x","y","z"]].to_numpy().astype(np.float32)

    contH = None
    if CONT_ATTRS and columns_exist(dH, CONT_ATTRS):
        contH = dH[CONT_ATTRS].to_numpy().astype(np.float32)

    catH = None
    if CATEG_ATTRS:
        cat_cols = []
        for c in CATEG_ATTRS:
            if c not in dH.columns:
                raise ValueError(f"[frame {i}] Missing categorical '{c}' in HR")
            lm = label_map.get(c, {})
            if not lm:
                raise ValueError(f"[frame {i}] Empty label map for '{c}'")
            vals = dH[c].astype(str).tolist()
            idxs = np.array([lm.get(v, -1) for v in vals], dtype=np.int64)
            if (idxs < 0).any():
                raise ValueError(f"[frame {i}] Unmapped labels in '{c}'")
            cat_cols.append(idxs.reshape(-1,1))
        catH = np.concatenate(cat_cols, axis=1).astype(np.int64)

    return posL, posH, contH, catH

def make_dataset(frame_ids, label_map):
    data = []
    for i in frame_ids:
        try:
            posL, posH, contH, catH = load_frame(i, label_map)
            data.append((i, posL, posH, contH, catH))
        except Exception as e:
            print(f"[WARN] Skipping frame {i}: {e}")
    return data

# -------------------------
# Model
# -------------------------
class SuperResWithAttrs(nn.Module):
    def __init__(self, in_dim=3, hid=HIDDEN, k=K, n_cont=len(CONT_ATTRS), cat_sizes=None):
        super().__init__()
        self.k = k
        self.n_cont = n_cont
        self.cat_sizes = cat_sizes or {}
        sum_cat = sum(self.cat_sizes.values()) if self.cat_sizes else 0
        out_per_child = 3 + self.n_cont + sum_cat
        out_dim = out_per_child * self.k

        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)
        )

        self._cat_split_sizes = []
        if sum_cat > 0:
            for c in CATEG_ATTRS:
                self._cat_split_sizes.append(self.cat_sizes[c])

    def forward(self, x):
        B = x.shape[0]
        out = self.net(x).view(B*self.k, -1)

        off = 0
        pos = out[:, off:off+3]; off += 3
        cont = None
        if self.n_cont > 0:
            cont = out[:, off:off+self.n_cont]; off += self.n_cont
        cat_logits_list = []
        if len(self._cat_split_sizes) > 0:
            cat_flat = out[:, off:]
            splits = torch.split(cat_flat, self._cat_split_sizes, dim=1)
            cat_logits_list = list(splits)
        return pos, cont, cat_logits_list

# -------------------------
# Losses
# -------------------------
def chamfer_distance(x, y):
    x_ = x.unsqueeze(1)
    y_ = y.unsqueeze(0)
    dist = torch.sum((x_ - y_)**2, dim=2)
    min_x, _ = torch.min(dist, dim=1)
    min_y, _ = torch.min(dist, dim=0)
    return min_x.mean() + min_y.mean(), dist

def centroid_align_loss(x, y):
    return F.mse_loss(x.mean(dim=0), y.mean(dim=0))

def nearest_indices(dist_matrix):
    return torch.argmin(dist_matrix, dim=1)

# -------------------------
# Train
# -------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame_ids = find_available_frames()
    if not frame_ids:
        raise RuntimeError("No LR/HR frame pairs found.")
    print(f"[INFO] Found {len(frame_ids)} frames.")

    label_map = load_json(LABELMAP_JSON)
    if label_map is None and len(CATEG_ATTRS) > 0:
        print("[INFO] Building label map...")
        label_map = build_label_map(frame_ids)
        save_json(LABELMAP_JSON, label_map)
        print(f"[✓] Saved label map -> {LABELMAP_JSON}")
    elif label_map is not None:
        print("[INFO] Loaded existing label map.")

    cat_sizes = {c: len(label_map.get(c, {})) for c in CATEG_ATTRS} if label_map else {}

    stats = load_json(STATS_JSON)
    if stats is None:
        print("[INFO] Computing normalization stats...")
        stats = compute_norm_stats(frame_ids)
        save_json(STATS_JSON, stats)
        print(f"[✓] Saved stats -> {STATS_JSON}")
    x_mean = np.array(stats["x_mean"], dtype=np.float32)
    x_std  = np.array(stats["x_std"],  dtype=np.float32)
    c_mean = np.array(stats["c_mean"], dtype=np.float32) if stats["c_mean"] is not None else None
    c_std  = np.array(stats["c_std"],  dtype=np.float32) if stats["c_std"] is not None else None

    dataset = make_dataset(frame_ids, label_map)
    print(f"[INFO] Loaded {len(dataset)} valid frame pairs.")

    model = SuperResWithAttrs(k=K, n_cont=len(CONT_ATTRS), cat_sizes=cat_sizes).to(device)
    opt   = optim.Adam(model.parameters(), lr=LR_RATE)

    best = float("inf")
    for epoch in range(1, EPOCHS+1):
        if epoch <= POS_WARMUP_EPOCHS:
            W_CONT_eff, W_CATEG_eff = 0.0, 0.0
        else:
            t = (epoch - POS_WARMUP_EPOCHS) / max(1, (EPOCHS - POS_WARMUP_EPOCHS))
            W_CONT_eff  = TARGET_W_CONT  * t
            W_CATEG_eff = TARGET_W_CATEG * t

        sum_total, sum_ch, sum_cent, sum_cont, sum_cat = 0.0, 0.0, 0.0, 0.0, 0.0
        n_batches = 0

        for (fid, posL, posH, contH, catH) in dataset:
            x_np     = norm(posL, x_mean, x_std)
            y_pos_np = norm(posH, x_mean, x_std)
            x     = torch.from_numpy(x_np).to(device)
            y_pos = torch.from_numpy(y_pos_np).to(device)

            y_cont = None
            if contH is not None and c_mean is not None:
                y_cont_np = norm(contH, c_mean, c_std)
                y_cont = torch.from_numpy(y_cont_np).to(device)

            y_cat_list = []
            if catH is not None:
                for j, _c in enumerate(CATEG_ATTRS):
                    y_cat_list.append(torch.from_numpy(catH[:, j]).to(device))

            pred_pos, pred_cont, pred_cat_logits_list = model(x)

            chamfer, D = chamfer_distance(pred_pos, y_pos)
            loss = W_CHAMFER * chamfer
            cent = centroid_align_loss(pred_pos, y_pos) if W_CENTROID > 0 else torch.tensor(0., device=device)
            loss = loss + W_CENTROID * cent

            idx_nn = nearest_indices(D)

            c_loss = torch.tensor(0., device=device)
            if pred_cont is not None and y_cont is not None and W_CONT_eff > 0:
                y_cont_g = y_cont[idx_nn]
                if USE_HUBER:
                    c_loss = F.smooth_l1_loss(pred_cont, y_cont_g)
                else:
                    c_loss = F.mse_loss(pred_cont, y_cont_g)
                loss = loss + W_CONT_eff * c_loss

            k_loss = torch.tensor(0., device=device)
            if len(pred_cat_logits_list) > 0 and len(y_cat_list) > 0 and W_CATEG_eff > 0:
                for logits, y_cat in zip(pred_cat_logits_list, y_cat_list):
                    y_g = y_cat[idx_nn]
                    k_loss = k_loss + F.cross_entropy(logits, y_g, label_smoothing=LABEL_SMOOTHING)
                loss = loss + W_CATEG_eff * k_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()

            sum_total += float(loss.item())
            sum_ch    += float(chamfer.item())
            sum_cent  += float(cent.item())
            sum_cont  += float(c_loss.item())
            sum_cat   += float(k_loss.item())
            n_batches += 1

        avg_total = sum_total / n_batches
        avg_ch    = sum_ch / n_batches
        avg_cent  = sum_cent / n_batches
        avg_cont  = sum_cont / max(1, n_batches)
        avg_cat   = sum_cat / max(1, n_batches)

        print(f"[Epoch {epoch}] total={avg_total:.4f} | chamfer={avg_ch:.4f} "
              f"| centroid={avg_cent:.4f} | cont={avg_cont:.4f} (x{W_CONT_eff:.2f}) "
              f"| cat={avg_cat:.4f} (x{W_CATEG_eff:.2f})")

        if avg_total < best:
            best = avg_total
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "K": K, "HIDDEN": HIDDEN,
                    "CONT_ATTRS": CONT_ATTRS,
                    "CATEG_ATTRS": CATEG_ATTRS,
                    "cat_sizes": cat_sizes,
                    "W_CHAMFER": W_CHAMFER, "W_CENTROID": W_CENTROID,
                    "TARGET_W_CONT": TARGET_W_CONT, "TARGET_W_CATEG": TARGET_W_CATEG,
                    "POS_WARMUP_EPOCHS": POS_WARMUP_EPOCHS,
                    "LABEL_SMOOTHING": LABEL_SMOOTHING,
                    "USE_HUBER": USE_HUBER,
                    "CLIP_NORM": CLIP_NORM,
                },
                "stats": stats,
                "label_map": label_map
            }, CKPT_PATH)
            print(f"  [✓] Saved checkpoint {CKPT_PATH}")

    print("[✔] Training complete.")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    train()
