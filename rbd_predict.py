# =======================================
# rbd_predict.py - Predict HR + Comparisons
# =======================================
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# CONFIG
# -------------------------
LR_DIR   = r"E:\RBD_MACHINE_LEARNING_MODEL\LR"
HR_DIR   = r"E:\RBD_MACHINE_LEARNING_MODEL\HR"   # optional, for comparison
PRED_DIR = r"E:\RBD_MACHINE_LEARNING_MODEL\Preds"
FIG_DIR  = r"E:\RBD_MACHINE_LEARNING_MODEL\Figures"

os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Use the FULL checkpoint that includes positions + attributes
CKPT     = r"E:\RBD_MACHINE_LEARNING_MODEL\Runs\superres_rbd_full_best.pt"
K        = 8
HIDDEN   = 256  # must match training

# -------------------------
# Model (same structure as training)
# -------------------------
class SuperResWithAttrs(nn.Module):
    """
    Emits per-child outputs: [x,y,z] + continuous attrs + concatenated categorical logits.
    cat_sizes: dict like {'name': num_classes, ...}
    """
    def __init__(self, in_dim=3, hid=HIDDEN, k=K, n_cont=0, cat_sizes=None):
        super().__init__()
        self.k = k
        self.n_cont = n_cont
        self.cat_sizes = cat_sizes or {}
        self.sum_cat = sum(self.cat_sizes.values()) if self.cat_sizes else 0

        out_per_child = 3 + self.n_cont + self.sum_cat
        out_dim = out_per_child * self.k

        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_dim)
        )

        # for splitting categorical logits back into per-attr chunks
        self._cat_split_sizes = []
        if self.sum_cat > 0:
            # NOTE: prediction-time order must match training's CATEG_ATTRS order
            # We'll read that from ckpt['config']['CATEG_ATTRS']
            pass

    def set_cat_split_sizes(self, sizes_list):
        self._cat_split_sizes = sizes_list

    def forward(self, x):
        """
        x: (B, 3)
        returns:
          pos_pred:   (B*K, 3)
          cont_pred:  (B*K, n_cont) or None
          cat_logits: list of tensors, each (B*K, num_classes_c) or empty list
        """
        B = x.shape[0]
        out = self.net(x).view(B*self.k, -1)

        off = 0
        pos_pred = out[:, off:off+3]; off += 3

        cont_pred = None
        if self.n_cont > 0:
            cont_pred = out[:, off:off+self.n_cont]; off += self.n_cont

        cat_logits_list = []
        if len(self._cat_split_sizes) > 0:
            cat_flat = out[:, off:]
            splits = torch.split(cat_flat, self._cat_split_sizes, dim=1)
            cat_logits_list = list(splits)

        return pos_pred, cont_pred, cat_logits_list

# -------------------------
# Visualization helpers
# -------------------------
def scatter_overlay(lr, hr, pred, out_path, title="3D Comparison"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if lr is not None: ax.scatter(lr[:,0], lr[:,1], lr[:,2], s=5, c="blue",  alpha=0.4, label="LR")
    if hr is not None: ax.scatter(hr[:,0], hr[:,1], hr[:,2], s=5, c="green", alpha=0.4, label="HR")
    if pred is not None: ax.scatter(pred[:,0], pred[:,1], pred[:,2], s=5, c="red",   alpha=0.6, label="Pred")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def density_side_by_side(lr, hr, pred, out_path, title="Density Comparison"):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    datasets = [("LR", lr), ("HR", hr), ("Pred", pred)]
    planes = [("XY", (0,1)), ("XZ", (0,2)), ("YZ", (1,2))]

    for i, (name, data) in enumerate(datasets):
        for j, (pname, (a,b)) in enumerate(planes):
            ax = axes[i, j]
            if data is not None and len(data) > 0:
                sns.kdeplot(x=data[:,a], y=data[:,b], fill=True, ax=ax, cmap="viridis")
            ax.set_title(f"{name} {pname}")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# -------------------------
# Small helpers
# -------------------------
def norm(arr, mean, std):
    mean = np.asarray(mean, dtype=np.float32)
    std  = np.asarray(std, dtype=np.float32)
    return (arr - mean) / std

# -------------------------
# Predict + Compare
# -------------------------
@torch.no_grad()
def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(CKPT):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT}")

    # Load checkpoint with config, stats, label map
    ckpt = torch.load(CKPT, map_location=device)
    cfg  = ckpt["config"]
    stats = ckpt["stats"]
    label_map = ckpt.get("label_map", {}) or {}

    # Build inverse label map for decoding categories
    inv_label_map = {c: {v: k for k, v in mapping.items()} for c, mapping in label_map.items()}

    # Model (match training config)
    cat_sizes = cfg.get("cat_sizes", {})
    model = SuperResWithAttrs(
        in_dim=3, hid=cfg.get("HIDDEN", HIDDEN),
        k=cfg.get("K", K),
        n_cont=len(cfg.get("CONT_ATTRS", [])),
        cat_sizes=cat_sizes
    ).to(device)

    # Order of categorical attrs must match training's list
    cat_attrs = cfg.get("CATEG_ATTRS", [])
    if cat_attrs and cat_sizes:
        model.set_cat_split_sizes([cat_sizes[c] for c in cat_attrs])

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Normalization stats
    x_mean = np.array(stats["x_mean"], dtype=np.float32)
    x_std  = np.array(stats["x_std"],  dtype=np.float32)
    c_mean = np.array(stats["c_mean"], dtype=np.float32) if stats["c_mean"] is not None else None
    c_std  = np.array(stats["c_std"],  dtype=np.float32) if stats["c_std"] is not None else None

    frames = sorted([f for f in os.listdir(LR_DIR) if f.endswith(".csv")])
    for f in frames:
        base = os.path.splitext(f)[0]  # e.g., rbd_Low_frame007
        path = os.path.join(LR_DIR, f)
        dL = pd.read_csv(path)

        if not {"x","y","z"}.issubset(dL.columns):
            print(f"[WARN] Skipping {f}: missing x,y,z")
            continue

        # LR positions -> normalize -> predict
        posL = dL[["x","y","z"]].to_numpy().astype(np.float32)
        x_in = torch.from_numpy(norm(posL, x_mean, x_std)).to(device)

        pred_pos, pred_cont, pred_cat_logits_list = model(x_in)

        # Denormalize positions & continuous attrs
        pred_pos = pred_pos.cpu().numpy() * x_std + x_mean
        if pred_cont is not None and c_mean is not None:
            pred_cont = pred_cont.cpu().numpy() * c_std + c_mean
        else:
            pred_cont = None

        # Decode categorical predictions into strings
        pred_cats = None
        if len(pred_cat_logits_list) > 0 and len(cat_attrs) > 0:
            decoded_cols = []
            for logits, col in zip(pred_cat_logits_list, cat_attrs):
                idx = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()
                decoded = [inv_label_map.get(col, {}).get(int(i), str(int(i))) for i in idx]
                decoded_cols.append(decoded)
            pred_cats = np.array(decoded_cols, dtype=object).T  # (N, num_cat_cols)

        # Save prediction CSV with all learned attrs
        df_out = {
            "x": pred_pos[:,0], "y": pred_pos[:,1], "z": pred_pos[:,2]
        }
        # Continuous attributes (preserve original names/order from training config)
        cont_attrs = cfg.get("CONT_ATTRS", [])
        if pred_cont is not None and len(cont_attrs) == pred_cont.shape[1]:
            for j, name in enumerate(cont_attrs):
                df_out[name] = pred_cont[:, j]
        # Categorical attributes (decoded)
        if pred_cats is not None:
            for j, name in enumerate(cat_attrs):
                df_out[name] = pred_cats[:, j]

        out_csv = os.path.join(PRED_DIR, f.replace("Low", "Pred"))
        pd.DataFrame(df_out).to_csv(out_csv, index=False)
        print(f"[✓] Wrote {out_csv}")

        # Optional HR load (for visuals)
        hr = None
        hr_file = os.path.join(HR_DIR, f.replace("Low", "High"))
        if os.path.exists(hr_file):
            dH = pd.read_csv(hr_file)
            if {"x","y","z"}.issubset(dH.columns):
                hr = dH[["x","y","z"]].to_numpy().astype(np.float32)

        # generate visuals (position-only)
        scatter_overlay(posL, hr, pred_pos, os.path.join(FIG_DIR, f"{base}_scatter_overlay.png"), f"{base} 3D Overlay")
        density_side_by_side(posL, hr, pred_pos, os.path.join(FIG_DIR, f"{base}_density_compare.png"), f"{base} Density Compare")

    print("[✔] Prediction + Comparison Visuals complete.")

if __name__ == "__main__":
    predict()
