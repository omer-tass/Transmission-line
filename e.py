import os
import json
import numpy as np
import matplotlib.pyplot as plt
import numpy.lib.scimath as sm

import skrf as rf

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise ImportError(
        "PyTorch is required for the VAE part. Install it with: pip install torch"
    ) from e


# =========================
# 0) CONFIG
# =========================

LINE_LENGTH = 4e-4          # meters (update if needed)
Z0_SYSTEM = 50
K_PAD = 0.4
SEED = 42

WINDOW = 32
STRIDE = 1

LATENT_DIM = 8
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-3
BETA_KL = 1.0

# Prof split request
F_TRAIN_MAX = 67e9
F_TEST_MIN  = 67e9
F_TEST_MAX  = 110e9


# =========================
# 1) Helpers (RF / pads)
# =========================

def enforce_reciprocal_symmetric(network: rf.Network) -> rf.Network:
    s = network.s.copy()
    n_freq, n_ports, _ = s.shape
    for i in range(n_freq):
        s[i] = (s[i] + s[i].T) / 2
        avg_diag = np.mean(np.diag(s[i]))
        np.fill_diagonal(s[i], avg_diag)
    return rf.Network(f=network.f, s=s, z0=network.z0)

def extract_pad_discriminant(net: rf.Network, k: float = 0.4):
    """
    Defter-based:
      a = 1
      b = 2*z21*(k-1)
      c = -2*k*z21*(2*z21 + (z11 - z21))
    """
    Z = net.z
    Z11 = Z[:, 0, 0]
    Z21 = Z[:, 1, 0]
    Z1, Z2, Z3 = [], [], []

    Z2_val_prev = None
    for i, (z11, z21) in enumerate(zip(Z11, Z21)):
        a = 1.0
        b = 2.0 * z21 * (k - 1.0)
        c = -2.0 * k * z21 * (2.0 * z21 + (z11 - z21))

        disc = b**2 - 4*a*c
        sqrt_disc = sm.sqrt(disc)

        Z2_pos = (-b + sqrt_disc) / (2*a)
        Z2_neg = (-b - sqrt_disc) / (2*a)

        # root continuity (avoid hopping)
        if Z2_val_prev is None:
            Z2_val = Z2_pos if np.imag(Z2_pos) > np.imag(Z2_neg) else Z2_neg
        else:
            dist_pos = np.abs(Z2_pos - Z2_val_prev)
            dist_neg = np.abs(Z2_neg - Z2_val_prev)
            Z2_val = Z2_pos if dist_pos < dist_neg else Z2_neg

        Z2_val_prev = Z2_val

        Z3_val = (Z2_val**2 - 2*z21*Z2_val) / (2*z21)
        Z1_val = k * Z3_val

        Z1.append(Z1_val)
        Z2.append(Z2_val)
        Z3.append(Z3_val)

    return net.f, np.array(Z1), np.array(Z2), np.array(Z3)

def tee_network(f, Z1, Z2, Z3):
    frequency = rf.Frequency.from_f(f, unit='Hz')
    n = len(f)

    Z11 = Z1 + Z2
    Z12 = Z2
    Z21 = Z2
    Z22 = Z2 + Z3

    z = np.zeros((2, 2, n), dtype=complex)
    z[0, 0, :] = Z11
    z[0, 1, :] = Z12
    z[1, 0, :] = Z21
    z[1, 1, :] = Z22

    z = np.moveaxis(z, 2, 0)  # (n_freq, 2, 2)
    s = rf.z2s(z, z0=Z0_SYSTEM)
    return rf.Network(frequency=frequency, s=s, z0=Z0_SYSTEM)

def extract_rlgc(network: rf.Network, line_length: float):
    """
    Extract per-unit-length RLGC from de-embedded line network.
    """
    abcd = network.a
    A = abcd[:, 0, 0]
    B = abcd[:, 0, 1]
    C = abcd[:, 1, 0]
    D = abcd[:, 1, 1]

    gamma = (1.0 / line_length) * np.arccosh((A + D) / 2.0)
    Z0 = np.sqrt(B / C)

    Z = gamma * Z0
    Y = gamma / Z0

    freq = network.f
    omega = 2 * np.pi * freq

    R = np.real(Z)
    L = np.imag(Z) / omega
    G = np.real(Y)
    Cc = np.imag(Y) / omega

    return freq, R, L, G, Cc


# =========================
# 2) Preprocess + Scaling (no sklearn)
# =========================

def clean_array(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float64)
    x[~np.isfinite(x)] = np.nan
    if np.all(np.isnan(x)):
        return np.zeros_like(x)

    # forward fill
    for i in range(1, len(x)):
        if np.isnan(x[i]) and not np.isnan(x[i-1]):
            x[i] = x[i-1]
    # back fill
    for i in range(len(x)-2, -1, -1):
        if np.isnan(x[i]) and not np.isnan(x[i+1]):
            x[i] = x[i+1]

    x[np.isnan(x)] = 0.0
    return x

class SimpleMinMaxScaler:
    def __init__(self, eps=1e-12):
        self.eps = eps
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, X: np.ndarray):
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = np.maximum(self.data_max_ - self.data_min_, self.eps)
        return self

    def transform(self, X: np.ndarray):
        return (X - self.data_min_) / self.data_range_

    def inverse_transform(self, Xs: np.ndarray):
        return Xs * self.data_range_ + self.data_min_

def build_ml_features(freq, R, L, G, Cc, fit_mask_points: np.ndarray):
    """
    Features = [log1p(R), L, log1p(G), C]
    Fit scaler ONLY on train-frequency points to avoid leakage.
    """
    freq = np.array(freq, dtype=np.float64)
    R = clean_array(R)
    L = clean_array(L)
    G = clean_array(G)
    Cc = clean_array(Cc)

    R_t = np.log1p(np.maximum(R, 0.0))
    G_t = np.log1p(np.maximum(G, 0.0))

    X = np.stack([R_t, L, G_t, Cc], axis=1)  # (N,4)

    scaler = SimpleMinMaxScaler().fit(X[fit_mask_points])
    X_scaled = scaler.transform(X)

    meta = {
        "features": ["log1p(R)", "L", "log1p(G)", "C"],
        "freq_hz_min": float(freq.min()),
        "freq_hz_max": float(freq.max()),
        "n_points": int(len(freq)),
        "window": int(WINDOW),
        "stride": int(STRIDE),
        "train_max_hz": float(F_TRAIN_MAX),
        "test_min_hz": float(F_TEST_MIN),
        "test_max_hz": float(F_TEST_MAX),
        "scaler_fit_note": "Scaler fitted only on freq <= 67 GHz points (no leakage).",
    }
    return X, X_scaled, scaler, meta

def make_windows(X_scaled, window=32, stride=1):
    N = X_scaled.shape[0]
    samples = []
    starts = []
    for start in range(0, N - window + 1, stride):
        samples.append(X_scaled[start:start+window, :])
        starts.append(start)
    return np.stack(samples, axis=0), np.array(starts, dtype=int)  # windows (M,W,4), start_idx (M,)


# =========================
# 3) VAE
# =========================

class WindowDataset(Dataset):
    def __init__(self, X_flat: np.ndarray):
        self.x = torch.tensor(X_flat, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx]

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # inputs are scaled ~ [0,1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar

def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    recon = torch.mean((x - x_hat) ** 2)  # MSE
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.detach(), kl.detach()


# =========================
# 4) Post-processing: RLCG -> alpha/beta/Z0
# =========================

def inv_feature_transform(X_scaled: np.ndarray, scaler: SimpleMinMaxScaler):
    """
    X_scaled: (N,4) where columns are [log1p(R), L, log1p(G), C] scaled.
    Return R,L,G,C in physical domain.
    """
    X = scaler.inverse_transform(X_scaled)
    logR, L, logG, Cc = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    R = np.expm1(logR)
    G = np.expm1(logG)
    R = np.maximum(R, 0.0)
    G = np.maximum(G, 0.0)
    return R, L, G, Cc

def rlcg_to_gamma_z0(freq, R, L, G, Cc):
    omega = 2 * np.pi * freq
    Zs = R + 1j * omega * L
    Ys = G + 1j * omega * Cc
    gamma = sm.sqrt(Zs * Ys)           # alpha + j beta
    Z0 = sm.sqrt(Zs / Ys)              # characteristic impedance
    alpha = np.real(gamma)
    beta  = np.imag(gamma)
    return alpha, beta, np.real(Z0), np.imag(Z0)

def reconstruct_sequence_from_windows(windows_scaled_pred, starts, N, window):
    """
    Overlap-average reconstruction back to point-wise (N,4).
    windows_scaled_pred: (M_sel, W, 4)
    starts: (M_sel,)
    """
    sum_arr = np.zeros((N, 4), dtype=np.float64)
    cnt_arr = np.zeros((N, 1), dtype=np.float64)

    for w, st in zip(windows_scaled_pred, starts):
        sum_arr[st:st+window, :] += w
        cnt_arr[st:st+window, 0] += 1.0

    out = np.full((N, 4), np.nan, dtype=np.float64)
    mask = cnt_arr[:, 0] > 0
    out[mask, :] = sum_arr[mask, :] / cnt_arr[mask, :]
    return out, mask


# 5) MAIN


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ---- Load S2P files ----
    L_line = rf.Network("2013_400_1.s2p")
    L2_line = rf.Network("2013_800_1.s2p")

    # ---- L-2L style pad extraction ----
    pads = L_line ** L2_line.inv ** L_line
    rs_pads = enforce_reciprocal_symmetric(pads)

    # ---- Extract pad Z1,Z2,Z3 ----
    f_pad, Z1, Z2, Z3 = extract_pad_discriminant(rs_pads, k=K_PAD)

    # ---- Build pad networks ----
    left_pad = tee_network(f_pad, Z1, Z2, Z3)
    right_pad = tee_network(f_pad, Z3, Z2, Z1)

    # ---- De-embed intrinsic line ----
    L_line_isolated = left_pad.inv ** L_line ** right_pad.inv

    # ---- Extract RLGC (true/reference) ----
    freq, R, L, G, Cc = extract_rlgc(L_line_isolated, LINE_LENGTH)

    print(f"[INFO] Frequency range: {freq.min()/1e9:.3f} GHz .. {freq.max()/1e9:.3f} GHz (N={len(freq)})")

    # ---- Build features & scaler (fit only on <=67 GHz points) ----
    fit_mask_points = freq <= F_TRAIN_MAX
    X_raw, X_scaled, scaler, meta = build_ml_features(freq, R, L, G, Cc, fit_mask_points)

    # ---- Create sliding windows ----
    windows, starts = make_windows(X_scaled, window=WINDOW, stride=STRIDE)
    M, W, Fdim = windows.shape
    print(f"[OK] Windows created: {windows.shape}")

    # Save dataset + scaler
    np.savez("dataset_rlgc.npz", freq=freq, X_scaled=X_scaled, windows=windows, starts=starts)
    with open("scaler_meta.json", "w", encoding="utf-8") as fmeta:
        json.dump(meta, fmeta, indent=2)

    scaler_pack = {
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
        "data_range_": scaler.data_range_.tolist(),
        "feature_names": meta["features"],
    }
    with open("scaler_params.json", "w", encoding="utf-8") as fs:
        json.dump(scaler_pack, fs, indent=2)

    # =========================
    # TRAIN/TEST SPLIT BY FREQUENCY 
    # =========================
    # window i covers [starts[i] ... starts[i]+W-1]
    end_idx = starts + (W - 1)
    start_freq = freq[starts]
    end_freq = freq[end_idx]

    train_mask_win = (end_freq <= F_TRAIN_MAX)
    test_mask_win  = (start_freq >= F_TEST_MIN) & (end_freq <= F_TEST_MAX)

    train_idx = np.where(train_mask_win)[0]
    test_idx  = np.where(test_mask_win)[0]

    print(f"[INFO] Train windows (end<=67GHz): {len(train_idx)}")
    print(f"[INFO] Test  windows (start>=67GHz, end<=110GHz): {len(test_idx)}")

    if len(train_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError("Train/Test split produced empty set. Check frequency range or WINDOW size.")

    # Flatten windows for MLP VAE
    X_flat = windows.reshape(M, W * Fdim)

    Xtr = X_flat[train_idx]
    Xte = X_flat[test_idx]

    train_ds = WindowDataset(Xtr)
    test_ds  = WindowDataset(Xte)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Train VAE ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = VAE(input_dim=W*Fdim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    opt = optim.Adam(vae.parameters(), lr=LR)

    train_losses = []
    test_losses = []

    for epoch in range(1, EPOCHS + 1):
        vae.train()
        tl = 0.0
        for xb in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            xhat, mu, logvar = vae(xb)
            loss, recon, kl = vae_loss(xb, xhat, mu, logvar, beta=BETA_KL)
            loss.backward()
            opt.step()
            tl += loss.item() * xb.size(0)
        tl /= len(train_ds)
        train_losses.append(tl)

        vae.eval()
        vl = 0.0
        with torch.no_grad():
            for xb in test_loader:
                xb = xb.to(device)
                xhat, mu, logvar = vae(xb)
                loss, _, _ = vae_loss(xb, xhat, mu, logvar, beta=BETA_KL)
                vl += loss.item() * xb.size(0)
        vl /= len(test_ds)
        test_losses.append(vl)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{EPOCHS} | train={tl:.6f} | test={vl:.6f}")

    torch.save({
        "model_state": vae.state_dict(),
        "window": WINDOW,
        "stride": STRIDE,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "feature_dim": Fdim,
        "train_rule": "end_freq<=67GHz",
        "test_rule": "start_freq>=67GHz and end_freq<=110GHz",
    }, "vae_rlgc.pt")
    print("[OK] Saved model: vae_rlgc.pt")

    # ---- Plot loss ----
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="train")
    plt.plot(test_losses, label="test")
    plt.title("VAE Training Loss (freq-based split)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

    # =========================
    # PREDICT alpha, beta, Zreal, Zimag (Prof requirement)
    # =========================
    vae.eval()

    # Predict reconstructed windows for TEST windows
    Xte_tensor = torch.tensor(Xte, dtype=torch.float32, device=device)
    with torch.no_grad():
        Xte_hat, _, _ = vae(Xte_tensor)
    Xte_hat_np = Xte_hat.detach().cpu().numpy()  # (n_test, 128)

    # Reshape back to (n_test, W, 4)
    win_pred_scaled = Xte_hat_np.reshape(-1, W, Fdim)

    # Reconstruct point-wise sequence in scaled domain (only from test windows)
    starts_test = starts[test_idx]
    X_pred_scaled_seq, mask_pred = reconstruct_sequence_from_windows(
        win_pred_scaled, starts_test, N=len(freq), window=W
    )

    # Only evaluate points in test band and that have prediction coverage
    band_mask = (freq >= F_TEST_MIN) & (freq <= F_TEST_MAX) & mask_pred

    # True (reference) alpha/beta/Z0 from true RLGC
    alpha_t, beta_t, Zre_t, Zim_t = rlcg_to_gamma_z0(freq, R, L, G, Cc)

    # Predicted RLGC from predicted scaled features -> inverse scale + inverse log
    # For points without prediction coverage, X_pred_scaled_seq is NaN -> handle by masking band_mask
    R_p, L_p, G_p, C_p = inv_feature_transform(X_pred_scaled_seq[band_mask], scaler)
    f_band = freq[band_mask]

    alpha_p, beta_p, Zre_p, Zim_p = rlcg_to_gamma_z0(f_band, R_p, L_p, G_p, C_p)

    # Also slice true to band for plotting
    alpha_t_b = alpha_t[band_mask]
    beta_t_b  = beta_t[band_mask]
    Zre_t_b   = Zre_t[band_mask]
    Zim_t_b   = Zim_t[band_mask]

    # ---- Plots requested by prof ----
    def plot_true_pred(x, y_true, y_pred, title, ylabel):
        plt.figure(figsize=(8,5))
        plt.plot(x/1e9, y_true, label="True")
        plt.plot(x/1e9, y_pred, label="Predicted")
        plt.title(title)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        plt.show()

    plot_true_pred(f_band, alpha_t_b, alpha_p, "Alpha (attenuation) — True vs Predicted", "alpha (Np/m)")
    plot_true_pred(f_band, beta_t_b,  beta_p,  "Beta (phase) — True vs Predicted", "beta (rad/m)")
    plot_true_pred(f_band, Zre_t_b,   Zre_p,   "Z0 Real Part — True vs Predicted", "Re{Z0} (Ohm)")
    plot_true_pred(f_band, Zim_t_b,   Zim_p,   "Z0 Imag Part — True vs Predicted", "Im{Z0} (Ohm)")

    print("[OK] Plotted predicted vs true: alpha, beta, Zreal, Zimag (67–110 GHz)")

if __name__ == "__main__":
    main()
