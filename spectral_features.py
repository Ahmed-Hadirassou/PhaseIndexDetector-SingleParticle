# ---- Spectral features -- wavelet energy decomposition ----
#
# E = sum |x(t)|^2 = sum |X(f)|^2  (Parseval)
#
# R0 (stable):     energy in low frequencies
# R1 (critical):   energy in high frequencies
# R2 (transition): energy migrating between bands
#
# Manual CWT with a Ricker (Mexican hat) wavelet
# (scipy.signal.cwt was removed in scipy >= 1.12)
#
# Note: not currently imported by main_v304_soft_labels.py. The feature
# split there (SLOW_IDX/FAST_IDX, 80+124=204 dims) is sized for the five
# base feature classes plus velocity features only -- confirm whether this
# module's 40 columns are meant to be merged in before wiring it up.

import numpy as np
import pandas as pd


def _ricker(points, a):
    """Ricker (Mexican hat) wavelet. a = scale."""
    A = 2.0 / (np.sqrt(3.0 * a) * (np.pi**0.25))
    wsq = a**2
    t = np.arange(points) - (points - 1.0) / 2
    mod = 1.0 - (t**2) / wsq
    gauss = np.exp(-(t**2) / (2.0 * wsq))
    return A * mod * gauss


def _cwt_ricker(data, scales):
    """Manual CWT with a Ricker wavelet via convolution."""
    n = len(data)
    output = np.zeros((len(scales), n))
    for i, scale in enumerate(scales):
        # Wavelet width: 10*scale or len(data), whichever is smaller
        width = min(10 * int(scale) + 1, n)
        if width < 3:
            width = 3
        wavelet = _ricker(width, scale)
        # Convolve (same mode to preserve length)
        output[i] = np.convolve(data, wavelet, mode='same')
    return output


def build_spectral_features(returns):
    """
    40 spectral features from wavelet decomposition.

    4 frequency bands x 5 windows x 2 features = 40:
      - Band energy fractions (4 per window)
      - HF/LF ratio (1 per window)
      - Spectral entropy (1 per window)
      - Spectral centroid (1 per window)
      - Total energy (1 per window)
    """
    print("  Spectral features (wavelet decomposition)...")

    scales_hf = np.array([2, 3, 4, 5])
    scales_mf = np.array([6, 8, 10, 14, 20])
    scales_lf = np.array([25, 30, 40, 50, 60])
    scales_vlf = np.array([80, 100, 120])

    windows = [60, 90, 120, 180, 252]
    feats = {}
    ret_vals = returns.values.astype(np.float64)

    for w in windows:
        n = len(returns)
        e_hf = np.full(n, np.nan)
        e_mf = np.full(n, np.nan)
        e_lf = np.full(n, np.nan)
        e_vlf = np.full(n, np.nan)
        ratio_hf_lf = np.full(n, np.nan)
        s_entropy = np.full(n, np.nan)
        s_centroid = np.full(n, np.nan)
        e_total = np.full(n, np.nan)

        # Select scales that fit in the window
        valid_hf = scales_hf[scales_hf < w // 2]
        valid_mf = scales_mf[scales_mf < w // 2]
        valid_lf = scales_lf[scales_lf < w // 2]
        valid_vlf = scales_vlf[scales_vlf < w // 2]
        all_valid = np.concatenate([valid_hf, valid_mf, valid_lf, valid_vlf])

        if len(all_valid) < 4:
            # Not enough scales for this window
            for name, arr in [('ehf', e_hf), ('emf', e_mf), ('elf', e_lf),
                               ('evlf', e_vlf), ('hflf', ratio_hf_lf),
                               ('entropy', s_entropy), ('centroid', s_centroid),
                               ('etotal', e_total)]:
                feats[f'sp_{name}_{w}'] = pd.Series(arr, index=returns.index)
            continue

        n_hf = len(valid_hf)
        n_mf = len(valid_mf)
        n_lf = len(valid_lf)
        n_vlf = len(valid_vlf)

        for i in range(w, n):
            chunk = ret_vals[i - w:i]
            if np.any(np.isnan(chunk)):
                continue

            try:
                coeffs = _cwt_ricker(chunk, all_valid)
                power = np.mean(coeffs**2, axis=1)

                idx_start = 0
                p_hf = power[idx_start:idx_start + n_hf].sum() if n_hf > 0 else 0
                idx_start += n_hf
                p_mf = power[idx_start:idx_start + n_mf].sum() if n_mf > 0 else 0
                idx_start += n_mf
                p_lf = power[idx_start:idx_start + n_lf].sum() if n_lf > 0 else 0
                idx_start += n_lf
                p_vlf = power[idx_start:idx_start + n_vlf].sum() if n_vlf > 0 else 0

                p_total = p_hf + p_mf + p_lf + p_vlf + 1e-12

                e_hf[i] = p_hf / p_total
                e_mf[i] = p_mf / p_total
                e_lf[i] = p_lf / p_total
                e_vlf[i] = p_vlf / p_total
                ratio_hf_lf[i] = p_hf / (p_lf + 1e-12)
                e_total[i] = np.log(p_total + 1e-12)

                p_norm = power / (power.sum() + 1e-12)
                p_pos = p_norm[p_norm > 0]
                s_entropy[i] = -np.sum(p_pos * np.log(p_pos + 1e-12))

                freqs = 1.0 / all_valid
                s_centroid[i] = np.sum(freqs * power) / (power.sum() + 1e-12)

            except Exception:
                continue

        feats[f'sp_ehf_{w}'] = pd.Series(e_hf, index=returns.index)
        feats[f'sp_emf_{w}'] = pd.Series(e_mf, index=returns.index)
        feats[f'sp_elf_{w}'] = pd.Series(e_lf, index=returns.index)
        feats[f'sp_evlf_{w}'] = pd.Series(e_vlf, index=returns.index)
        feats[f'sp_hflf_{w}'] = pd.Series(ratio_hf_lf, index=returns.index)
        feats[f'sp_entropy_{w}'] = pd.Series(s_entropy, index=returns.index)
        feats[f'sp_centroid_{w}'] = pd.Series(s_centroid, index=returns.index)
        feats[f'sp_etotal_{w}'] = pd.Series(e_total, index=returns.index)

    df = pd.DataFrame(feats, index=returns.index)
    print(f"  Spectral features: {df.shape[1]}")
    return df
