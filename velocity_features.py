# ---- Velocity features -- first and second differences of realized volatility ----
import pandas as pd
import numpy as np


def build_velocity_features(returns):
    print("  Velocity features...")
    feats_vel = {}

    # Base indicator: 20-day realized volatility
    rvol_20 = returns.rolling(20).std() * np.sqrt(252)

    # Velocity (first differences)
    for lag in [5, 10, 20]:
        feats_vel[f'vel_rvol_d{lag}'] = rvol_20.diff(lag)

    # Acceleration (second difference)
    feats_vel['acc_rvol'] = rvol_20 - 2 * rvol_20.shift(5) + rvol_20.shift(10)

    return pd.DataFrame(feats_vel, index=returns.index)
