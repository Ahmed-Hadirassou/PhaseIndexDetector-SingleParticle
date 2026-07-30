# ---- Features -- 5 classes x 40 = 200 total ----

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import config

def _shannon_entropy(x):
    if len(x) < 10: return np.nan
    counts, _ = np.histogram(x, bins=max(5, len(x)//10))
    p = counts / counts.sum(); p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))

def _renyi_entropy(x, q=2):
    if len(x) < 10: return np.nan
    counts, _ = np.histogram(x, bins=max(5, len(x)//10))
    p = counts / counts.sum(); p = p[p > 0]
    return float(np.log(np.sum(p**q) + 1e-12) / (1 - q))

def _mc_var(x, conf=0.95, n_sim=1000):
    if len(x) < 10: return np.nan
    mu, sigma = np.mean(x), np.std(x)
    if sigma < 1e-10: return 0.0
    return float(-np.percentile(np.random.normal(mu, sigma, n_sim), (1-conf)*100))

def _cvar(x, conf=0.95):
    if len(x) < 10: return np.nan
    thr = np.percentile(x, (1-conf)*100)
    tail = x[x <= thr]
    return float(-np.mean(tail)) if len(tail) > 0 else np.nan

def _hurst(x):
    if len(x) < 20: return np.nan
    try:
        n = len(x)
        lags = [int(n*r) for r in [0.1,0.2,0.3,0.4,0.5] if int(n*r)>=4]
        if len(lags) < 2: return np.nan
        rs_vals = []
        for lag in lags:
            rs_sub = []
            for s in range(0, n-lag, lag):
                chunk = x[s:s+lag]; S = np.std(chunk)
                if S > 1e-10:
                    dev = np.cumsum(chunk - np.mean(chunk))
                    rs_sub.append((np.max(dev)-np.min(dev))/S)
            if rs_sub: rs_vals.append(np.mean(rs_sub))
        if len(rs_vals) < 2: return np.nan
        slope, *_ = stats.linregress(np.log(lags[:len(rs_vals)]), np.log(rs_vals))
        return float(slope)
    except: return np.nan

def _mdd(x):
    cum = np.cumprod(1+x); rm = np.maximum.accumulate(cum)
    return float(np.min((cum-rm)/(rm+1e-8)))

# ---- Class 1 -- statistical ----
def features_statistical(returns, market):
    feats = {}
    windows_stat = [10, 20, 40, 90, 252]
    
    for w in windows_stat:
        feats[f's_shannon_{w}'] = returns.rolling(w).apply(_shannon_entropy, raw=True)
        feats[f's_renyi_{w}'] = returns.rolling(w).apply(_renyi_entropy, raw=True)
        feats[f's_mcvar_{w}'] = returns.rolling(w).apply(_mc_var, raw=True)
        feats[f's_cvar_{w}'] = returns.rolling(w).apply(_cvar, raw=True)
        
    for w in config.WINDOWS_HURST:
        feats[f's_hurst_{w}'] = returns.rolling(w).apply(_hurst, raw=True)
        
    for w in windows_stat:
        betas = []
        for i in range(len(returns)):
            sr = returns.iloc[max(0,i-w):i].values
            sm = market.iloc[max(0,i-w):i].values
            if len(sr) < 5: betas.append(np.nan); continue
            c = np.cov(sr, sm)
            betas.append(c[0,1]/c[1,1] if c[1,1]>1e-10 else np.nan)
        feats[f's_beta_{w}'] = pd.Series(betas, index=returns.index)
        feats[f's_diffusion_{w}'] = returns.rolling(w).var()
        feats[f's_corr_mkt_{w}'] = returns.rolling(w).corr(market)

    df = pd.DataFrame(feats, index=returns.index)
    print(f"  Statistical features: {df.shape[1]}")
    return df

# ---- Class 2 -- volatility ----
def features_volatility(prices, returns):
    feats = {}
    rv20 = returns.rolling(20).std()
    windows_vol = [5, 10, 20, 40, 60, 90, 120, 252]
    
    for w in windows_vol:
        feats[f'v_rvol_{w}'] = returns.rolling(w).std() * np.sqrt(252)
        
    for lam in [0.90, 0.94, 0.96, 0.97, 0.99]:
        feats[f'v_ewma_{int(lam*100)}'] = returns.ewm(alpha=1-lam).std() * np.sqrt(252)
        
    for w in [20,40,60,90,120]:
        feats[f'v_volofvol_{w}'] = rv20.rolling(w).std()
        
    for w in [10,20,40,60,90]:
        feats[f'v_downside_{w}'] = returns.rolling(w).apply(
            lambda x: float(np.sqrt(np.mean(np.minimum(x,0)**2))*np.sqrt(252)), raw=True)
            
    for w in [20,40,60,90,120]:
        feats[f'v_sharpe_{w}'] = returns.rolling(w).mean()/(returns.rolling(w).std()+1e-8)*np.sqrt(252)
        feats[f'v_mdd_{w}'] = returns.rolling(w).apply(_mdd, raw=True)
        
    for w in [40,60,90,120]:
        mu = rv20.rolling(w).mean(); sd = rv20.rolling(w).std()
        feats[f'v_zscore_{w}'] = (rv20-mu)/(sd+1e-8)
        
    for w in [20,40,60]:
        feats[f'v_ratio_{w}'] = returns.rolling(w).std()/(returns.rolling(252).std()+1e-8)

    df = pd.DataFrame(feats, index=returns.index)
    print(f"  Volatility features: {df.shape[1]}")
    return df

# ---- Class 3 -- momentum ----
def features_momentum(prices, returns):
    feats = {}
    log_p = np.log(prices+1e-8)
    delta = prices.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    windows_mom = [5, 10, 20, 40, 60, 90, 120, 252]
    
    for w in windows_mom:
        feats[f'm_ret_{w}'] = returns.rolling(w).sum()
        
    for w in [10,14,20,40,60]:
        g = gain.rolling(w).mean(); l = loss.rolling(w).mean()
        feats[f'm_rsi_{w}'] = 100-(100/(1+g/(l+1e-8)))
        
    for w in [20,40,60,90,120]:
        feats[f'm_ma_ratio_{w}'] = prices/(prices.rolling(w).mean()+1e-8)
        
    for s,l in [(5,20),(10,40),(20,60),(20,90),(40,120)]:
        feats[f'm_macd_{s}_{l}'] = (prices.rolling(s).mean()-prices.rolling(l).mean())/(prices+1e-8)
        
    for w in [20,40,60,90,120]:
        def _slope(x):
            if len(x)<5: return np.nan
            sl,*_ = stats.linregress(np.arange(len(x)), x); return float(sl)
        feats[f'm_slope_{w}'] = log_p.rolling(w).apply(_slope, raw=True)
        
    for w in [10,20,40,60,90]:
        feats[f'm_ac1_{w}'] = returns.rolling(w).apply(
            lambda x: float(np.corrcoef(x[:-1],x[1:])[0,1]) if len(x)>2 else np.nan, raw=True)
            
    for w in [10,20,40,60]:
        mu=prices.rolling(w).mean(); sd=prices.rolling(w).std()
        feats[f'm_bband_{w}'] = (prices-mu)/(sd+1e-8)
        
    for w in [20,40,60]:
        feats[f'm_pctpos_{w}'] = returns.rolling(w).apply(lambda x: float(np.mean(x>0)), raw=True)

    df = pd.DataFrame(feats, index=returns.index)
    print(f"  Momentum features: {df.shape[1]}")
    return df

# ---- Class 4 -- market ----
def features_market(prices, returns, all_prices, all_returns):
    feats = {}
    
    for col in ['TLT','GLD','EEM']:
        if col in all_returns.columns:
            for w in [20,40,60,90,120]:
                feats[f'k_corr_{col}_{w}'] = returns.rolling(w).corr(all_returns[col])
        else:
            for w in [20,40,60,90,120]:
                feats[f'k_corr_{col}_{w}'] = pd.Series(np.nan, index=returns.index)
                
    if 'HYG' in all_returns.columns and 'IEF' in all_returns.columns:
        sp = all_returns['HYG']-all_returns['IEF']
        for w in [10,20,40,60,90]:
            feats[f'k_credit_{w}'] = sp.rolling(w).mean()
    else:
        for w in [10,20,40,60,90]:
            feats[f'k_credit_{w}'] = pd.Series(np.nan, index=returns.index)
            
    if 'TLT' in all_prices.columns and 'SHY' in all_prices.columns:
        yc = np.log(all_prices['TLT']/(all_prices['SHY']+1e-8))
        for w in [10,20,40,60,90]:
            feats[f'k_yc_{w}'] = yc.rolling(w).mean()
    else:
        for w in [10,20,40,60,90]:
            feats[f'k_yc_{w}'] = pd.Series(np.nan, index=returns.index)
            
    for w in [20,40,60,90,120]:
        def _vr(x):
            if len(x)<w//2: return np.nan
            v1=np.var(x); agg=[np.sum(x[i:i+2]) for i in range(0,len(x)-1,2)]
            v2=np.var(agg)/2 if len(agg)>1 else np.nan
            return float(v2/v1) if v1>1e-10 else np.nan
        feats[f'k_vr_{w}'] = returns.rolling(w*2).apply(_vr, raw=True)
        
    sc = [c for c in config.TICKERS_SECTOR if c in all_prices.columns]
    for w in [20,40,60,90,120]:
        if len(sc)>=5:
            vals = []
            for i in range(len(prices)):
                above = sum(1 for c in sc if i>=50 and
                            all_prices[c].iloc[i]>all_prices[c].iloc[max(0,i-50):i].mean())
                vals.append(above/len(sc))
            feats[f'k_breadth_{w}'] = pd.Series(vals, index=prices.index)
        else:
            feats[f'k_breadth_{w}'] = pd.Series(np.nan, index=prices.index)
            
    if 'TLT' in all_returns.columns:
        for w in [20,40,60,90,120]:
            feats[f'k_ftq_{w}'] = returns.rolling(w).corr(all_returns['TLT'])
    else:
        for w in [20,40,60,90,120]:
            feats[f'k_ftq_{w}'] = pd.Series(np.nan, index=returns.index)

    df = pd.DataFrame(feats, index=prices.index)
    print(f"  Market features: {df.shape[1]}")
    return df

# ---- Class 5 -- macro ----
def features_macro(prices, returns, all_prices, all_returns):
    feats = {}
    windows_macro = [10, 20, 40, 60, 90, 120, 180, 252]
    
    ro_cols = [c for c in ['SPY','QQQ','EEM','HYG'] if c in all_returns.columns]
    rf_cols = [c for c in ['TLT','GLD','SHY','UUP'] if c in all_returns.columns]
    if ro_cols and rf_cols:
        ro = all_returns[ro_cols].mean(axis=1)
        rf = all_returns[rf_cols].mean(axis=1)
        for w in windows_macro:
            feats[f'c_roro_{w}'] = ro.rolling(w).sum()-rf.rolling(w).sum()
    else:
        for w in windows_macro:
            feats[f'c_roro_{w}'] = pd.Series(np.nan, index=returns.index)
            
    if 'UUP' in all_returns.columns:
        for w in [10,20,40,60,90]:
            feats[f'c_dollar_{w}'] = all_returns['UUP'].rolling(w).sum()
    else:
        for w in [10,20,40,60,90]:
            feats[f'c_dollar_{w}'] = pd.Series(np.nan, index=returns.index)
            
    if 'TIP' in all_prices.columns and 'IEF' in all_prices.columns:
        infl = np.log(all_prices['TIP']/(all_prices['IEF']+1e-8))
        for w in [20,40,60,90,120]:
            feats[f'c_infl_{w}'] = infl.rolling(w).mean()
    else:
        for w in [20,40,60,90,120]:
            feats[f'c_infl_{w}'] = pd.Series(np.nan, index=returns.index)
            
    gc = [c for c in ['SPY','TLT','GLD','EEM','HYG'] if c in all_returns.columns]
    for w in [40,60,90,120,180]:
        if len(gc)>=4:
            def _ac(i,w=w):
                sl=all_returns[gc].iloc[max(0,i-w):i]
                if sl.shape[0]<20: return np.nan
                corr=sl.corr().values; n=corr.shape[0]
                return float(np.sum(np.triu(corr,1))/(n*(n-1)/2))
            feats[f'c_gcorr_{w}'] = pd.Series([_ac(i) for i in range(len(returns))], index=returns.index)
        else:
            feats[f'c_gcorr_{w}'] = pd.Series(np.nan, index=returns.index)
            
    sc = [c for c in config.TICKERS_SECTOR if c in all_returns.columns]
    for w in [20,40,60,90,120]:
        if len(sc)>=5:
            feats[f'c_sdisp_{w}'] = all_returns[sc].rolling(w).sum().std(axis=1)
        else:
            feats[f'c_sdisp_{w}'] = pd.Series(np.nan, index=returns.index)
            
    for col in ['TLT','IEF','SHY','HYG']:
        if col in all_returns.columns:
            feats[f'c_bond_{col}'] = all_returns[col].rolling(60).sum()
        else:
            feats[f'c_bond_{col}'] = pd.Series(np.nan, index=returns.index)
            
    for col in ['EFA','EEM','FXI','EWJ']:
        if col in all_returns.columns:
            feats[f'c_intl_{col}'] = all_returns[col].rolling(60).sum()
        else:
            feats[f'c_intl_{col}'] = pd.Series(np.nan, index=returns.index)
            
    if 'TLT' in all_prices.columns and 'SHY' in all_prices.columns:
        yc = np.log(all_prices['TLT']/(all_prices['SHY']+1e-8))
        for w in [5,10,20,40]:
            feats[f'c_yc_chg_{w}'] = yc.diff(w)
    else:
        for w in [5,10,20,40]:
            feats[f'c_yc_chg_{w}'] = pd.Series(np.nan, index=returns.index)

    df = pd.DataFrame(feats, index=returns.index)
    print(f"  Macro features: {df.shape[1]}")
    return df

# ---- Master builder ----
def build_base_features(prices, all_prices):
    returns     = np.log(prices/prices.shift(1)).dropna()
    all_returns = np.log(all_prices/all_prices.shift(1))
    market      = all_returns['SPY'] if 'SPY' in all_returns.columns else returns

    print("Building base features...")
    df_stat = features_statistical(returns, market)
    df_vol  = features_volatility(prices, returns)
    df_mom  = features_momentum(prices, returns)
    df_mkt  = features_market(prices, returns, all_prices, all_returns)
    df_mac  = features_macro(prices, returns, all_prices, all_returns)

    return returns, df_stat, df_vol, df_mom, df_mkt, df_mac
