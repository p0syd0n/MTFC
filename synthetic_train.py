import numpy as np
import pandas as pd

# Data config
TRAIN_LEN = 5000
TEST_LEN  = 2000
S_TRAIN   = 42
S_TEST    = 99

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_candles(n_rows, seed):
    rng = np.random.default_rng(seed)
    
    # Starting states
    p_last = 100.0
    v_last = 100.0
    raw_data = []

    # Pad for lookbacks
    for _ in range(n_rows + 20):
        o = p_last + rng.normal(0, 0.05)
        b = rng.normal(0, 0.4)
        c = o + b
        
        # Derived candle props
        r = abs(b) + abs(rng.normal(0, 0.15))
        h = max(o, c) + abs(rng.normal(0, r * 0.25))
        l = min(o, c) - abs(rng.normal(0, r * 0.25))
        
        # Volume/Delta sim
        tks = max(10, int(rng.normal(200, 70)))
        dlt = rng.normal(b * 50, 25)
        
        v_prev = v_last
        v_last = v_last * 0.96 + c * 0.04 # EMA-style VPOC

        raw_data.append({
            "open": o, "close": c, "high": h, "low": l,
            "ticks_raw": tks, "delta_raw": dlt,
            "vpoc": v_last, "vpoc_prev": v_prev
        })
        p_last = c

    df = pd.DataFrame(raw_data)

    # Signal math
    body = df["close"] - df["open"]
    rng_val = (df["high"] - df["low"]).replace(0, 1e-6)
    
    # Feature engineering
    feats = pd.DataFrame({
        "open": df["open"],
        "close": df["close"],
        "high": df["high"],
        "low": df["low"],
        "body": body,
        "range": rng_val,
        "upper_wick": df["high"] - df[["open", "close"]].max(axis=1),
        "lower_wick": df[["open", "close"]].min(axis=1) - df["low"],
        "body_ratio": body / rng_val,
        "Ticks": df["ticks_raw"],
        "delta": df["delta_raw"],
        "delta_ratio": df["delta_raw"] / df["ticks_raw"].replace(0, 1),
        "close_vs_vpoc": df["close"] - df["vpoc"],
        "vpoc_move": df["vpoc"] - df["vpoc_prev"]
    })

    # Rolling windows
    avg_r20 = feats["range"].rolling(20, min_periods=1).mean()
    feats["range_ratio"] = feats["range"] / avg_r20.replace(0, 1e-6)
    feats["momentum_3"]  = feats["body"].rolling(3, min_periods=1).sum()
    feats["momentum_10"] = feats["body"].rolling(10, min_periods=1).sum()
    feats["LinRegSlope"] = feats["body"].rolling(5, min_periods=1).mean()

    # Apply probabilistic patterns
    n = len(feats)
    p_up = np.full(n, 0.50)
    
    for i in range(n):
        l_odds = logit(0.50)
        
        # 1. Price action / Momentum
        b_val = feats.at[i, "body"]
        if b_val > 0: l_odds += logit(0.75) - logit(0.50)
        elif b_val < 0: l_odds += logit(0.25) - logit(0.50)

        # 2. Volume confirmation
        t_val = feats.at[i, "Ticks"]
        if t_val > 280:
            l_odds += (logit(0.80) if b_val > 0 else logit(0.20)) - logit(0.50)

        # 3. Consolidation/Narrow range
        if feats.at[i, "range_ratio"] < 0.65:
            l_odds += (logit(0.70) if b_val > 0 else logit(0.30)) - logit(0.50)

        # 4. Value area positioning
        if feats.at[i, "close_vs_vpoc"] > 0:
            l_odds += logit(0.65) - logit(0.50)
        else:
            l_odds += logit(0.35) - logit(0.50)

        # 5. Delta imbalance
        dr = feats.at[i, "delta_ratio"]
        if dr > 0.3: l_odds += logit(0.68) - logit(0.50)
        elif dr < -0.3: l_odds += logit(0.32) - logit(0.50)

        # 6. Aggregate momentum
        m3 = feats.at[i, "momentum_3"]
        if m3 > 0.4: l_odds += logit(0.72) - logit(0.50)
        elif m3 < -0.4: l_odds += logit(0.28) - logit(0.50)

        p_up[i] = sigmoid(l_odds)

    # Force next-candle labels based on P(Up)
    new_c = feats["close"].values.copy()
    o_vals = feats["open"].values
    
    for i in range(n - 1):
        nxt = i + 1
        is_up = rng.random() < p_up[i]
        
        curr_o = o_vals[nxt]
        curr_c = new_c[nxt]
        diff = abs(curr_c - curr_o)
        
        # Shift close to match simulated direction
        if is_up and curr_c < curr_o:
            new_c[nxt] = curr_o + diff + rng.uniform(0.01, 0.05)
        elif not is_up and curr_c >= curr_o:
            new_c[nxt] = curr_o - diff - rng.uniform(0.01, 0.05)

    feats["close"] = np.round(new_c, 4)
    
    # Trim warmup rows and return
    return feats.iloc[20:20 + n_rows].reset_index(drop=True)

if __name__ == "__main__":
    print("--- Generating Synthetic Datasets ---")
    
    train = generate_candles(TRAIN_LEN, S_TRAIN)
    test  = generate_candles(TEST_LEN, S_TEST)
    
    train.to_csv("synthetic_train.csv", index=False)
    test.to_csv("synthetic_test.csv", index=False)
    
    print(f"Done. Train: {len(train)} rows | Test: {len(test)} rows")
    
    # Quick sanity check on correlations
    print("\nFeature Target Correlations:")
    target = (train["close"].shift(-1) >= train["open"].shift(-1)).astype(int).iloc[:-1]
    check_cols = ["body", "Ticks", "delta_ratio", "close_vs_vpoc", "momentum_3", "range_ratio"]
    
    for c in check_cols:
        r = train[c].iloc[:-1].corr(target)
        print(f" -> {c:15} r = {r:+.4f}")

    print(f"\nTarget distribution (Buy %): {target.mean()*100:.2f}%")