import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("btc_corrected_mtfc_2.csv")
# percent returns
df["close_ret"] = df["close"].pct_change()
df["open_ret"]  = df["open"].pct_change()
df["high_ret"]  = df["high"].pct_change()
df["low_ret"]   = df["low"].pct_change()

df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

# target
df["next_open"]  = df["open"].shift(-1)
df["next_close"] = df["close"].shift(-1)
df["target"] = (df["next_close"] > df["next_open"]).astype(int)

# fix NaNs and infs caused by returns + shifts
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()


# features
X = df.drop(columns=["target", "next_open", "next_close"])
y = df["target"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
)

model.fit(X_train, y_train)
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
