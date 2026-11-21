# -*- coding: utf-8 -*-
"""通过 ucimlrepo API 获取 UCI 信用卡违约数据集并保存为规范 CSV。"""
import os, pandas as pd
from ucimlrepo import fetch_ucirepo

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(BASE, "data", "UCI_Credit_Card.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)

print("[FETCH] fetching UCI dataset id=350 ...", flush=True)
ds = fetch_ucirepo(id=350)
X = ds.data.features.copy()
y = ds.data.targets.copy()
print(f"[FETCH] X={X.shape} y={y.shape}", flush=True)
print("[FETCH] X columns:", list(X.columns), flush=True)
print("[FETCH] y columns:", list(y.columns), flush=True)

# ucimlrepo 返回的 X 有 23 列（已去掉 ID）：X1=LIMIT_BAL ... X23=PAY_AMT6
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
assert X.shape[1] == len(FEATURES), f"列数不符 {X.shape[1]} vs {len(FEATURES)}"
X.columns = FEATURES
target_col = y.columns[0]
y = y.rename(columns={target_col: "default"})

df = pd.concat([X, y], axis=1)
# 先写临时文件，成功后再替换，避免中断留下损坏文件
tmp = out + ".tmp"
df.to_csv(tmp, index=False, encoding="utf-8-sig")
os.replace(tmp, out)

print(f"[SAVED] {out} shape={df.shape}", flush=True)
print(f"[STAT] 违约率={df['default'].mean()*100:.2f}% 缺失值={int(df.isnull().sum().sum())}", flush=True)
print(df.head(3).to_string(), flush=True)
