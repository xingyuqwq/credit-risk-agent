# -*- coding: utf-8 -*-
"""训练信用风险模型，标定判别阈值并保存模型包。

流程为标准化 -> SMOTE -> 逻辑回归；在训练集 5 折袋外预测上做 Platt 校准，
并按 F1 最大原则标定判别中心。产物供 risk_agent.py 推理时加载。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "default"

# 显式声明特征顺序，保证训练与推理阶段一致
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]

# 灰区半宽：在标定阈值两侧各扩展该宽度，区间内转人工复核
GREY_MARGIN = 0.15


def load_dataset() -> pd.DataFrame:
    """读取清洗后的 UCI 信用卡违约数据集（CSV，UTF-8 with BOM）。"""
    candidates = sorted(DATA_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"未在 {DATA_DIR} 找到数据文件，请先运行 code/fetch_data.py 获取数据集"
        )
    df = pd.read_csv(candidates[0])
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"数据缺少必要列：{missing}")
    return df


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """把概率转成对数几率，作为 Platt 校准的输入特征。"""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def calibrate_threshold(oof_prob: np.ndarray, y_true: np.ndarray) -> float:
    """在袋外概率上扫描阈值，取 F1 最大的工作点作为自动判定中心 tau。

    选择 F1 而非准确率，是因为在 22% 违约率的不平衡数据上，准确率会被
    多数类主导（全部预测为未违约即可获得 78%），无法反映风险识别能力。
    """
    grid = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(y_true, (oof_prob >= t).astype(int), zero_division=0) for t in grid]
    best_idx = int(np.argmax(scores))
    return float(grid[best_idx])


def train(random_state: int = RANDOM_STATE) -> dict:
    """训练模型并返回可直接落盘的模型包（dict）。"""
    df = load_dataset()
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )

    # 只转 ndarray，避免 sklearn 记录列名后在推理阶段报警告
    X_train_np = X_train.to_numpy(dtype=float)
    X_test_np = X_test.to_numpy(dtype=float)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_np)
    X_test_s = scaler.transform(X_test_np)

    # SMOTE + 逻辑回归：重采样只作用于训练阶段，predict 阶段自动跳过
    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=random_state)),
        ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
    ])

    # ---- 5 折袋外预测，用于阈值标定----
    oof_prob = np.zeros(len(y_train))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    y_train_arr = y_train.to_numpy()
    for tr_idx, va_idx in skf.split(X_train_s, y_train_arr):
        fold_model = ImbPipeline([
            ("smote", SMOTE(random_state=random_state)),
            ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
        ])
        fold_model.fit(X_train_s[tr_idx], y_train_arr[tr_idx])
        oof_prob[va_idx] = fold_model.predict_proba(X_train_s[va_idx])[:, 1]

    # Platt 校准：在袋外概率的对数几率上再拟合一维逻辑回归，
    # 只改变数值刻度不改变排序，AUC 不变而 Brier 明显下降
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(_logit(oof_prob).reshape(-1, 1), y_train_arr)
    oof_cal = calibrator.predict_proba(_logit(oof_prob).reshape(-1, 1))[:, 1]

    # 未校准概率上的标定点也一并保存，供消融实验对比使用
    tau_raw = calibrate_threshold(oof_prob, y_train_arr)
    tau = calibrate_threshold(oof_cal, y_train_arr)

    # ---- 用全部训练数据重新拟合，得到最终模型 ----
    pipeline.fit(X_train_s, y_train_arr)

    # 测试集表现 + 校准前后对比：AUC 应基本不变，Brier 与平均预测率应更接近真实
    from sklearn.metrics import brier_score_loss, roc_auc_score
    raw_test = pipeline.predict_proba(X_test_s)[:, 1]
    cal_test = calibrator.predict_proba(_logit(raw_test).reshape(-1, 1))[:, 1]
    test_auc = float(roc_auc_score(y_test, raw_test))
    brier_raw = float(brier_score_loss(y_test, raw_test))
    brier_cal = float(brier_score_loss(y_test, cal_test))

    coef = pipeline.named_steps["clf"].coef_[0]
    intercept = float(pipeline.named_steps["clf"].intercept_[0])

    tau_low = round(max(0.05, tau - GREY_MARGIN), 4)
    tau_high = round(min(0.95, tau + GREY_MARGIN), 4)

    bundle = {
        "scaler": scaler,
        "pipeline": pipeline,
        "calibrator": calibrator,
        "features": FEATURES,
        "coefficients": {f: float(c) for f, c in zip(FEATURES, coef)},
        "intercept": intercept,
        "thresholds": {
            "auto_low": tau_low,          # 低于此值：自动通过（低风险）
            "auto_high": tau_high,        # 高于此值：高风险，提请人工确认
            "center": round(tau, 4),      # 标定得到的 F1 最优点（校准后概率）
            "raw_center": round(tau_raw, 4),  # 未校准概率上的等价标定点（消融用）
            "grey_margin": GREY_MARGIN,
        },
        "metrics": {
            "test_auc": round(test_auc, 4),
            "brier_raw": round(brier_raw, 4),
            "brier_calibrated": round(brier_cal, 4),
            "mean_pred_raw": round(float(raw_test.mean()), 4),
            "mean_pred_calibrated": round(float(cal_test.mean()), 4),
            "base_rate": round(float(y_test.mean()), 4),
        },
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_state": random_state,
    }
    return bundle


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("信用风险模型训练与阈值标定")
    print("=" * 66)
    bundle = train()

    model_path = MODEL_DIR / "credit_risk_model.joblib"
    joblib.dump(bundle, model_path)

    meta = {
        "model": "StandardScaler + SMOTE + LogisticRegression",
        "n_features": len(bundle["features"]),
        "features": bundle["features"],
        "thresholds": bundle["thresholds"],
        "metrics": bundle["metrics"],
        "trained_at": bundle["trained_at"],
        "top_risk_factors": sorted(
            bundle["coefficients"].items(), key=lambda kv: kv[1], reverse=True
        )[:5],
        "top_protective_factors": sorted(
            bundle["coefficients"].items(), key=lambda kv: kv[1]
        )[:5],
    }
    with open(RESULT_DIR / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    th = bundle["thresholds"]
    m = bundle["metrics"]
    print(f"[模型] 已保存：{model_path}")
    print(f"[模型] 测试集 AUC = {m['test_auc']:.4f}")
    print(f"[校准] 平均预测违约率：校准前 {m['mean_pred_raw']:.3f} "
          f"-> 校准后 {m['mean_pred_calibrated']:.3f}（真实 {m['base_rate']:.3f}）")
    print(f"[校准] Brier 分数：校准前 {m['brier_raw']:.4f} -> 校准后 {m['brier_calibrated']:.4f}")
    print(f"[阈值] 标定中心 tau = {th['center']:.3f}，"
          f"灰区 = [{th['auto_low']:.3f}, {th['auto_high']:.3f}]")
    print(f"[阈值] 低于 {th['auto_low']:.2f} 自动通过；"
          f"区间内转人工复核；高于 {th['auto_high']:.2f} 提请优先核查")
    print("[完成] 元数据已写入 results/model_meta.json")


if __name__ == "__main__":
    main()
