# -*- coding: utf-8 -*-
"""生成 Agent 相关的四张图：流程图、灰区权衡、消融对比、校准曲线。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

from risk_agent import CreditRiskAgent
from train_model import FEATURES, RANDOM_STATE, TARGET, TEST_SIZE, load_dataset

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"


def setup_chinese_font() -> None:
    """用字体文件显式注册中文字体，避免依赖字体名解析。"""
    import matplotlib.font_manager as fm
    for name in ["simhei.ttf", "msyh.ttc"]:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            fm.fontManager.addfont(str(path))
            plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=str(path)).get_name()]
            break
    plt.rcParams["axes.unicode_minus"] = False


setup_chinese_font()
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 170
plt.rcParams["savefig.bbox"] = "tight"


def draw_agent_flow() -> None:
    """绘制 Agent 六步工作流。"""
    fig, ax = plt.subplots(figsize=(13.2, 8))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 8)
    ax.axis("off")

    edge = "#2c3e50"
    arrow_kw = dict(arrowstyle="-|>", color="#34495e", lw=1.4,
                    shrinkA=0, shrinkB=0, mutation_scale=14)

    def box(x, y, w, h, text, color, fs=10):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=color, edgecolor=edge, linewidth=1.3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, linespacing=1.65)
        return (x, y, w, h)

    def arrow(p1, p2):
        ax.annotate("", xy=p2, xytext=p1, arrowprops=arrow_kw)

    C_IN, C_STEP, C_MODEL = "#dfe6ec", "#d6e4f0", "#cfe3d4"
    C_TOOL, C_OUT = "#f6e7c1", "#e8d5d5"

    ax.text(6.6, 7.62, "信用风险审查 Agent 工作流程", ha="center",
            fontsize=15.5, fontweight="bold")

    ytop, htop = 4.8, 1.05
    b1 = box(0.30, ytop, 1.45, htop, "申请件输入\n（JSON）", C_IN)
    b2 = box(2.05, ytop, 1.65, htop, "1 输入校验\n字段 / 类型 / 取值", C_STEP)
    b3 = box(4.00, ytop, 1.65, htop, "2 特征构造\n派生业务指标", C_STEP)
    b4 = box(5.95, ytop, 1.75, htop, "3 模型推演\nSMOTE-LR 概率", C_MODEL)
    b5 = box(8.00, ytop, 1.75, htop, "4 证据生成\n系数贡献 + 规则", C_STEP)
    b6 = box(10.05, ytop, 1.80, htop, "5 决策分流\n按概率三路", C_STEP)

    for a, b in [(b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6)]:
        arrow((a[0] + a[2], a[1] + a[3] / 2), (b[0], b[1] + b[3] / 2))

    # 上侧旁路：模型工件与规则知识库
    mf = box(5.95, 6.45, 1.75, 0.85, "模型工件\nscaler + LR + 校准器", C_TOOL, 9)
    arrow((mf[0] + mf[2] / 2, mf[1]), (mf[0] + mf[2] / 2, ytop + htop))
    kb = box(8.00, 6.45, 1.75, 0.85, "规则知识库\n11 条业务规则", C_TOOL, 9.5)
    arrow((kb[0] + kb[2] / 2, kb[1]), (kb[0] + kb[2] / 2, ytop + htop))

    # 下侧：异常出口 + 三条分流
    ylow, hlow = 2.10, 1.20
    bad = box(2.05, ylow, 1.65, hlow, "输入异常\n拒答并列出问题", C_OUT, 9.5)
    arrow((bad[0] + bad[2] / 2, ytop), (bad[0] + bad[2] / 2, ylow + hlow))
    ax.text(bad[0] + bad[2] / 2 + 0.16, (ytop + ylow + hlow) / 2, "校验失败",
            fontsize=8.5, color="#8a4b4b", ha="left", rotation=90, va="center")

    outs = [
        (3.95, "低风险  P < 0.17\n自动通过", "#d5ecd9"),
        (6.95, "灰区  0.17 – 0.47\n转人工复核", "#fbe8c8"),
        (9.95, "高风险  P > 0.47\n优先核查，不自动拒绝", "#f3d2d2"),
    ]
    for x, text, color in outs:
        box(x, ylow, 2.55, hlow, text, color, 9.5)

    # 分流总线：决策出口先下探到中间高度，再横向分发
    bus_y = 3.72
    ax.plot([b6[0] + b6[2] / 2, b6[0] + b6[2] / 2], [ytop, bus_y],
            color="#34495e", lw=1.4)
    ax.plot([5.22, b6[0] + b6[2] / 2], [bus_y, bus_y], color="#34495e", lw=1.4)
    for x in [5.22, 8.22, 11.22]:
        arrow((x, bus_y), (x, ylow + hlow))

    ax.text(6.6, 1.55,
            "免责声明：结论仅用于风险排序与人工辅助，不构成自动拒贷依据。",
            ha="center", fontsize=9.5, color="#8a4b4b", style="italic")

    fig.savefig(FIG_DIR / "09_agent_flow.png")
    plt.close(fig)
    print("[图] 09_agent_flow.png")


def draw_tradeoff() -> None:
    """灰区下界扫描：自动通过率与复核覆盖率、误放行率之间的权衡。"""
    df = pd.read_csv(RESULT_DIR / "agent_threshold.csv")
    fig, ax1 = plt.subplots(figsize=(8.6, 5.2))
    ax2 = ax1.twinx()

    l1, = ax1.plot(df["自动通过率"], df["复核覆盖违约客户比例"], "o-",
                   color="#2980b9", lw=2, ms=4.5, label="复核覆盖违约客户比例（左轴）")
    l2, = ax2.plot(df["自动通过率"], df["自动通过中实际违约率"], "s--",
                   color="#c0392b", lw=1.8, ms=4.5,
                   label="自动通过中实际违约率（右轴）")

    ax1.set_xlabel("自动通过率（无需人工处理的样本占比）")
    ax1.set_ylabel("复核覆盖违约客户比例", color="#2980b9")
    ax2.set_ylabel("自动通过样本中的实际违约率", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax1.set_title("灰区下界扫描：效率与安全性的权衡")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0.25, 1.05)
    ax1.legend(handles=[l1, l2], loc="lower left", fontsize=8.8)

    current = df[df["是否为标定方案"]]
    if not current.empty:
        x = float(current.iloc[0]["自动通过率"])
        y = float(current.iloc[0]["复核覆盖违约客户比例"])
        ax1.plot([x], [y], "o", ms=11, mfc="none", mec="#1a5276", mew=2)
        ax1.annotate(f"标定方案：自动通过 {x:.1%}\n复核覆盖违约客户 {y:.1%}",
                     xy=(x, y), xytext=(x + 0.13, y - 0.30),
                     fontsize=9, color="#1a5276",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1a5276", alpha=0.85),
                     arrowprops=dict(arrowstyle="->", color="#1a5276", lw=1.2))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_agent_tradeoff.png")
    plt.close(fig)
    print("[图] 10_agent_tradeoff.png")


def draw_ablation() -> None:
    """消融实验：自动放行漏判数与复核覆盖率的对比。"""
    df = pd.read_csv(RESULT_DIR / "agent_ablation.csv")
    labels = ["A 完整Agent", "B 无灰区分流", "C 无概率校准", "D 无类别平衡"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    colors = ["#2980b9", "#95a5a6", "#e67e22", "#8e44ad"]
    for ax, col, title in [
        (axes[0], "自动放行漏判数", "自动放行中的违约漏判数（越低越好）"),
        (axes[1], "复核覆盖违约客户比例", "转人工复核覆盖的违约客户比例"),
    ]:
        bars = ax.bar(labels, df[col], color=colors, alpha=0.85)
        ax.set_title(title, fontsize=11.5)
        ax.set_ylim(0, max(df[col]) * 1.25)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=9)
        for b, v in zip(bars, df[col]):
            txt = f"{v:.0f}" if col == "自动放行漏判数" else f"{v:.1%}"
            ax.text(b.get_x() + b.get_width() / 2, v, txt,
                    ha="center", va="bottom", fontsize=9.5)

    fig.suptitle("消融实验：各组件对分流安全性与人复核覆盖的影响", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_agent_ablation.png")
    plt.close(fig)
    print("[图] 11_agent_ablation.png")


def draw_calibration() -> None:
    """校准前后可靠性曲线。"""
    agent = CreditRiskAgent()
    b = agent.bundle
    df = load_dataset()
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    Xn = X_test.to_numpy(dtype=float)
    z = b["scaler"].transform(Xn)
    raw = b["pipeline"].predict_proba(z)[:, 1]

    from train_model import _logit
    calibrated = b["calibrator"].predict_proba(_logit(raw).reshape(-1, 1))[:, 1]

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="完全校准参考线")
    for prob, name, color in [
        (raw, "校准前（SMOTE 原始输出）", "#c0392b"),
        (calibrated, "校准后（Platt scaling）", "#2980b9"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
        ax.plot(mean_pred, frac_pos, "o-", color=color, lw=2, ms=5, label=name)
    ax.set_xlabel("预测违约概率（分箱均值）")
    ax.set_ylabel("实际违约比例")
    ax.set_title("概率可靠性曲线：校准前后对比")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_calibration_curve.png")
    plt.close(fig)
    print("[图] 12_calibration_curve.png")


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    draw_agent_flow()
    draw_tradeoff()
    draw_ablation()
    draw_calibration()
    print("[完成] Agent 图表已生成")
