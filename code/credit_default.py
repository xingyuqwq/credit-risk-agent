# -*- coding: utf-8 -*-
"""信用违约预测（高级难度）：逻辑回归 + 类别不平衡处理 + ROC/AUC 对比。

对比四种策略（不处理、随机欠采样、SMOTE、类别权重）在同一测试集上的表现，
输出探索性图表、混淆矩阵、ROC/PR 曲线与指标对比，结果写入 results/。
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无界面后端，保证在服务器上也能出图
import matplotlib.pyplot as plt
from matplotlib import font_manager

import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, roc_auc_score, classification_report,
    precision_recall_curve, average_precision_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 路径：以脚本所在目录的上级作为项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR = os.path.join(BASE_DIR, "figures")
RESULT_DIR = os.path.join(BASE_DIR, "results")
for d in (FIG_DIR, RESULT_DIR):
    os.makedirs(d, exist_ok=True)

# 中文字体：按字体名设置 rcParams 常失效，改为注册字体文件
def setup_chinese_font():
    font_files = [
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
    ]
    for fp in font_files:
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
                name = font_manager.FontProperties(fname=fp).get_name()
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["font.family"] = "sans-serif"
                print(f"[字体] 已加载中文字体：{name}（{fp}）")
                break
            except Exception as e:
                print(f"[字体] 加载 {fp} 失败：{e}")
    else:
        print("[警告] 未找到中文字体，图表中文可能显示为方框")
    plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

sns.set_style("whitegrid")
setup_chinese_font()  # 必须放在 set_style 之后，否则字体会被重置
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# 数据加载：UCI 原始的 .xls 实为 HTML 表格，优先读 CSV，其余格式兜底
def load_data():
    import glob
    xls_files = glob.glob(os.path.join(DATA_DIR, "*.xls"))
    xlsx_files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

    if csv_files:
        df = pd.read_csv(csv_files[0])
        print(f"[数据] 读取 CSV：{os.path.basename(csv_files[0])}")
        return clean_columns(df)

    target = (xlsx_files or xls_files)[0]
    print(f"[数据] 读取文件：{os.path.basename(target)}")
    try:
        df = pd.read_html(target)[0]
        # 原始 xls 第 1 行是占位表头，第 2 行才是真实列名
        if str(df.iloc[0, 0]).strip().upper().startswith("X") and "ID" in list(df.iloc[0]):
            header = df.iloc[0].tolist()
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = header
    except Exception as e:
        print(f"[数据] read_html 失败({e})，尝试 read_excel")
        df = pd.read_excel(target)
    return clean_columns(df)


def clean_columns(df):
    """统一列名、删除无意义 ID 列、处理重复列名。"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df.columns = [c.replace("\n", "").replace(" ", "_") for c in df.columns]
    y_candidates = [c for c in df.columns if "default" in c.lower() or c.upper() == "Y"]
    if y_candidates:
        df.rename(columns={y_candidates[0]: "default"}, inplace=True)
    if "ID" in df.columns:
        df.drop(columns=["ID"], inplace=True)
    return df


# 数据探索：类别分布与关键特征分布
def explore(df):
    print("\n" + "=" * 60)
    print("【2. 数据探索】")
    print("=" * 60)
    print(f"样本数：{df.shape[0]}，特征数：{df.shape[1]}")
    print(f"缺失值总数：{int(df.isnull().sum().sum())}")
    print(df["default"].value_counts().rename({0: "未违约", 1: "违约"}))
    ratio = df["default"].mean()
    print(f"违约率：{ratio:.2%}（正样本占比，体现类别不平衡）")

    fig, ax = plt.subplots(figsize=(6, 4))
    cnt = df["default"].value_counts().sort_index()
    bars = ax.bar(["未违约 (0)", "违约 (1)"], cnt.values, color=["#4C72B0", "#C44E52"])
    for b, v in zip(bars, cnt.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 100, f"{v}\n({v/len(df):.1%})",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("目标变量类别分布（类别不平衡）")
    ax.set_ylabel("样本数量")
    ax.set_ylim(0, max(cnt.values) * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_class_distribution.png"))
    plt.close(fig)

    if "LIMIT_BAL" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, color, name in [(0, "#4C72B0", "未违约"), (1, "#C44E52", "违约")]:
            sns.histplot(df.loc[df.default == label, "LIMIT_BAL"],
                         bins=30, color=color, alpha=0.55, label=name, ax=ax)
        ax.set_xlabel("授信额度 (新台币)")
        ax.set_ylabel("频数")
        ax.set_title("授信额度在两类客户中的分布")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "02_limit_bal_distribution.png"))
        plt.close(fig)
    return ratio


# 相关性热力图
def correlation_heatmap(df):
    num_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 9))
    corr = num_df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.8}, linewidths=0.3)
    plt.title("特征相关性热力图")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_correlation_heatmap.png"))
    plt.close()
    print("[探索] 相关性热力图已保存")


# 分类指标
def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
    }


def main():
    print("=" * 60)
    print("信用违约预测（高级难度）—— 逻辑回归 + 不平衡处理 + ROC/AUC 对比")
    print("=" * 60)

    df = load_data()
    print("\n[数据前 5 行]")
    print(df.head())
    print("\n[数据类型]")
    print(df.dtypes)

    ratio = explore(df)
    correlation_heatmap(df)

    # 5.1 分层抽样划分，保持两类比例一致
    X = df.drop(columns=["default"])
    y = df["default"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n[划分] 训练集 {X_train.shape[0]} 条，测试集 {X_test.shape[0]} 条")
    print(f"[划分] 训练集违约率 {y_train.mean():.2%}，测试集违约率 {y_test.mean():.2%}")

    # 5.2 标准化，只在训练集上拟合
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_train_s = pd.DataFrame(X_train_s, columns=X.columns)
    X_test_s = pd.DataFrame(X_test_s, columns=X.columns)

    # 5.3 四种策略，统一在同一测试集上评估
    strategies = {}

    # 基线：不做任何处理
    strategies["基线(不处理)"] = LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE
    )

    # 随机欠采样
    strategies["随机欠采样"] = ImbPipeline([
        ("under", RandomUnderSampler(random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])

    # SMOTE 过采样
    strategies["SMOTE过采样"] = ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])

    # 类别权重
    strategies["类别权重(balanced)"] = LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
    )

    # 5.4 训练、预测、评估
    results = []
    roc_data = {}
    pr_data = {}
    cm_fig, cm_axes = plt.subplots(2, 2, figsize=(11, 9))
    cm_axes = cm_axes.ravel()

    for idx, (name, model) in enumerate(strategies.items()):
        print("\n" + "-" * 60)
        print(f"策略：{name}")
        print("-" * 60)

        # 交叉验证只在训练集上做
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        # n_jobs=1：多进程下 threadpoolctl 读不到 BLAS 版本会报错
        cv_auc = cross_val_score(
            model, X_train_s, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
        print(f"5折交叉验证 AUC：{cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)[:, 1]

        metrics = evaluate(y_test, y_pred, y_prob)
        metrics["CV_AUC_mean"] = cv_auc.mean()
        metrics["CV_AUC_std"] = cv_auc.std()
        results.append({"策略": name, **metrics})
        print(f"测试集指标：{metrics}")

        # 曲线数据留待统一作图
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, metrics["AUC"])
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_data[name] = (precision, recall, average_precision_score(y_test, y_prob))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["预测未违约", "预测违约"],
                    yticklabels=["实际未违约", "实际违约"], ax=cm_axes[idx])
        cm_axes[idx].set_title(f"{name}\nRecall={metrics['Recall']:.3f}  AUC={metrics['AUC']:.3f}")
    cm_fig.suptitle("四种策略的混淆矩阵对比", fontsize=14)
    cm_fig.tight_layout()
    cm_fig.savefig(os.path.join(FIG_DIR, "04_confusion_matrices.png"))
    plt.close(cm_fig)

    # 5.5 ROC 曲线对比
    colors = {"基线(不处理)": "#7f8c8d", "随机欠采样": "#e67e22",
              "SMOTE过采样": "#2980b9", "类别权重(balanced)": "#c0392b"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc_val) in roc_data.items():
        ax.plot(fpr, tpr, lw=2, color=colors.get(name),
                label=f"{name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="随机猜测")
    ax.set_xlabel("假正例率 (FPR)")
    ax.set_ylabel("真正例率 (TPR)")
    ax.set_title("ROC 曲线对比（不同不平衡处理策略）")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "05_roc_comparison.png"))
    plt.close(fig)

    # 5.6 PR 曲线对比
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (precision, recall, ap) in pr_data.items():
        ax.plot(recall, precision, lw=2, color=colors.get(name),
                label=f"{name} (AP = {ap:.3f})")
    ax.set_xlabel("召回率 (Recall)")
    ax.set_ylabel("精确率 (Precision)")
    ax.set_title("PR 曲线对比（不平衡数据下更有参考价值）")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "06_pr_comparison.png"))
    plt.close(fig)

    # 5.7 指标对比
    results_df = pd.DataFrame(results).set_index("策略")
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    plot_df = results_df[metrics_to_plot]
    ax = plot_df.plot(kind="bar", figsize=(11, 5), rot=15,
                      color=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("指标值")
    ax.set_title("不同策略的分类性能指标对比")
    ax.legend(loc="upper right", ncol=5, fontsize=9)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "07_metrics_comparison.png"))
    plt.close(fig)

    # 5.8 最优策略的分类报告
    best_name = results_df["AUC"].idxmax()
    print("\n" + "=" * 60)
    print(f"最优策略（按 AUC）：{best_name}")
    print("=" * 60)
    best_model = strategies[best_name]
    y_pred_best = best_model.predict(X_test_s)
    print(classification_report(y_test, y_pred_best,
                                target_names=["未违约", "违约"], digits=4))

    # 5.9 基线模型的特征系数
    base_model = strategies["基线(不处理)"]
    coef = pd.Series(base_model.coef_[0], index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    colors_bar = ["#C44E52" if v > 0 else "#4C72B0" for v in coef.values]
    coef.plot(kind="barh", color=colors_bar, ax=ax)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("逻辑回归特征系数（基线模型，标准化后）")
    ax.set_xlabel("系数值（正=增加违约概率，负=降低违约概率）")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "08_feature_coefficients.png"))
    plt.close(fig)

    # 5.10 保存结果
    results_df.round(4).to_csv(os.path.join(RESULT_DIR, "strategy_metrics.csv"),
                               encoding="utf-8-sig")
    with open(os.path.join(RESULT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("信用违约预测（高级难度）实验结果汇总\n")
        f.write("=" * 60 + "\n")
        f.write(f"数据集：UCI Default of Credit Card Clients\n")
        f.write(f"样本总数：{len(df)}，特征数：{X.shape[1]}\n")
        f.write(f"违约率：{ratio:.2%}\n")
        f.write(f"训练集/测试集：{len(X_train)}/{len(X_test)}\n\n")
        f.write(results_df.round(4).to_string())
        f.write(f"\n\n最优策略（按 AUC）：{best_name}\n")
        f.write("\n详细分类报告：\n")
        f.write(classification_report(y_test, y_pred_best,
                                      target_names=["未违约", "违约"], digits=4))

    print("\n[完成] 所有图表已保存到 figures/，结果已保存到 results/")
    print(results_df.round(4))


if __name__ == "__main__":
    main()
