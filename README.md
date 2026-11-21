# 信用卡违约风险审查 Agent

围绕人工授信复核场景构建的端到端 Agent 原型：用逻辑回归与类别不平衡处理给出违约概率，
把概率翻译成可核查的风险证据，并在置信度不足时主动交还给人。

**本系统不做自动拒贷。** 它的定位是给信贷审核员排序和解释，最终决定权仍在人。

课程：机器学习（2026 秋）　学号：028123300　姓名：刘星语

---

## 1. 提交材料

| 文件 | 内容 |
|---|---|
| `028123300_刘星语_信用违约风险审查Agent.pdf` | 课程报告，含封面、目录、正文四章、参考文献 16 篇、附录 A 主要代码、附录 B 组件清单 |
| `028123300_刘星语_信用违约风险审查Agent_复用组件清单与许可证.pdf` | 组件版本、许可证、来源与用途的单页说明 |
| `028123300_刘星语_信用违约风险审查Agent_代码与演示.zip` | 源码、数据、模型工件、测试样例、图表与实验结果 |
| `运行演示截图.png` | 一次完整运行的控制台输出（批量样例与异常样例） |

报告正文约 6600 中文字（不含图表、代码与参考文献），24 页。

## 2. 目录结构

```
code/            核心代码
  fetch_data.py      从 UCI 获取数据集（一次性）
  train_model.py     训练 SMOTE-逻辑回归、Platt 校准、阈值标定
  risk_agent.py      Agent 核心与命令行入口
  evaluate_agent.py  端到端评估与消融实验
  credit_default.py  模型层对比实验（四种不平衡策略）
  make_figures.py    流程图、权衡曲线、消融图、校准曲线
  demo_run.py        完整运行演示，写出真实日志
  render_terminal.py 把运行日志渲染为终端截图
  generate_report.py 生成报告 DOCX 与 PDF
data/
  UCI_Credit_Card.csv   30000 条样本，23 个特征
  knowledge_rules.json  11 条业务规则
models/
  credit_risk_model.joblib  模型工件（scaler + pipeline + 校准器 + 阈值）
examples/        五个测试样例（低风险 / 灰区 / 高风险 / 批量 / 异常）
tests/           单元测试，16 条
figures/         实验图表
results/         指标、日志、失败案例
```

## 3. 环境与依赖

Python 3.11，依赖见 `requirements.txt`：

```bash
pip install -r requirements.txt
```

主要依赖为 scikit-learn 1.2.2、imbalanced-learn 0.11.0、pandas 2.1.4、numpy 1.26.4、
matplotlib 3.8.0、seaborn 0.12.2、joblib 1.2.0、python-docx 1.2.0。

仅运行 Agent 推理时，`models/credit_risk_model.joblib` 已随包提供，无需重新训练。

## 4. 快速开始

```bash
# 1) 首次运行：获取数据（包内已含 CSV，可跳过）
python code/fetch_data.py

# 2) 训练模型与阈值标定（约 20 秒）
python code/train_model.py

# 3) 跑单个申请件
python code/risk_agent.py --input examples/sample_grey_zone.json

# 4) 跑批量
python code/risk_agent.py --input examples/sample_batch.json --batch

# 5) 异常输入会拒答并列出问题
python code/risk_agent.py --input examples/sample_invalid.json

# 6) 内置自检（低 / 高 / 异常三个用例）
python code/risk_agent.py --self-test
```

## 5. 复现报告中的结果

按下列顺序执行可完整复现报告中的全部图表与指标，每步会覆盖 `results/` 与 `figures/`
下的对应文件：

| 步骤 | 产出 |
|---|---|
| `python code/credit_default.py` | 表 2，图 1、图 2、图 6、图 7 |
| `python code/train_model.py` | 模型工件、`results/model_meta.json` |
| `python code/evaluate_agent.py` | 表 3、表 4 的数据，失败案例 CSV |
| `python code/make_figures.py` | 图 3 校准曲线、图 4 流程图、图 8 消融图 |
| `python code/demo_run.py` | `results/demo_run.log`（图 5 的来源） |
| `python code/render_terminal.py` | 图 5 终端截图 |
| `python -m unittest discover -s tests -v` | 16 条单元测试 |
| `python code/generate_report.py` | 重新生成报告 DOCX 与 PDF |

`generate_report.py` 以学院封面模板 `cover_template_converted.docx` 为底稿，该模板
不在提交包内，仅重新生成报告时需要（可由学院提供的 `.doc` 用 Word 另存为 `.docx`）。
只复现实验数据无需运行此脚本。

## 6. 输入与输出

单个申请件的输入格式：

```json
{
  "request_id": "C-0001",
  "applicant": {
    "LIMIT_BAL": 200000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 48000, "BILL_AMT2": 45000, "BILL_AMT3": 42000,
    "BILL_AMT4": 40000, "BILL_AMT5": 38000, "BILL_AMT6": 35000,
    "PAY_AMT1": 48000, "PAY_AMT2": 45000, "PAY_AMT3": 42000,
    "PAY_AMT4": 40000, "PAY_AMT5": 38000, "PAY_AMT6": 35000
  }
}
```

23 个字段必须齐全且为数值。批量模式用 `{"records": [...]}` 包裹。

每条结果返回：校准后的违约概率、风险等级、处置建议、主要影响因素（系数贡献）、
规则命中、敏感属性提示、处理耗时。`--json` 输出结构化结果，`--output file.json` 落盘。

## 7. 决策规则

| 区间 | 含义 | 处置 |
|---|---|---|
| P < 0.165 | 低风险 | 自动通过，建议维持现有策略 |
| 0.165 ≤ P ≤ 0.465 | 灰区 | 转人工复核，不给自动结论 |
| P > 0.465 | 高风险 | 提请人工优先核查，**不自动拒绝** |

阈值由训练集五折袋外预测按 F1 最优标定，灰区半宽 0.15。

## 8. 数据与许可

* 数据集：UCI Machine Learning Repository, *Default of Credit Card Clients*（ID 350），
  许可证 CC BY 4.0，DOI 10.24432/C5XK51。已匿名化，不含真实身份信息。
* 复用组件的版本与许可证见报告中表 1、附录 B，以及单独提交的组件清单 PDF。
* 本仓库不含任何密钥、账号口令或个人信息。
* 模型输入包含性别、婚姻、教育、年龄等敏感字段（原数据集如此）。Agent 命中这些因素时
  会给出显式提示，避免其成为拒绝授信的唯一依据；正式生产环境应做公平性审计或直接剔除。

## 9. 已知局限

* 测试集 AUC 约 0.71，属中等判别力，因此灰区较宽、人工复核比例偏高（约 60%）。
* 概率校准显著改善了绝对数值（平均预测率 0.452 → 0.222，真实值 0.221），但高概率区间
  样本稀疏，可靠性曲线仍有波动。
* 规则知识库为人工整理，覆盖 11 条常见情形，未覆盖的边界情况依赖模型与人工。
* 未做公平性量化审计，仅在输出层做了提示。
