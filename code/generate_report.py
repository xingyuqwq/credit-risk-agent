# -*- coding: utf-8 -*-
"""生成期末项目报告 DOCX（论文体例：封面、目录、六章正文、参考文献、附录）。"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
TEMPLATE = ROOT / "cover_template_converted.docx"
OUT = ROOT / "028123300_刘星语_信用违约风险审查Agent.docx"

CN_BODY, EN_BODY = "宋体", "Times New Roman"
CN_HEAD, EN_HEAD = "黑体", "Times New Roman"


# --------------------------------------------------------------------------
# 基础排版
# --------------------------------------------------------------------------
def set_run(run, size=12, bold=False, cn=CN_BODY, en=EN_BODY):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = en
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), cn)


def para(doc, text="", size=12, indent=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         line=1.5):
    p = doc.add_paragraph()
    p.alignment = align
    fmt = p.paragraph_format
    fmt.line_spacing = line
    fmt.space_before = Pt(0)
    fmt.space_after = Pt(0)
    fmt.widow_control = True
    if indent:
        fmt.first_line_indent = Pt(size * 2)
    if text:
        set_run(p.add_run(text), size=size)
    return p


def heading(doc, text, level=1):
    p = doc.add_paragraph(text, style=f"Heading {level}")
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    fmt = p.paragraph_format
    fmt.first_line_indent = Pt(0)
    fmt.space_before = Pt(16 if level == 1 else 10 if level == 2 else 8)
    fmt.space_after = Pt(8 if level == 1 else 5 if level == 2 else 4)
    fmt.line_spacing = 1.5
    fmt.keep_with_next = True
    size = {1: 16, 2: 13.5}.get(level, 12)
    for r in p.runs:
        set_run(r, size=size, bold=True, cn=CN_HEAD, en=EN_HEAD)
    return p


def caption(doc, text, space_before=4, keep_next=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.keep_with_next = keep_next
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.first_line_indent = Pt(0)
    set_run(p.add_run(text), size=10.5)
    return p


def figure(doc, filename, cap, width=15.0):
    path = FIG / filename
    if not path.exists():
        para(doc, f"[缺少图片：{filename}]", size=10.5,
             align=WD_ALIGN_PARAGRAPH.CENTER)
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.keep_with_next = True   # 图与图题不得分页
    p.add_run().add_picture(str(path), width=Cm(width))
    caption(doc, cap)


def table(doc, headers, rows, cap, widths=None, font_size=10):
    caption(doc, cap, space_before=6, keep_next=True)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        set_run(p.add_run(h), size=font_size, bold=True, cn=CN_HEAD, en=EN_HEAD)
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), "E8EEF4")
        cell._tc.get_or_add_tcPr().append(shade)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            set_run(p.add_run(str(v)), size=font_size)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Cm(w)
    # 允许表格内部换页会割裂表格，这里整表保持在同一页
    for i, r in enumerate(t.rows):
        trPr = r._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
        if i < len(t.rows) - 1:
            for c in r.cells:
                for cp in c.paragraphs:
                    cp.paragraph_format.keep_with_next = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def code_block(doc, text, size=9):
    for line in text.split("\n"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.left_indent = Pt(12)
        set_run(p.add_run(line if line else " "), size=size,
                cn="Consolas", en="Consolas")


def bullet(doc, text, size=10.5, keep_next=True):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.keep_with_next = keep_next
    p.paragraph_format.line_spacing = 1.4
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.first_line_indent = Pt(0)
    p.paragraph_format.left_indent = Pt(18)
    set_run(p.add_run(text), size=size, cn="Consolas", en="Consolas")
    return p


def ensure_table_grid(doc):
    try:
        doc.styles["Table Grid"]
        return
    except KeyError:
        pass
    src = Document().styles["Table Grid"]
    doc.styles.element.append(deepcopy(src.element))


def ensure_heading_styles(doc):
    """模板由 .doc 转换而来，不含标题样式；从默认文档补入后改成中文论文样式。"""
    existing = [s.name for s in doc.styles]
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        if name not in existing:
            doc.styles.element.append(deepcopy(Document().styles[name].element))
    for name, size in (("Heading 1", 16), ("Heading 2", 13.5), ("Heading 3", 12)):
        st = doc.styles[name]
        st.font.name = EN_HEAD
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        rpr = st.element.get_or_add_rPr()
        rpr.rFonts.set(qn("w:eastAsia"), CN_HEAD)
        rpr.rFonts.set(qn("w:ascii"), EN_HEAD)
        rpr.rFonts.set(qn("w:hAnsi"), EN_HEAD)
        pf = st.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.line_spacing = 1.5
        pf.keep_with_next = True


def set_page_numbering(section, fmt=None, start=None):
    sectPr = section._sectPr
    pg = sectPr.find(qn("w:pgNumType"))
    if pg is None:
        pg = OxmlElement("w:pgNumType")
        cols = sectPr.find(qn("w:cols"))
        if cols is not None:
            cols.addprevious(pg)
        else:
            sectPr.append(pg)
    if fmt:
        pg.set(qn("w:fmt"), fmt)
    if start is not None:
        pg.set(qn("w:start"), str(start))


# --------------------------------------------------------------------------
# 封面与目录
# --------------------------------------------------------------------------
def build_cover(doc):
    """在学院封面模板上填入项目信息。"""
    title_cell = doc.tables[0].cell(0, 1)
    p = title_cell.paragraphs[0]
    for line in ["基于逻辑回归与类别不平衡处理的", "信用卡违约风险审查 Agent"]:
        if p.text.strip():
            p = title_cell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_run(p.add_run(line), size=15, bold=False, cn=CN_BODY)

    info = doc.tables[1]
    filled = {
        "专   业": "计算机科学与技术",
        "学号": "028123300",
        "姓名": "刘星语",
        "课程教师": "邱禧荷",
        "完成日期": "2026 年 11 月 21 日",
    }
    for row in info.rows:
        key = row.cells[0].text.strip()
        if key in filled and filled[key]:
            cell_p = row.cells[1].paragraphs[0]
            set_run(cell_p.add_run(filled[key]), size=14, cn=CN_BODY)

    # 课程要求封面注明组员及分工，个人项目补一行说明；复制末行以沿用模板行高与边框
    last = info.rows[-1]
    new_tr = deepcopy(last._tr)
    last._tr.addprevious(new_tr)
    row = info.rows[len(info.rows) - 2]
    for cell, text in zip(row.cells, ("组    员", "无")):
        p0 = cell.paragraphs[0]
        if p0.runs:
            p0.runs[0].text = text
            for r in p0.runs[1:]:
                r._r.getparent().remove(r._r)
        else:
            set_run(p0.add_run(text), size=14, cn=CN_BODY)

    # 信息表后模板带的空段落会把封面顶到第二页，一并清掉
    body = doc.element.body
    el = info._tbl.getnext()
    while el is not None and el.tag == qn("w:p"):
        nxt = el.getnext()
        body.remove(el)
        el = nxt



def build_toc(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(12)
    set_run(p.add_run("目　　录"), size=18, bold=True, cn=CN_HEAD, en=EN_HEAD)

    # 插入 TOC 域，页码由 Word 生成后固化到文档中
    holder = doc.add_paragraph()
    run = holder.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-2" \\h \\z \\u'
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    tip = OxmlElement("w:t")
    tip.text = "目录将在打开文档时生成"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for el in (begin, instr, sep, tip, end):
        run._r.append(el)


# --------------------------------------------------------------------------
# 正文
# --------------------------------------------------------------------------
def build_ch1(doc):
    heading(doc, "1  引言", 1)

    heading(doc, "1.1  选题背景", 2)
    para(doc, "信用卡发卡机构在审批新卡、提升额度和贷后管理中都要判断客户是否会违约，这一判断直接影响"
              "授信额度、利率与是否放款，属于高影响决策。现有做法分两类：人工审核要同时权衡二十余项"
              "账务字段，效率低且标准不统一；纯模型打分速度快，但只给出一个概率数字，审核员既不清楚"
              "它的来源也不敢采信，遇到客户异议时拿不出可复核的依据。")
    para(doc, "更根本的困难来自样本分布。违约客户通常只占少数，本文所用数据集为 22.12%。在这种分布下，"
              "把全部客户判为不违约的模型也能取得 77.88% 的准确率，却完全丧失风险识别能力。由此产生"
              "三个问题：模型容易退化为全部判为正常；概率无法转化为审核意见；对边界样本强行下结论，"
              "风险远大于交由人工。")

    heading(doc, "1.2  研究目的与意义", 2)
    para(doc, "本文的目标是构建一个面向一线信贷审核员的辅助审查系统。系统输入是含 23 项账务字段的申请件，"
              "输出为结构化结论：违约概率、风险等级、处置建议、五个主要影响因素及其贡献值、命中的业务"
              "规则与风险提示。边界需要明确：本系统是辅助工具而非决策系统，不自动拒绝任何申请，不参与"
              "额度定价，也不替代审核员的最终判断；概率落在边界附近或处于高位时，主动把决定权交还给人。")
    para(doc, "为便于检验，设定五条可量化标准：自动通过样本的实际违约率显著低于 22.12%；转人工样本覆盖"
              "多数真实违约客户；单件延迟低于 50 毫秒；异常输入全部拒答并列出问题；结论给出可核查的"
              "影响因素及方向。围绕这些目标，本文完成四项工作：构建六步审查流程并实现端到端原型；以"
              "逻辑回归为核心完成不平衡处理、概率校准与阈值标定；设计基于系数分解的局部解释与规则库"
              "证据生成；在完整测试集上完成评估、消融与失败案例分析。")
    para(doc, "研究意义有二。实践上，把模型嵌入 Agent 工作流可以缓解上述三对矛盾：模型把多维账务信息"
              "压缩为可排序的风险信号，Agent 负责校验输入、把概率翻译成证据，并在把握不足时把决定权"
              "交还给人，使自动化程度与可控性不必二选一。方法上，本文给出的校准与灰区标定流程不依赖"
              "特定模型，可迁移到其他高影响、低容错的分类场景。")


def build_ch2(doc):
    heading(doc, "2  背景介绍", 1)

    heading(doc, "2.1  国内外研究现状", 2)
    para(doc, "信用评分是机器学习在金融领域应用最成熟的方向之一。早期研究以逻辑回归为主，系数可直接"
              "解释为各变量对对数几率的影响，至今仍是评分卡建模的主流做法[2]。随着集成学习普及，梯度"
              "提升树在公开数据上普遍取得更高 AUC，国内多项工作将 XGBoost 与重采样结合用于信用卡违约"
              "预测，AUC 可提升至 0.77 左右[8][14]；也有研究用图神经网络刻画客户关联[16]，但对数据规模"
              "要求较高。")
    para(doc, "类别不平衡是核心难点。已有综述把处理思路归为数据层、算法层与代价敏感三类[5]：随机欠采样"
              "简单但丢弃多数类信息[7]，SMOTE 及其改进通过插值合成少数类样本，召回率表现更好[5][6]。"
              "重采样改变了训练集先验分布，输出概率会系统性偏移，概率校准通常是必要的后续步骤[1][4]。"
              "在可解释性方面，逻辑回归具备原生可解释性，集成模型依赖 SHAP 等事后归因[9][11]，而国内"
              "综述指出高风险场景中可解释性往往比几个百分点的性能提升更重要[10]。")
    para(doc, "把模型做成可交互系统是近年的另一条线索。大模型与工具调用能力成熟后，Agent 被用于串联"
              "数据校验、模型推演与结果解释等环节，其价值不在模型本身，而在流程控制：明确每一步的输入"
              "边界，并在证据不足时拒绝给出结论。这类系统在金融风控中的公开实现仍以原型为主，缺乏统一"
              "的评估口径，本文的评估设计也参考了上述工作的做法。")

    heading(doc, "2.2  相关技术与算法", 2)
    para(doc, "本文的判别模型选用逻辑回归。它假设违约的对数几率是特征的线性组合，即 ln(p/(1-p)) = "
              "w·x + b，通过最大化对数似然求解参数。其优点是系数方向明确、可加和还原，单条样本的预测"
              "可以拆成各特征的贡献之和，这与本文需要的可核查解释直接契合；缺点是无法自动刻画特征交互，"
              "判别力通常弱于树模型。")
    para(doc, "类别不平衡的处理思路可分为数据层与算法层两类。数据层方法通过重采样改变训练集分布："
              "随机欠采样删除多数类样本，实现简单"
              "但损失信息；SMOTE 对每个少数类样本在其 k 近邻中随机取一点，沿两点连线插值生成新样本，"
              "在不重复原样本的前提下扩大少数类的分布区域。算法层方法则修改学习过程，例如给少数类更高"
              "的错分代价，或直接设置类别权重。三类方法各有适用条件，需在同一测试集上实测比较。")
    para(doc, "重采样后训练集的先验分布已偏离真实人群，模型输出的概率不再等于实际违约频率，"
              "表现为系统性偏高。Platt 缩放用一维逻辑回归在原始分数的对数几率上拟合标签，把数值刻度"
              "拉回真实区间。它是单调变换，不改变样本间排序，因此 AUC 不变而 Brier 分数与可靠性曲线"
              "明显改善，这一点在文献[4]中也有验证。")
    para(doc, "分类器输出的连续概率需经阈值转化为处置动作。医学与风控领域的共识是，"
              "当两类错误的代价不对称且样本存在固有重叠时，单阈值并不合适，更稳妥的做法是在阈值两侧"
              "留出拒绝域，把无法可靠判定的样本交给人工[10]。本文的灰区即按此思路设置。")
    para(doc, "可解释性方面，全局层面可看系数或特征重要性；局部层面对线性模型可直接用系数乘标准化取值，"
              "对树模型则依赖 SHAP 等基于合作博弈的归因方法[9][11]。前者计算代价近似为零且可精确加和，"
              "后者适用范围广但存在近似误差，选择取决于场景对可核查性的要求。")

    heading(doc, "2.3  任务定义与评价指标", 2)
    para(doc, "本文把任务定义为二分类概率估计：给定 23 维匿名化的账务与人口学特征，输出下月违约的概率，"
              "再由决策规则映射为处置动作。评价分两个层次。模型层使用 AUC、精确率、召回率与 F1，其中"
              "AUC 衡量排序能力，不受阈值影响；概率质量另用 Brier 分数与可靠性曲线衡量，前者越小越好。")
    para(doc, "系统层增加四个业务指标：自动通过率、转人工率、自动通过样本的实际违约率、复核覆盖违约"
              "客户的比例。只看模型指标容易得出错误结论——全部转人工可以得到零漏判，却使系统失去意义，"
              "因此必须同时观察自动化程度与风险覆盖。此外记录单件延迟，检验是否满足批量跑批的要求。")


def build_ch3(doc):
    heading(doc, "3  实验过程", 1)

    heading(doc, "3.1  数据集与预处理", 2)
    para(doc, "数据集为 UCI 的 Default of Credit Card Clients（编号 350），是信用违约研究中广泛使用的"
              "公开基准[3]。数据来自中国台湾某银行 2005 年 4 至 9 月的信用卡客户，共 30000 条、23 个"
              "特征，标签为下个月是否违约，已匿名化。特征分五组：授信额度、人口学属性、近六期还款状态、"
              "近六期账单金额与还款金额。数据无缺失，违约率 22.12%。")
    figure(doc, "01_class_distribution.png", "图 1  目标变量类别分布", width=10.5)
    para(doc, "图 1 显示两类样本的规模差异，这是必须做不平衡处理的原因；图 2 显示六期账单金额之间、"
              "还款状态之间相关性较强，提示模型可能受多重共线性影响。预处理采用 Z-score 标准化，只在"
              "训练集拟合参数，避免数据泄露。")
    figure(doc, "03_correlation_heatmap.png", "图 2  特征相关性热力图", width=12.5)

    heading(doc, "3.2  模型设计", 2)
    heading(doc, "3.2.1  训练流程", 3)
    para(doc, "训练集与测试集按 8:2 分层抽样划分，测试集 6000 条全程不参与训练、校准器"
              "拟合与阈值标定。建模管道为：标准化特征经 SMOTE 扩充后送入逻辑回归，随机种子固定为 42。"
              "针对 22.12% 的不平衡分布，本文设置四种策略作为对照：不处理的基线、随机欠采样、SMOTE "
              "过采样与类别权重，四者在同一训练集拟合、同一测试集评估，并做五折交叉验证，结果见 4.1 节。")
    heading(doc, "3.2.2  概率校准", 3)
    para(doc, "SMOTE 把训练集正类比例抬升至 50%，逻辑回归截距被系统性推高，输出概率"
              "不再等于真实违约率。实测未校准模型在测试集上的平均预测违约率为 0.452，真实值仅为 0.221。"
              "若对无逾期记录、额度使用率 24% 的优质客户输出 0.57，审核员会认为模型不可信。因此用 Platt "
              "缩放在五折袋外概率上拟合对数几率到真实标签的映射。校准只改数值刻度不改排序，AUC 不受"
              "影响，但概率含义显著改善：平均预测违约率降至 0.222，与真实值 0.221 一致；Brier 由 0.2085 "
              "降至 0.1473，下降 29.4%。图 3 显示校准后的可靠性曲线更贴合对角线。")
    figure(doc, "12_calibration_curve.png", "图 3  概率可靠性曲线：校准前后对比", width=11.0)
    heading(doc, "3.2.3  阈值标定与灰区划分", 3)
    para(doc, "在训练集五折袋外预测上扫描 181 个候选阈值，取 F1 最大处作为"
              "判别中心 0.315，两侧各扩展 0.15 得到灰区 [0.165, 0.465]。不取 0.5 是因为校准后的概率已"
              "接近真实频率，0.5 意味着违约可能性超过一半，在 22% 的基准下门槛过高。灰区承认模型的能力"
              "边界：落在其中的样本无论判向哪边错误率都接近一半，交给人工更负责任。")
    heading(doc, "3.2.4  模型定位", 3)
    para(doc, "模型只承担一件事：把 23 维特征压缩成可排序的风险信号，不判断输入是否"
              "合法、不解释自身、也不决定是否放行。这种切分使模型可以替换，也使能力边界能够被显式表达。"
              "从系数看，最强的正向因素是最近一期还款状态（0.601），账单金额与还款金额为负，反映高消费"
              "且按时还款的客户资质更好；这是相关性而非因果，不能直接当作业务规则。")

    heading(doc, "3.3  Agent 系统设计", 2)
    para(doc, "系统分三层。交互层接收申请件并输出结论，提供单件、批量、自检三种入口；Agent 层含输入"
              "校验、特征构造、模型推演、证据生成、决策分流五个环节；资源层提供模型工件与规则知识库，"
              "整体流程见图 4。解释能力来自两个互补来源：模型回答违约倾向有多高，规则库回答业务上有"
              "什么值得注意的事实，前者是统计模式，后者是业务常识。")
    figure(doc, "09_agent_flow.png", "图 4  信用违约风险审查 Agent 的工作流程", width=15.5)
    para(doc, "第 1 步输入校验。检查字段是否齐全、是否为数值、是否落在业务范围内。规则按真实分布标定，"
              "账单金额允许为负（溢缴款），教育程度允许为 0（未知）。全部问题一次性返回，不通过则直接"
              "拒答，不进入模型。第 2 步特征构造。计算最长逾期月数、逾期期数、额度使用率、还款占账单"
              "比例、账单环比增长率五个派生指标。")
    para(doc, "第 3 步模型推演。特征经标准化后送入 SMOTE-逻辑回归管道得到原始概率，再由 Platt 校准器"
              "映射为校准概率，只输出数值不做判断。第 4 步证据生成。以系数与标准化取值之积计算各特征"
              "对对数几率的贡献，按绝对值取前五并标注方向，同时检索规则库；模型是线性的，各项贡献可"
              "精确加和还原为依据。第 5 步决策分流。概率低于 0.165 判低风险自动通过；0.165 至 0.465 判"
              "灰区转人工；高于 0.465 判高风险并提请优先核查，后两者均不输出自动结论。第 6 步结果输出。"
              "出具可读结论，附敏感属性提示、高影响决策提示、免责声明与处理耗时。")
    para(doc, "关键取舍有三处。模型上，梯度提升树在本数据集通常能把 AUC 提到 0.77 左右，优于逻辑回归"
              "的 0.71，本文仍选逻辑回归，因为解释的可核查性优先：审核员看到最近一期还款状态贡献 +1.61 "
              "能回溯到模型参数，树模型则须依赖 SHAP 等事后方法。不平衡策略上，欠采样丢弃多数类信息，"
              "类别权重对边界推动有限，SMOTE 兼顾两者且 AUC 与召回率最高，代价是精确率下降，本场景更"
              "怕漏判，可以接受。阈值上不取 0.5，也不取准确率最优点，因为准确率被占比 78% 的多数类主导。")
    para(doc, "系统对五类失败情形做了显式处理，原则是宁可不出结果也不出错误结果：输入异常一律拒答并"
              "列出全部问题；概率落在灰区不给自动结论，只说明不可靠并附证据；概率高于上界也只标记为"
              "提请优先核查；模型工件缺失或损坏时抛异常并提示重新训练；规则库单条格式异常只跳过该条。"
              "所有输出固定附带免责声明，说明结论仅用于风险排序与人工辅助。")

    heading(doc, "3.4  复用组件与许可", 2)
    para(doc, "系统复用的第三方组件（机器学习库、开发工具与公开数据集）见表 1。模型训练、Agent 逻辑、"
              "规则库、评估与报告脚本均为自行编写，第三方库只承担通用算法实现与文件读写。全部组件采用"
              "BSD-3-Clause、MIT 或 PSF 类宽松许可；数据集采用 CC BY 4.0 许可，要求署名。代码与数据中"
              "不含任何密钥或个人信息。")
    rows = [
        ["scikit-learn", "1.2.2", "BSD-3-Clause", "标准化、逻辑回归、交叉验证、评估指标、Platt 校准"],
        ["imbalanced-learn", "0.11.0", "MIT", "SMOTE、RandomUnderSampler、imblearn Pipeline"],
        ["pandas / numpy", "2.1.4 / 1.26.4", "BSD-3-Clause", "数据读取清洗、数值运算"],
        ["matplotlib / seaborn", "3.8.0 / 0.12.2", "BSD 兼容", "全部实验图表与热力图"],
        ["joblib / python-docx", "1.2.0 / 1.2.0", "BSD-3 / MIT", "模型持久化、报告生成"],
        ["ucimlrepo", "0.0.7", "MIT", "通过 UCI 官方接口获取数据集"],
        ["UCI 350", "—", "CC BY 4.0", "训练与评估数据（30000 条，已匿名化）"],
    ]
    table(doc, ["组件", "版本", "许可证", "在本项目中的用途"], rows,
          "表 1  复用组件清单", widths=[3.8, 2.6, 3.0, 6.6], font_size=9.5)
    para(doc, "复用的边界需要说明：本文未修改任何第三方库源码，改动集中在配置层面，包括固定随机种子、"
              "在管道前后增加标准化与校准、按真实分布调整校验范围。自行编写的部分包括六步审查流程、"
              "输入校验与派生特征、含 11 条规则的业务知识库、模型训练与阈值标定、概率校准，以及 16 条"
              "单元测试。")
    para(doc, "与本文效果关系最直接的是 imbalanced-learn 的 SMOTE。它解决的子任务是：少数类样本过少时"
              "分类器边界向多数类偏移，召回率偏低；SMOTE 在少数类样本之间插值合成新样本，扩大其分布"
              "区域，把边界推回中间。输入是特征矩阵与标签，输出是扩充后的样本集，不改变特征维度，默认"
              "近邻数取 5，在本数据集上运行不足 1 秒。其限制主要有四点。第一，对类别型特征不友好：性别、"
              "教育、婚姻是整数编码，会被当作连续值插值，可能合成出性别取值为 1.5 这类无意义样本；本次"
              "实验确实出现过非整数类别特征，因系数较小而影响有限，但正式使用前应先做独热编码。第二，"
              "它不引入新信息。第三，它改变先验分布并导致概率偏移，必须配合校准。第四，高维稀疏下近邻"
              "距离会失去区分意义。")
    para(doc, "该组件不依赖外部服务与网络，除 imbalanced-learn 与 scikit-learn 外无额外依赖，运行时不需要"
              "任何密钥或权限。行为分两步核验。先读官方文档与源码确认调用契约：fit_resample 只作用于传入"
              "的训练子集，"
              "不触碰测试集；sampling_strategy 决定合成比例，random_state 决定近邻抽样；在管道中它位于"
              "标准化之后、分类器之前，预测阶段被自动跳过。再在本地做小规模核对：输入约 2.4 万条训练"
              "样本，重采样后正负样本接近各半，说明它只扩充少数类而没有复制多数类。最后用 4.1 节的四"
              "策略对照检验实际贡献，SMOTE 的召回率最高但 AUC 与其他两种平衡策略几乎相同，说明它改变的"
              "是决策边界位置而非判别能力，这一结论直接决定了后续概率校准的设计。")

    heading(doc, "3.5  实验环境与步骤", 2)
    para(doc, "运行环境为 Python 3.11，依赖见 requirements.txt。在项目根目录依次执行下列命令即可复现"
              "全流程，数据获取只在首次运行时需要，脚本会按顺序产出图表、指标与模型工件。")
    cmds = ["pip install -r requirements.txt              # 安装依赖",
            "python code/fetch_data.py                    # 获取数据集（仅首次）",
            "python code/train_model.py                   # 训练、校准、标定阈值",
            "python code/risk_agent.py --self-test        # 内置自检",
            "python code/risk_agent.py --input examples/sample_batch.json --batch",
            "python code/evaluate_agent.py                # 端到端评估与消融",
            "python code/credit_default.py                # 四策略对比实验",
            "python code/make_figures.py                  # 生成图表",
            "python -m unittest discover -s tests -v      # 单元测试"]
    for i, line in enumerate(cmds):
        bullet(doc, line, keep_next=i < len(cmds) - 1)
    p = para(doc, "目录结构为：code/ 核心代码，data/ 数据集与规则库，models/ 模型工件，figures/ 图表，"
                  "results/ 指标与日志，tests/ 单元测试。examples/ 存放五个测试样例：三个单件样例取自"
                  "测试集，分别对应低风险（预测概率 0.028）、灰区（0.347）与高风险（0.830）三档；一个"
                  "批量文件含四条记录，用于检验批量入口；一个异常件由人工构造，同时触发字段缺失、越界"
                  "与类型错误。实验步骤共四步：先"
                  "统计类别分布与相关性确定预处理方式；再在训练集上比较四种不平衡策略并选定 SMOTE；"
                  "然后在袋外预测上完成校准与阈值标定；最后把模型接入 Agent，在测试集上做端到端评估、"
                  "消融与失败案例分析。随机种子固定为 42，结果可重复。本项目为个人独立完成，无组员"
                  "分工；报告、代码、实验与图表均由本人负责。本文在调试与文字润色环节使用了生成式 AI "
                  "工具，实验数据均由脚本实际运行产生。")
    p.paragraph_format.space_before = Pt(6)
    para(doc, "把通用组件组装成可用系统时遇到四个问题，均已在交付代码中修复。第一个是量纲错配：训练时"
              "模型在标准化特征上拟合，而最初的推理实现把原始值直接送入管道，结果对数几率被推到极负，"
              "所有样本概率塌缩到 0.000。这一缺陷不报错、只是结果全错，相当隐蔽，修复后在测试中断言"
              "概率必须落在开区间内。第二个是校验规则与真实数据不符：最初把账单金额限定为非负，批量"
              "评估时大量样本被拒答，原始数据中确实存在 -1 这类负值。第三个是版本兼容：交叉验证开启多"
              "进程时 threadpoolctl 2.2.0 无法解析当前 BLAS 版本，升级到 3.8.0 后解决。第四个是中文字体："
              "matplotlib 按字体名设置中文常失效，改为用 addfont 注册字体文件，且必须放在样式设置之后。")

    heading(doc, "3.6  实验运行截图", 2)
    figure(doc, "13_demo_run.png", "图 5  完整运行过程（批量样例与异常样例）", width=11.4)
    para(doc, "图 5 是一次完整运行的终端记录，覆盖三个风险档位与一个异常申请件，由演示脚本实际执行产生。"
              "BATCH-001 概率 0.028，自动通过；BATCH-002 概率 0.347，落在灰区，系统要求复核并给出还款"
              "状态 1 推高风险、贡献 +0.542；BATCH-003 概率 0.830，判高风险并命中 6 条规则；DEMO-BAD "
              "同时存在字段缺失、越界与类型错误，系统一次性列出全部问题并拒答。")


def build_ch4(doc):
    heading(doc, "4  结果分析", 1)

    heading(doc, "4.1  类别不平衡处理策略对比", 2)
    para(doc, "四种策略在同一测试集上的指标见表 2，分类指标与 ROC 曲线的直观对比分别见图 6 与图 7。")
    df = pd.read_csv(RES / "strategy_metrics.csv")
    rows = []
    for _, r in df.iterrows():
        rows.append([r["策略"], f"{r['Accuracy']:.4f}", f"{r['Precision']:.4f}",
                     f"{r['Recall']:.4f}", f"{r['F1']:.4f}", f"{r['AUC']:.4f}",
                     f"{r['CV_AUC_mean']:.4f}±{r['CV_AUC_std']:.4f}"])
    table(doc, ["策略", "准确率", "精确率", "召回率", "F1", "AUC", "五折 CV AUC"],
          rows, "表 2  四种类别不平衡处理策略的性能对比（测试集 6000 条）",
          widths=[3.6, 1.7, 1.7, 1.7, 1.6, 1.6, 3.0], font_size=9.5)
    figure(doc, "07_metrics_comparison.png", "图 6  四种策略的分类指标对比", width=13.6)
    figure(doc, "05_roc_comparison.png", "图 7  四种策略的 ROC 曲线", width=9.6)
    para(doc, "基线准确率 0.8078 看似最好，但召回率只有 0.2396，1327 个违约客户中只识别出 318 个，漏判"
              "率 76.0%。三种平衡策略把召回率提到 0.62 左右，提升约 2.6 倍，代价是准确率降至 0.67、"
              "精确率降至 0.36；这不是模型变差，而是它不再把全部判为正常当作最优解。四种策略的 AUC "
              "差异在一个百分点以内，说明是否处理不平衡比用哪种方式处理更重要，与国内研究结论一致[5]。")

    heading(doc, "4.2  端到端分流效果", 2)
    ev = pd.read_csv(RES / "agent_eval.csv")
    a = ev.iloc[0]
    rows = [
        ["违约概率判别力（AUC）", f"{a['AUC']:.4f}"],
        ["准确率 / 精确率 / 召回率 / F1",
         f"{a['Accuracy']:.4f} / {a['Precision']:.4f} / {a['Recall']:.4f} / {a['F1']:.4f}"],
        ["自动通过率", f"{a['自动通过率']:.2%}"],
        ["转人工复核率", f"{a['转人工率']:.2%}"],
        ["高风险标记率（P > 0.465）", f"{a['高风险标记率']:.2%}"],
        ["自动通过样本中的实际违约率", f"{a['自动通过中实际违约率']:.2%}"],
        ["复核覆盖违约客户比例", f"{a['复核覆盖违约客户比例']:.2%}"],
        ["自动放行漏判数（共 1327 个违约）", f"{int(a['自动放行漏判数'])} 件"],
        ["单件延迟（平均 / P95）与吞吐", "0.196 ms / 0.220 ms，约 3900 件/秒"],
    ]
    table(doc, ["指标", "数值"], rows, "表 3  完整系统在测试集上的端到端表现",
          widths=[8.6, 6.4])
    para(doc, "在 6000 条测试样本上逐条调用完整流程，结果见表 3。系统把 40.48% 的样本判为低风险自动通过，这部分"
              "实际违约率仅 13.01%，显著低于总体的 22.12%；其余 59.52% 进入人工流程，覆盖了 76.19% 的"
              "真实违约客户。换算成业务语言：审核员只需处理六成申请，就能圈住七成六的风险客户，而自动"
              "放行的四成中坏客户比例从 22% 降到 13%。单件延迟 0.196 毫秒，远低于 50 毫秒的目标。")

    heading(doc, "4.3  消融实验", 2)
    para(doc, "为判断各组件是否起作用，设计四组消融：A 完整系统，B 关闭灰区分流，C 关闭概率校准，"
              "D 关闭类别平衡，结果见表 4 与图 8。")
    rows = []
    for _, r in ev.iterrows():
        rows.append([r["配置"], f"{r['AUC']:.4f}", f"{r['Recall']:.4f}",
                     f"{r['自动通过率']:.2%}", f"{r['自动通过中实际违约率']:.2%}",
                     f"{r['复核覆盖违约客户比例']:.2%}", f"{int(r['自动放行漏判数'])}"])
    table(doc, ["配置", "AUC", "召回率", "自动通过率", "自动通过中违约率", "复核覆盖率", "漏判数"],
          rows, "表 4  消融实验：各组件的边际贡献",
          widths=[4.6, 1.6, 1.6, 2.0, 2.6, 2.2, 1.6], font_size=9)
    figure(doc, "11_agent_ablation.png", "图 8  消融实验对比", width=15.0)
    para(doc, "A 与 B 的对比最能说明灰区的价值。两者使用同一模型与同一组概率，唯一区别是 B 取消灰区、"
              "全部自动决策：B 漏判 724 件，A 为 316 件，下降 56.4%；代价是 A 有 59.52% 的样本需人工"
              "处理，B 为零。灰区用六成人工换来了四成以上的漏判减少。")
    para(doc, "C 组关闭校准后 AUC 不变、召回率甚至略高，但这是阈值位置不同造成的假象，未校准概率的"
              "标定点为 0.58，与 0.315 不可直接比较。校准的价值在概率的绝对含义：关闭后平均预测违约"
              "率为 0.452，是真实值的两倍。D 组关闭类别平衡后 AUC 降至 0.7076，漏判 257 件虽少于 A，"
              "但自动通过率只有 32.00%，比 A 低 8 个百分点，即它更保守。评估分流系统不能只看漏判数，"
              "否则全部转人工会得到零漏判的伪最优解。")

    heading(doc, "4.4  失败案例分析", 2)
    para(doc, "完整失败清单保存在 results/ 目录下。第一类是信息缺失型漏判：部分违约客户在观察窗口内的"
              "还款记录完全正常，模型给出很低的概率，但客户下月仍然违约；根源不在模型而在数据，数据集"
              "不含收入变化、失业、其他机构负债等变量。")
    para(doc, "第二类是边界挤压型漏判：概率略低于 0.165，恰好落入自动通过区，实际却违约；这类样本含"
              "一两个轻微风险信号，被其他看似正常的特征抵消，本质是 AUC 只有 0.71 的直接后果。第三类"
              "是过度保守型误报：转人工样本中相当一部分并未违约，共同特征是额度较低或年龄较小。这类"
              "误报会增加工作量，长期削弱审核员的信任。")

    heading(doc, "4.5  性能评价与改进方案", 2)
    heading(doc, "4.5.1  优点", 3)
    para(doc, "召回率由 0.24 提升到 0.45，七成六的违约客户被圈进人工流程；每条结论附带"
              "五项特征贡献与命中规则，审核员可直接写入意见；近六成的不确定样本被交还人工。系统的价值"
              "不在于替代审核员，而在于让审核员先看分层结论、再针对性核对两三个因素。概率的绝对含义"
              "也得到恢复，0.22 的平均预测违约率与真实基准一致，使阈值与灰区可以按业务成本设定。")
    heading(doc, "4.5.2  局限与风险", 3)
    para(doc, "最核心的不足是 AUC 只有 0.71，直接导致灰区偏宽、人工复核比例接近"
              "六成，系统的自动化收益被压缩。第二是敏感属性的公平性：模型使用了性别、婚姻、教育、年龄"
              "四个字段，它们在数据中确与违约率相关，但用于信贷决策在许多地区受限；本文在输出层加了"
              "提示，只能算提醒而非解决，真正合规需做公平性审计。第三是数据时效性：数据集来自 2005 年"
              "的特定地区，模型权重未必仍然成立。第四是校准器只适用于当前分布，样本构成变化后需要重估。")
    heading(doc, "4.5.3  改进方案", 3)
    para(doc, "其一，换用梯度提升树[13]，它在这类表格数据上通常能提升五到八个百分点，"
              "代价是失去原生可解释性，需借助 SHAP 做事后解释[9][11]；可行做法是双模型并行，用树模型"
              "排序、用逻辑回归出解释。其二，自适应灰区：引入漏判成本与复核成本，求解使总成本最小的"
              "上下界，替代当前固定半宽的做法。其三，补充特征，引入收入变动、多头负债等外部数据，缓解"
              "信息缺失型漏判。其四，反事实解释，回答怎样做能降低风险，使结论对客户也可沟通。最后需要"
              "说明，本文交付的是可运行的端到端原型而非生产系统，缺少业务系统对接、并发容错、模型版本"
              "管理与线上监控，这些工程问题值得后续补齐。")


def build_refs(doc):
    heading(doc, "参考文献", 1)
    refs = [
        "韦勇凤, 向一波. 基于Group-Lasso方法的非均衡数据信用评分模型[J]. 中国科学院大学学报, 2021, "
        "38(2): 181-188. DOI:10.7523/j.issn.2095-6134.2021.02.004.",
        "张俊丽, 郭双颜, 任翠萍, 等. 基于逻辑回归的个人信用评分卡模型研究[J]. 现代信息科技, 2024, "
        "8(5): 12-16.",
        "王江源. 基于STACKING融合模型的信用贷款违约预测研究——以UCI credit card数据集为例[J]. "
        "产业创新研究, 2023(11): 156-158.",
        "刘雨倩, 衡鑫乐, 陈士县. 融合单调约束与概率校准的信用卡违约预测可复现研究[J]. "
        "阜阳师范大学学报(自然科学版), 2026, 43(2): 27-33, 51. "
        "DOI:10.14096/j.cnki.cn34-1334/n.2026.06.004.",
        "李艳霞, 柴毅, 胡友强, 等. 不平衡数据分类方法综述[J]. 控制与决策, 2019, 34(4): 673-688. "
        "DOI:10.13195/j.kzyjc.2018.0865.",
        "张永清, 卢荣钊, 乔少杰, 等. 一种基于样本空间的类别不平衡数据采样方法[J]. 自动化学报, 2022, "
        "48(10): 2549-2563. DOI:10.16383/j.aas.c200034.",
        "王俊红, 闫家荣. 基于欠采样和代价敏感的不平衡数据分类算法[J]. 计算机应用, 2021, 41(1): 48-52.",
        "田园, 郭红烈, 吉倩. 基于SMOTEENN-XGBoost的信用卡风险客户预测[J]. 软件导刊, 2024, 23(7): "
        "138-143.",
        "陈玉沂, 刘高勇, 蔡焕仪. 个人信贷违约预测机器学习模型的解释性方法研究[J]. 现代计算机, 2024, "
        "30(13): 68-72. DOI:10.3969/j.issn.1007-1423.2024.13.012.",
        "王家祺, 冯毅, 刘华锋, 等. 可解释深度学习的概念建模方法综述[J]. 软件学报, 2026, 37(4): "
        "1591-1614. DOI:10.13328/j.cnki.jos.007530.",
        "雷欣南, 林乐凡, 肖斌卿, 等. 小微企业违约特征再探索: 基于SHAP解释方法的机器学习模型[J]. "
        "中国管理科学, 2024, 32(5): 1-12.",
        "窦慧, 张凌茗, 韩峰, 等. 卷积神经网络的可解释性研究综述[J]. 软件学报, 2024, 35(1): 159-184. "
        "DOI:10.13328/j.cnki.jos.006758.",
        "王朝辉, 高保禄, 刘璇. 基于强化学习和XGBoost的信贷风险预测模型[J]. 计算机工程与设计, 2023, "
        "44(2): 379-385. DOI:10.16208/j.issn1000-7024.2023.02.009.",
        "赵阳, 张杰萌, 严国义. 基于SMOTE-XGBoost算法的信用卡违约预测模型研究[J]. 武汉工程大学学报, "
        "2025, 47(3): 343-348. DOI:10.19843/j.cnki.CN42-1779/TQ.202312031.",
        "赵川, 鞠红梅, 王美玲. 基于机器学习的信贷违约预测研究[J]. 电脑知识与技术, 2023, 19(5): 9-11. "
        "DOI:10.14004/j.cnki.ckt.2023.0254.",
        "唐灵慧, 李林, 李丹. 基于图卷积网络的个人信用评估研究[J]. 计算机与数字工程, 2024, 52(3): "
        "768-774.",
    ]
    for i, r in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.line_spacing = 1.4
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Pt(26)
        p.paragraph_format.first_line_indent = Pt(-26)
        set_run(p.add_run(f"[{i}]  {r}"), size=10.5)


def build_appendix(doc):
    heading(doc, "附录 A  主要代码", 1)
    para(doc, "附录给出核心代码：A.1 为审查 Agent 的决策与分流主流程（摘自 code/risk_agent.py 的 "
              "decide 方法），A.2 为模型训练与阈值标定（摘自 code/train_model.py）。两段代码保留了"
              "关键步骤的注释，运行入口分别为 python code/risk_agent.py 与 python code/train_model.py，"
              "完整运行顺序见 3.5 节。其余模块结构与正文描述一致，完整代码见项目目录。")

    heading(doc, "A.1  决策与分流主流程", 2)
    src = (ROOT / "code" / "risk_agent.py").read_text(encoding="utf-8")
    start = src.find("    def decide(self, payload: dict) -> RiskDecision:")
    end = src.find("    @staticmethod\n    def _render")
    snippet = src[start:end].rstrip() if 0 < start < end else src[:3000]
    code_block(doc, snippet, size=8)

    heading(doc, "A.2  模型训练与阈值标定", 2)
    src2 = (ROOT / "code" / "train_model.py").read_text(encoding="utf-8")
    s2 = src2.find("def calibrate_threshold(")
    e2 = src2.find("def main(")
    snippet2 = src2[s2:e2].rstrip() if 0 < s2 < e2 else ""
    code_block(doc, snippet2, size=8)

    heading(doc, "附录 B  复用组件来源与许可证", 1)
    para(doc, "按课程要求列出全部复用组件的名称、版本、许可证、来源与用途。组件均为公开发布的开源库，"
              "本项目未修改其源码，只在配置层面调用；数据集为匿名化的公开数据，采用 CC BY 4.0 许可，"
              "使用时保留原始出处署名。")
    rows = [
        ["scikit-learn", "1.2.2", "BSD-3-Clause", "scikit-learn.org",
         "标准化、逻辑回归、交叉验证、评估指标、Platt 校准"],
        ["imbalanced-learn", "0.11.0", "MIT", "imbalanced-learn.org",
         "SMOTE、RandomUnderSampler、imblearn Pipeline"],
        ["pandas", "2.1.4", "BSD-3-Clause", "pandas.pydata.org",
         "数据读取、清洗与结果汇总"],
        ["numpy", "1.26.4", "BSD-3-Clause", "numpy.org", "数值运算与数组操作"],
        ["matplotlib", "3.8.0", "PSF-based", "matplotlib.org", "全部实验图表绘制"],
        ["seaborn", "0.12.2", "BSD-3-Clause", "seaborn.pydata.org",
         "热力图与图表样式"],
        ["joblib", "1.2.0", "BSD-3-Clause", "joblib.readthedocs.io",
         "模型工件持久化"],
        ["python-docx", "1.2.0", "MIT", "python-docx.readthedocs.io",
         "报告生成与排版"],
        ["ucimlrepo", "0.0.7", "MIT", "github.com/uci-ml-repo/ucimlrepo",
         "通过 UCI 官方接口获取数据集"],
        ["UCI 350", "—", "CC BY 4.0",
         "archive.ics.uci.edu/dataset/350",
         "训练与评估数据（30000 条，已匿名化）"],
    ]
    table(doc, ["组件", "版本", "许可证", "来源", "用途"], rows,
          "表 5  复用组件的来源与许可证", widths=[2.9, 1.5, 2.3, 5.4, 3.4], font_size=8)


def finalize_with_word(path: Path, pdf: Path | None = None) -> dict:
    """用 Word 更新目录域并固化页码，可选导出 PDF。失败时静默跳过。"""
    info = {}
    try:
        import pythoncom
        import win32com.client as win32
    except ImportError:
        return info
    pythoncom.CoInitialize()
    app = None
    try:
        app = win32.DispatchEx("Word.Application")
        app.Visible = False
        app.DisplayAlerts = 0
        d = app.Documents.Open(str(path))
        d.Fields.Update()
        for toc in d.TablesOfContents:
            toc.Update()
        d.Repaginate()
        info["pages"] = d.ComputeStatistics(2)
        info["words"] = d.ComputeStatistics(0)
        d.Save()
        if pdf is not None:
            d.ExportAsFixedFormat(str(pdf), 17)
            info["pdf"] = str(pdf)
        d.Close(0)
    except Exception as exc:  # noqa: BLE001
        info["error"] = repr(exc)
    finally:
        if app is not None:
            try:
                app.Quit()
            except Exception:  # noqa: BLE001
                pass
        pythoncom.CoUninitialize()
    return info


def main():
    doc = Document(TEMPLATE)
    ensure_table_grid(doc)
    ensure_heading_styles(doc)

    sec = doc.sections[0]
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.5)
    sec.left_margin = Cm(3.0)
    sec.right_margin = Cm(2.5)

    style = doc.styles["Normal"]
    style.font.name = EN_BODY
    style.font.size = Pt(12)
    style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), CN_BODY)

    build_cover(doc)
    # 封面与目录同属前置部分：小写罗马页码，封面本身不显示页码
    # 分页段落用 1 磅字号，避免它在封面填满时被挤到下一页而多出空白页
    brk = doc.add_paragraph()
    brk.paragraph_format.space_before = Pt(0)
    brk.paragraph_format.space_after = Pt(0)
    brk.paragraph_format.line_spacing = 1.0
    brk_run = brk.add_run()
    brk_run.font.size = Pt(1)
    brk_run.add_break(WD_BREAK.PAGE)
    build_toc(doc)
    # 正文另起一节，阿拉伯页码从 1 开始
    doc.add_section(WD_SECTION.NEW_PAGE)

    build_ch1(doc)
    build_ch2(doc)
    build_ch3(doc)
    build_ch4(doc)
    build_refs(doc)
    build_appendix(doc)

    front, body = doc.sections[0], doc.sections[1]
    front.different_first_page_header_footer = True
    set_page_numbering(front, fmt="lowerRoman", start=0)
    body.different_first_page_header_footer = False
    set_page_numbering(body, fmt="decimal", start=1)

    doc.save(OUT)

    txt = "\n".join(p.text for p in doc.paragraphs)
    cn = len(re.findall(r"[\u4e00-\u9fff]", txt))
    print(f"[报告] 已生成：{OUT}")
    print(f"[报告] 中文字数 {cn}，段落 {len(doc.paragraphs)}，"
          f"图片 {len(doc.inline_shapes)}，表格 {len(doc.tables)}")

    pdf = OUT.with_suffix(".pdf")
    info = finalize_with_word(OUT, pdf)
    print(f"[报告] Word 处理结果：{info}")


if __name__ == "__main__":
    main()
