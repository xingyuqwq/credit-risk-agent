# -*- coding: utf-8 -*-
"""把 results/demo_run.log 渲染成终端风格图片，只做排版，不改写内容。"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "demo_run.log"
OUT = ROOT / "figures" / "13_demo_run.png"


def display_width(s: str) -> int:
    """按终端习惯计算显示宽度：全角字符占 2 列。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def setup_fonts() -> list[str]:
    """注册等宽英文字体与中文字体，返回 family 列表供 fallback。"""
    families = []
    for fname, alias in [("consola.ttf", "Consolas"), ("simhei.ttf", "SimHei")]:
        path = Path(r"C:\Windows\Fonts") / fname
        if path.exists():
            fm.fontManager.addfont(str(path))
            families.append(alias)
    if not families:
        families = ["DejaVu Sans"]
    return families


def main() -> None:
    text = LOG.read_text(encoding="utf-8").rstrip("\n")
    lines = text.split("\n")
    max_w = max(display_width(ln) for ln in lines)

    families = setup_fonts()
    plt.rcParams["font.family"] = families

    # 按字号与行距精确换算画布尺寸，避免文字溢出画布底部
    fs, ls = 7.2, 1.35
    char_w = fs * 0.55 / 72      # Consolas 单字符宽度约 0.55 em
    line_h = fs * ls / 72
    fig_w = max_w * char_w + 0.34
    fig_h = len(lines) * line_h + 0.26

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1b1b1b")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#1b1b1b")
    ax.axis("off")

    # 顶栏：模拟终端标题条
    ax.add_patch(plt.Rectangle((0, 0.986), 1, 0.014, color="#2d2d2d",
                               transform=ax.transAxes, clip_on=False))

    ax.text(0.008, 0.976, text, fontsize=fs, color="#d6d6d6",
            va="top", ha="left", linespacing=ls, family=families,
            transform=ax.transAxes)

    fig.savefig(OUT, dpi=190, facecolor="#1b1b1b")
    plt.close(fig)
    print(f"[图] {OUT}  ({max_w} 列 x {len(lines)} 行, {fig_w:.2f}x{fig_h:.2f} in)")


if __name__ == "__main__":
    main()
