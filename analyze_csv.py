#!/usr/bin/env python3
"""
input_analysis 内のCSVをまとめて読み込み、前処理後に2種類の図を保存:
1) Scatter: Price vs Status
2) Box + Points: Brand vs Status (sorted by mean, seaborn style)

Outputs:
- output_analysis/scatter_status_vs_price.png
- output_analysis/box_status_vs_brand.png
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Utility ----------
def load_csvs_from_input_folder(folder: str = "input_analysis", header_row: int = 1) -> pd.DataFrame:
    p = Path(folder)
    files = sorted(p.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"{folder} にCSVファイルが見つかりません。")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, header=header_row)
        except UnicodeDecodeError:
            df = None
            for enc in ("utf-8", "utf-8-sig", "cp932"):
                try:
                    df = pd.read_csv(f, header=header_row, encoding=enc)
                    break
                except Exception:
                    df = None
            if df is None:
                raise
        df["__source_file"] = f.name
        dfs.append(df)

    cat = pd.concat(dfs, ignore_index=True)
    cat.columns = [str(c).strip() for c in cat.columns]
    return cat


def to_numeric_clean(s: pd.Series) -> pd.Series:
    return (
        s.astype(str).str.strip()
        .str.replace(r"[\s,\u00A0\u3000¥￥]", "", regex=True)
        .str.replace("%", "", regex=False)
        .replace({"nan": None, "": None})
        .astype(float)
    )


def find_price_column(columns: list[str]) -> str | None:
    for c in columns:
        if c.lower() == "price":
            return c
    for cand in ("価格", "金額", "販売価格", "単価", "金額（税込）"):
        if cand in columns:
            return cand
    return None

# ---------- Main ----------
def main():
    # 見た目（Seabornテーマ）
    sns.set_theme(style="whitegrid", context="talk")
    # 日本語を使う場合はフォント指定を有効化（英語のみなら不要）
    # plt.rcParams["font.family"] = "IPAexGothic"  # Mac: 'Hiragino Sans', Win: 'MS Gothic' など

    df = load_csvs_from_input_folder()

    # 必須列チェック
    if "作業者" not in df.columns or "ステータス" not in df.columns:
        raise ValueError("CSVに '作業者' または 'ステータス' 列がありません。")

    price_col = find_price_column(list(df.columns))
    if not price_col:
        raise ValueError("'price' または '価格' 列が見つかりません。")

    # 前処理
    df = df[df["作業者"].astype(str).str.strip().ne("")]
    df["__status_num"] = to_numeric_clean(df["ステータス"])
    df = df[df["__status_num"] != -100]
    df["__price_num"] = to_numeric_clean(df[price_col])

    # 出力先
    out = Path("output_analysis")
    out.mkdir(exist_ok=True)

    # ---------- 1) Scatter: Price vs Status ----------
    plot_df = df.dropna(subset=["__status_num", "__price_num"])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sns.scatterplot(
        data=plot_df,
        x="__price_num", y="__status_num",
        alpha=0.7, edgecolor=None, ax=ax
    )
    ax.set_xlabel("Price")
    ax.set_ylabel("Status (%)")
    ax.set_title("Status vs Price")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out / "scatter_status_vs_price.png", dpi=150)
    plt.close(fig)

    # ---------- 2) Box + Points: Brand vs Status (mean-sorted) ----------
    brand_col = None
    for cand in ("brand", "Brand", "ブランド", "メーカー"):
        if cand in df.columns:
            brand_col = cand
            break

    if brand_col:
        bdf = df.dropna(subset=["__status_num", brand_col]).copy()
        # 並べ替え順（平均の降順）
        order = (
            bdf.groupby(brand_col)["__status_num"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

        fig, ax = plt.subplots(figsize=(11, 6.5))
        # 箱ひげ図（外れ値マーカーは控えめに）
        sns.boxplot(
            data=bdf, x=brand_col, y="__status_num",
            order=order,
            color="lightgray",
            width=0.6,
            fliersize=2,
            linewidth=1.2,
            ax=ax
        )
        # 個別点を重ねる（横に微 jitter）
        # サンプル数がとても多い場合は dodge=False の stripplot が安全
        sns.stripplot(
            data=bdf, x=brand_col, y="__status_num",
            order=order,
            alpha=0.55, size=3.5, jitter=0.25,
            edgecolor="gray", linewidth=0.3,
            ax=ax
        )

        # 平均値の赤い点も重ねる（視認性UP）
        means = bdf.groupby(brand_col)["__status_num"].mean().reindex(order)
        ax.scatter(
            x=np.arange(len(order)),
            y=means.values,
            s=70, marker="D", color="red", zorder=5, label="Mean"
        )

        ax.set_xlabel("Brand")
        ax.set_ylabel("Status (%)")
        ax.set_title("Status Distribution by Brand (sorted by mean)")
        ax.legend(frameon=False, loc="upper right")
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=6)
        sns.despine()
        fig.tight_layout()
        fig.savefig(out / "box_status_vs_brand.png", dpi=150)
        plt.close(fig)
    else:
        print("[WARN] 'brand' or 'ブランド' column not found. Boxplot skipped.")

    print("[INFO] Saved: scatter_status_vs_price.png, box_status_vs_brand.png")


if __name__ == "__main__":
    main()
