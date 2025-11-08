#!/usr/bin/env python3
"""
input_analysis 内のCSVをまとめて読み込み、前処理後に3種類の図を保存:
1) Scatter: Price vs Status
2) Box + Points: Brand vs Status (sorted by mean, seaborn style)
3) Box + Points: Price(¥1,000 bins) vs Status + bin means

Outputs:
- output_analysis/scatter_status_vs_price.png
- output_analysis/box_status_vs_brand.png
- output_analysis/box_status_vs_price_bins.png
- output_analysis/processed_data.csv
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams, font_manager

# ---------- Font (Japanese-safe) ----------
def enable_japanese_font(verbose: bool = True) -> None:
    """日本語グリフを含むフォントを優先指定し、豆腐(□)化を防ぐ。"""
    candidates = [
        "Hiragino Sans", "Hiragino Kaku Gothic ProN",  # macOS
        "Yu Gothic", "Meiryo",                        # Windows
        "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic", "TakaoGothic"  # OSS
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in candidates if n in installed), None)
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = candidates
    rcParams["axes.unicode_minus"] = False
    if verbose:
        print(f"[INFO] Japanese font chosen: {chosen or 'fallback from list'}")

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
    sns.set_theme(style="whitegrid", context="talk")
    enable_japanese_font(verbose=True)

    df = load_csvs_from_input_folder()

    # 必須列チェック
    if "作業者" not in df.columns or "利益割合" not in df.columns:
        raise ValueError("CSVに '作業者' または '利益割合' 列がありません。")

    price_col = find_price_column(list(df.columns))
    if not price_col:
        raise ValueError("'price' または '価格' 列が見つかりません。")

    # 前処理
    df = df[df["作業者"].astype(str).str.strip().ne("")]
    df["__status_num"] = (
        pd.to_numeric(
            df["利益割合"]
              .astype(str)
              .str.strip()
              .str.replace(",", "", regex=False)
              .str.replace("%", "", regex=False),
            errors="coerce"
        )
        .where(df["利益割合"].astype(str) != "#DIV/0!", np.nan)
    )

    exclude_conditions = ["中古C", "ジャンク", "Broken"]
    if "condition" in df.columns:
        df = df[~df["condition"].isin(exclude_conditions)].copy()

    df = df[df["__status_num"] != -100]
    df["__price_num"] = to_numeric_clean(df[price_col])

    # 出力先フォルダ
    out = Path("output_analysis")
    out.mkdir(exist_ok=True)

    # ---------- CSV保存 ----------
    csv_path = out / "processed_data.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved cleaned dataframe → {csv_path}")

    # ---------- 1) Scatter: Price vs Status ----------
    plot_df = df.dropna(subset=["__status_num", "__price_num"])
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sns.scatterplot(data=plot_df, x="__price_num", y="__status_num", alpha=0.7, edgecolor=None, ax=ax)
    ax.set_xlabel("Price")
    ax.set_ylabel("Status (%)")
    ax.set_title("Status vs Price")
    sns.despine()
    fig.tight_layout()
    fig.savefig(out / "scatter_status_vs_price.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- 2) Box + Points: Brand vs Status ----------
    brand_col = None
    for cand in ("brand", "Brand", "ブランド", "メーカー"):
        if cand in df.columns:
            brand_col = cand
            break

    if brand_col:
        bdf = df.dropna(subset=["__status_num", brand_col]).copy()
        order = (
            bdf.groupby(brand_col)["__status_num"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

        fig, ax = plt.subplots(figsize=(11, 6.5))
        sns.boxplot(data=bdf, x=brand_col, y="__status_num", order=order,
                    color="lightgray", width=0.6, fliersize=2, linewidth=1.2, ax=ax)
        sns.stripplot(data=bdf, x=brand_col, y="__status_num", order=order,
                      alpha=0.55, size=3.5, jitter=0.25, edgecolor="gray", linewidth=0.3, ax=ax)

        means = bdf.groupby(brand_col)["__status_num"].mean().reindex(order)
        ax.scatter(x=np.arange(len(order)), y=means.values, s=70, marker="D",
                   color="black", alpha=0.5, zorder=5, label="Mean")

        ax.set_xlabel("Brand")
        ax.set_ylabel("Status (%)")
        ax.set_title("Status Distribution by Brand (sorted by mean)")
        ax.legend(frameon=False, loc="upper right")
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=6)
        sns.despine()
        fig.tight_layout()
        fig.savefig(out / "box_status_vs_brand.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        print("[WARN] 'brand' or 'ブランド' column not found. Boxplot skipped.")

    # ---------- 3) Box + Points: Price(¥1,000 bins) vs Status + bin means ----------
    # 価格を¥1,000刻みの区間にビニング（[下限, 上限)の半開区間。例: 6500 → 6000–7000）
    bdf2 = df.dropna(subset=["__status_num", "__price_num"]).copy()
    if not bdf2.empty:
        p_min = float(np.nanmin(bdf2["__price_num"]))
        p_max = float(np.nanmax(bdf2["__price_num"]))
        start = int(np.floor(p_min / 1000.0) * 1000)
        end   = int(np.ceil (p_max / 1000.0) * 1000)
        edges = np.arange(start, end + 1000, 1000, dtype=int)
        if len(edges) >= 2:
            labels = [f"{lo}–{lo+1000}" for lo in edges[:-1]]
            bdf2["price_bin"] = pd.cut(
                bdf2["__price_num"],
                bins=edges,
                labels=labels,
                right=False,          # 上限は含まない: [6000,7000)
                include_lowest=True
            )

            bdf2 = bdf2.dropna(subset=["price_bin"])
            order_bins = labels  # 軸の並びを下限→上限の順に固定

            fig, ax = plt.subplots(figsize=(12, 6.5))
            # ボックスプロット
            sns.boxplot(
                data=bdf2, x="price_bin", y="__status_num",
                order=order_bins, color="lightgray",
                width=0.6, fliersize=2, linewidth=1.2, ax=ax
            )
            # 各点を重ねる（軽くジッター）
            sns.stripplot(
                data=bdf2, x="price_bin", y="__status_num",
                order=order_bins, alpha=0.55, size=3.5,
                jitter=0.25, edgecolor="gray", linewidth=0.3, ax=ax
            )
            # 各区間の平均を重ねる
            # 各区間の中央値を重ねる
            bin_medians = (
                bdf2.groupby("price_bin")["__status_num"]
                .median()
                .reindex(order_bins)
            )
            ax.scatter(
                x=np.arange(len(order_bins)),
                y=bin_medians.values,
                s=70, marker="D", color="black", alpha=0.7, zorder=5, label="Median"
            )

            ax.set_xlabel("Price (¥1,000 bins)")
            ax.set_ylabel("Status (%)")
            ax.set_title("Status Distribution by Price (¥1,000 bins)")
            ax.legend(frameon=False, loc="upper right")
            plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=7)
            sns.despine()
            fig.tight_layout()
            fig.savefig(out / "box_status_vs_price_bins.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            print("[WARN] Not enough price range to create ¥1,000 bins. Skipped price-bin boxplot.")
    else:
        print("[WARN] No valid rows for price/status. Skipped price-bin boxplot.")

    print("[INFO] Saved: scatter_status_vs_price.png, box_status_vs_brand.png, box_status_vs_price_bins.png")

if __name__ == "__main__":
    main()
