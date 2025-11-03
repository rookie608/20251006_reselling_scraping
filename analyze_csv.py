#!/usr/bin/env python3
"""
input_analysis フォルダ内のCSVをglob/pathlibでまとめて読み込み、2行目を列名としてpandasに取り込み、
以下の条件で加工して2種類のグラフを作成・保存します。

1. 散布図：横軸 Price、縦軸 Status（%）
2. ボックスプロット＋点表示：横軸 Brand、縦軸 Status（%）（平均値の高い順）

加工条件:
- 「作業者」がある行のみ抽出（空文字/NaNは除外）
- 「ステータス」が -100% の行は除外（-100 または "-100%" どちらにも対応）

出力:
- output_analysis/scatter_status_vs_price.png
- output_analysis/box_status_vs_brand.png
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === CSV読み込み関数 ===
def load_csvs_from_input_folder(folder: str = "input_analysis", header_row: int = 1) -> pd.DataFrame:
    folder_path = Path(folder)
    files = sorted(folder_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"{folder} にCSVファイルが見つかりません。")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, header=header_row)
        except UnicodeDecodeError:
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


# === 数値クレンジング関数 ===
def to_numeric_clean(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"[\s,\u00A0\u3000¥￥]", "", regex=True)
        .str.replace("%", "", regex=False)
        .replace({"nan": None, "": None})
        .astype(float)
    )


# === price列の自動検出 ===
def find_price_column(columns: list[str]) -> str | None:
    for c in columns:
        if c.lower() == "price":
            return c
    for cand in ("価格", "金額", "販売価格", "単価", "金額（税込）"):
        if cand in columns:
            return cand
    return None


# === メイン処理 ===
def main():
    df = load_csvs_from_input_folder()

    # 必須列チェック
    if "作業者" not in df.columns or "ステータス" not in df.columns:
        raise ValueError("CSVに '作業者' または 'ステータス' 列がありません。")

    price_col = find_price_column(list(df.columns))
    if not price_col:
        raise ValueError("'price' または '価格' 列が見つかりません。")

    # フィルタリング
    df = df[df["作業者"].astype(str).str.strip().ne("")]
    df["__status_num"] = to_numeric_clean(df["ステータス"])
    df = df[df["__status_num"] != -100]
    df["__price_num"] = to_numeric_clean(df[price_col])

    plot_df = df.dropna(subset=["__status_num", "__price_num"])

    output_dir = Path("output_analysis")
    output_dir.mkdir(exist_ok=True)

    # --------- グラフ① 散布図：Price vs Status ---------
    scatter_path = output_dir / "scatter_status_vs_price.png"
    plt.figure(figsize=(7, 5))
    plt.scatter(plot_df["__price_num"], plot_df["__status_num"], alpha=0.7)
    plt.xlabel("Price")
    plt.ylabel("Status (%)")
    plt.title("Status vs Price")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved scatter plot: {scatter_path}")

    # --------- グラフ② ボックスプロット＋点表示 ---------
    brand_col = None
    for cand in ("brand", "Brand", "ブランド", "メーカー"):
        if cand in df.columns:
            brand_col = cand
            break

    if brand_col:
        plot_data = df.dropna(subset=["__status_num", brand_col])

        # 平均値順に並び替え
        mean_order = (
            plot_data.groupby(brand_col)["__status_num"]
            .mean()
            .sort_values(ascending=False)
            .index
        )

        box_path = output_dir / "box_status_vs_brand.png"
        plt.figure(figsize=(10, 6))

        # ボックスプロット（平均順）
        data = [plot_data.loc[plot_data[brand_col] == b, "__status_num"] for b in mean_order]
        plt.boxplot(
            data,
            labels=mean_order,
            patch_artist=True,
            boxprops=dict(facecolor="lightgray", alpha=0.6),
            medianprops=dict(color="red", linewidth=1.2)
        )

        # 各点（個別データ）を重ねて描画
        for i, b in enumerate(mean_order, start=1):
            y = plot_data.loc[plot_data[brand_col] == b, "__status_num"]
            # 少し横方向にばらけさせて重なりを避ける
            x = np.random.normal(i, 0.04, size=len(y))
            plt.scatter(x, y, alpha=0.6, color="skyblue", s=30, edgecolors="gray", linewidths=0.3)

        plt.xlabel("Brand")
        plt.ylabel("Status (%)")
        plt.title("Status Distribution by Brand (sorted by mean)")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(box_path, dpi=150)
        plt.close()
        print(f"[INFO] Saved boxplot with scatter: {box_path}")
    else:
        print("[WARN] 'brand' or 'ブランド' column not found. Boxplot skipped.")


if __name__ == "__main__":
    main()
