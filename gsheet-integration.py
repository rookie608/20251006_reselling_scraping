#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
機能概要：
- output/ 配下の result_with_まとめ_*.csv を連結し、url重複を排除（df1）
- Googleスプレッドシート「リサーチシート」を取得（2行ヘッダー想定）
- df1 と df2 の url を比較し、df2 にない新規のみ抽出
- C〜O（13列）＋ B列（管理番号）を自動追記
  → B列は上のセルの管理番号（例 LC0579）を読み取り、連番で付与
- price / shipping_fee は数値として貼付
"""

from pathlib import Path
from typing import List, Dict, Optional
import sys
import re
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ===== 設定 =====
SPREADSHEET_KEY = "1TCDZ5rDTicA5lWISooKsb4ZyDvXzPiPkI11OgD1cpQI"
SHEET_NAME = "リサーチシート"

WRITE_START_COL = "B"  # ← B列から書く（管理番号含む）
WRITE_END_COL   = "O"  # ← B〜O で14列（管理番号+13列）

TARGET_HEADERS: List[str] = [
    "管理番号",  # B列
    "url",
    "file",
    "status",
    "saved_at",
    "image",
    "product_name",
    "brand",
    "model",
    "categories",
    "condition",
    "price",
    "shipping_fee",
    "search_keyword",
]

ALIASES: Dict[str, List[str]] = {
    "url": ["URL", "Url"],
    "file": ["input_file", "prod_file", "file_path", "file_name"],
    "status": ["state"],
    "saved_at": ["saved", "savedAt", "created_at", "timestamp", "scraped_at"],
    "image": ["image_url", "img", "thumbnail", "thumb"],
    "product_name": ["title", "name", "product"],
    "brand": ["maker", "manufacturer"],
    "model": ["model_name", "code", "型番"],
    "categories": ["category", "cat"],
    "condition": ["item_condition", "rank", "grade"],
    "price": ["amount", "price_jpy", "price_yen"],
    "shipping_fee": ["shipping", "postage", "delivery_fee", "shipping_cost"],
    "search_keyword": ["keyword", "search_key", "searchword"],
}

SERVICE_ACCOUNT_JSON = Path("service_account.json")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# ===== ユーティリティ =====
def get_credentials() -> Credentials:
    if not SERVICE_ACCOUNT_JSON.exists():
        raise FileNotFoundError("service_account.json が見つかりません。")
    return Credentials.from_service_account_file(str(SERVICE_ACCOUNT_JSON), scopes=SCOPES)

def col_to_index(letter: str) -> int:
    letter = letter.upper()
    num = 0
    for ch in letter:
        num = num * 26 + (ord(ch) - ord('A') + 1)
    return num

def find_first_empty_row_in_col(ws: gspread.Worksheet, col_letter: str, header_rows: int = 2) -> int:
    col_idx = col_to_index(col_letter)
    values = ws.col_values(col_idx)
    start_row = header_rows + 1
    for i in range(start_row, len(values) + 1):
        if values[i - 1] == "":
            return i
    return max(len(values) + 1, start_row)

def resolve_column_name(df: pd.DataFrame, target: str) -> Optional[str]:
    if target in df.columns:
        return target
    for c in df.columns:
        if c.lower() == target.lower():
            return c
    for alias in ALIASES.get(target, []):
        for c in df.columns:
            if c.lower() == alias.lower():
                return c
    return None

def merge_csvs(output_dir: Path) -> pd.DataFrame:
    files = sorted(output_dir.glob("result_with_まとめ_*.csv"))
    if not files:
        raise FileNotFoundError("output/ に result_with_まとめ_*.csv が見つかりません。")
    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except UnicodeDecodeError:
            df = pd.read_csv(fp, encoding="utf-8-sig")
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)

    url_col = resolve_column_name(merged, "url")
    if url_col is None:
        raise KeyError("CSVに 'url' 列が見つかりません。")
    if url_col != "url":
        merged = merged.rename(columns={url_col: "url"})
    merged = merged.dropna(subset=["url"]).copy()
    merged["url"] = merged["url"].astype(str).str.strip()
    merged = merged.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    return merged

def read_sheet_as_df(ws: gspread.Worksheet) -> pd.DataFrame:
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    if len(values) == 1:
        return pd.DataFrame(columns=values[0])
    header = values[1]
    body = values[2:]
    return pd.DataFrame(body, columns=header)

def coerce_numeric_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            def _to_number(x):
                if x is None:
                    return ""
                s = str(x).replace(",", "")
                s = re.sub(r"[^\d\.\-]", "", s)
                if s in ("", "-", ".", "-.", ".-"):
                    return ""
                try:
                    f = float(s)
                    return int(f) if f.is_integer() else f
                except ValueError:
                    return ""
            df[c] = df[c].apply(_to_number)
    return df

def get_next_ids(last_id: str, count: int) -> List[str]:
    """例: 'LC0579' -> ['LC0580','LC0581',...]"""
    prefix = re.match(r"[A-Za-z]+", last_id).group(0) if re.match(r"[A-Za-z]+", last_id) else ""
    num_part = re.search(r"\d+", last_id)
    start_num = int(num_part.group(0)) if num_part else 0
    width = len(num_part.group(0)) if num_part else 4
    return [f"{prefix}{str(start_num + i + 1).zfill(width)}" for i in range(count)]

# ===== メイン =====
def main() -> None:
    root = Path(__file__).resolve().parent
    output_dir = root / "output"

    # 1) CSV マージ
    df1 = merge_csvs(output_dir)
    (root / "result_merge.csv").write_text(df1.to_csv(index=False), encoding="utf-8")
    print(f"[INFO] CSVマージ完了: result_merge.csv（{len(df1)} 行）")

    # 2) シート接続
    creds = get_credentials()
    gc = gspread.authorize(creds)
    ws = gc.open_by_key(SPREADSHEET_KEY).worksheet(SHEET_NAME)
    df2 = read_sheet_as_df(ws)

    # 3) 新規URL抽出
    df2_urls = df2["url"].astype(str).str.strip().tolist() if "url" in df2.columns else []
    mask_new = ~df1["url"].astype(str).str.strip().isin(df2_urls)
    df_new = df1.loc[mask_new].reset_index(drop=True)
    print(f"[INFO] 新規URL 行数: {len(df_new)}")
    if df_new.empty:
        print("[INFO] 追加なし。終了。")
        return

    # 4) 数値列整形
    df_new = coerce_numeric_columns(df_new, ["price", "shipping_fee"])

    # 5) B列の管理番号を生成
    b_values = ws.col_values(col_to_index("B"))
    last_id = next((v for v in reversed(b_values) if v.strip()), "LC0000")
    new_ids = get_next_ids(last_id, len(df_new))
    print(f"[INFO] 管理番号: {last_id} → {new_ids[0]}〜")

    # 6) 書き込みデータ組成（B〜O固定）
    write_df = pd.DataFrame()
    write_df["管理番号"] = new_ids
    for h in TARGET_HEADERS[1:]:
        src_col = resolve_column_name(df_new, h)
        write_df[h] = df_new[src_col] if src_col else ""

    # 7) 書き込み範囲算出
    start_row = find_first_empty_row_in_col(ws, "C", header_rows=2)
    end_row   = start_row + len(write_df) - 1
    range_name = f"{WRITE_START_COL}{start_row}:{WRITE_END_COL}{end_row}"

    # 8) 貼り付け（USER_ENTERED）
    values_2d = write_df.fillna("").astype(str).values.tolist()
    ws.update(range_name, values_2d, value_input_option="USER_ENTERED")

    print(f"[INFO] 貼り付け完了: {len(values_2d)} 行（範囲: {range_name}）")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
