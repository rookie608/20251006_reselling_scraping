#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
機能概要：
- output/ 配下の result_with_まとめ_*.csv を連結し、'url' で重複排除（df1）
- Googleスプレッドシート「リサーチシート」を取得（2行ヘッダー想定：1行目=装飾、2行目=実ヘッダー）
- df1 と df2 を url で比較し、df2 に存在しない新規のみ抽出
- C〜O（13列）に、C列の最初の空行から“そのまま”固定幅で貼り付け
  - price / shipping_fee は数値化して貼付（USER_ENTERED）
  - 任意で円フォーマット適用（下の FLAG_APPLY_YEN_FORMAT を True に）
- 連結結果を result_merge.csv に保存

前提：
- 同ディレクトリに service_account.json（サービスアカウント鍵）
- pip install pandas gspread google-auth
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

WRITE_START_COL = "C"
WRITE_END_COL   = "O"   # C〜O = 13列

# シートの実ヘッダー（C〜O：スクショ準拠）
TARGET_HEADERS: List[str] = [
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

# CSV列名ゆれ吸収のエイリアス
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

# 円フォーマットを自動適用するか（M:N=price:shipping_fee）
FLAG_APPLY_YEN_FORMAT = False  # 必要なら True に

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
    """指定列（例 'C'）で、ヘッダー行の次から最初の空セル行(1始まり)を返す。"""
    col_idx = col_to_index(col_letter)
    values = ws.col_values(col_idx)
    start_row = header_rows + 1
    for i in range(start_row, len(values) + 1):
        if values[i - 1] == "":
            return i
    return max(len(values) + 1, start_row)

def resolve_column_name(df: pd.DataFrame, target: str) -> Optional[str]:
    """df における target 列名（またはエイリアス）を返す。"""
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
        raise KeyError("CSVに 'url' 列（同義含む）が見つかりません。")
    if url_col != "url":
        merged = merged.rename(columns={url_col: "url"})
    merged = merged.dropna(subset=["url"]).copy()
    merged["url"] = merged["url"].astype(str).str.strip()
    merged = merged.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    return merged

def read_sheet_as_df(ws: gspread.Worksheet) -> pd.DataFrame:
    """先頭2行ヘッダー想定（1行目=装飾、2行目=実ヘッダー）。"""
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    if len(values) == 1:
        return pd.DataFrame(columns=values[0])
    header = values[1]
    body = values[2:]
    return pd.DataFrame(body, columns=header)

def build_write_dataframe(df_source: pd.DataFrame, headers: List[str], max_cols: int) -> pd.DataFrame:
    cols = []
    for h in headers[:max_cols]:
        src_col = resolve_column_name(df_source, h)
        if src_col is None:
            series = pd.Series([""] * len(df_source), name=h)
        else:
            series = df_source[src_col]
        cols.append(series)
    write_df = pd.concat(cols, axis=1)
    while write_df.shape[1] < max_cols:
        write_df[f"_blank_{write_df.shape[1]+1}"] = ""
    write_df = write_df.iloc[:, :max_cols]
    return write_df.fillna("").astype(str)

def coerce_numeric_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """'¥9,790' 等を 9790 に正規化。空や非数は '' に。"""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            def _to_number(x):
                if x is None:
                    return ""
                s = str(x)
                s = s.replace(",", "")
                s = re.sub(r"[^\d\.\-]", "", s)  # 通貨記号など除去
                if s in ("", "-", ".", "-.", ".-"):
                    return ""
                try:
                    f = float(s)
                    return int(f) if f.is_integer() else f
                except ValueError:
                    return ""
            df[c] = df[c].apply(_to_number)
    return df


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

    # 3) 既存データ（2行ヘッダー）→ df2
    df2 = read_sheet_as_df(ws)

    # 4) 新規URL抽出
    df2_urls = df2["url"].astype(str).str.strip().tolist() if "url" in df2.columns else []
    mask_new = ~df1["url"].astype(str).str.strip().isin(df2_urls)
    df_new = df1.loc[mask_new].reset_index(drop=True)
    print(f"[INFO] 新規URL 行数: {len(df_new)}")
    if df_new.empty:
        print("[INFO] 追加なし。終了。")
        return

    # 5) 書き込みデータ組成（C〜O順に固定）
    start_col_idx = col_to_index(WRITE_START_COL)
    end_col_idx   = col_to_index(WRITE_END_COL)
    max_cols = end_col_idx - start_col_idx + 1  # 13
    write_df = build_write_dataframe(df_new, TARGET_HEADERS, max_cols)

    # 5.1 price / shipping_fee を数値化
    write_df = coerce_numeric_columns(write_df, ["price", "shipping_fee"])

    # 6) C列の最初の空行から固定レンジで貼付
    start_row = find_first_empty_row_in_col(ws, WRITE_START_COL, header_rows=2)
    end_row   = start_row + len(write_df) - 1
    range_name = f"{WRITE_START_COL}{start_row}:{WRITE_END_COL}{end_row}"
    values_2d = write_df.values.tolist()

    # 数値は数値として入れたいので USER_ENTERED を利用
    ws.update(range_name, values_2d, value_input_option="USER_ENTERED")
    print(f"[INFO] 貼り付け完了: {len(values_2d)} 行 x {max_cols} 列（範囲: {range_name}）")

    # 7) 任意：円フォーマットを適用（M:N = price, shipping_fee）
    if FLAG_APPLY_YEN_FORMAT and len(write_df) > 0:
        ws.spreadsheet.batch_update({
            "requests": [{
                "repeatCell": {
                    "range": {
                        "sheetId": ws.id,
                        "startRowIndex": start_row - 1,
                        "endRowIndex": end_row,
                        "startColumnIndex": col_to_index("M") - 1,
                        "endColumnIndex": col_to_index("N"),
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "numberFormat": {
                                "type": "CURRENCY",
                                "pattern": "¥#,##0"  # 小数ありなら "¥#,##0.00"
                            }
                        }
                    },
                    "fields": "userEnteredFormat.numberFormat"
                }
            }]
        })
        print("[INFO] 円フォーマット適用（M:N）。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
