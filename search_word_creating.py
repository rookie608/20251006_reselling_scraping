# make_search_keywords.py
# -*- coding: utf-8 -*-
"""
output/result.csv を読み込み、brand / product_name から
メルカリ・ラクマで検索可能な「ブランド名＋型式」の検索ワードを
OpenAI（ChatGPT）で生成し、search_keyword 列として追加して保存します。

インストール:
    pip install pandas openai python-dotenv

APIキー:
    .env に OPENAI_API_KEY を記載するか、環境変数で設定してください。
出力:
    output/result_with_keywords.csv
"""

import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai._exceptions import RateLimitError, APIError

# ========= 設定 =========
INPUT_CSV = "output/result.csv"
OUTPUT_CSV = "output/result_with_keywords.csv"
MODEL = "gpt-4.1-mini"
MAX_RETRIES = 5
RETRY_BASE_WAIT = 2.0  # 秒

SYSTEM_PROMPT = """あなたは中古EC（メルカリ・ラクマ）向けの検索キーワード生成アシスタントです。
出力は「ブランド名＋型式（型番）」を主軸に、日本語の揺れや英語表記の揺れに強い、短くシンプルな検索語を1行で返してください。

【出力ルール】
- 1行のみ。余計な説明や引用符は不要。
- 基本形は「{brand} {model}」。
- 型式（例: RB2140, RB2140-A, S12F, 10A0, S1、CD Diamond S1 など）を最優先で抽出。
- 型式が不明な場合は、ブランド名のみ（例: "Ray-Ban"）。
- カラーや性別は原則不要。ただし型式判別が困難でヒットが弱そうな場合のみ「サングラス」など1語を補助的に付けてもよい。
- ブランド名は公式英字表記優先。
- 出力は1行、テキストのみ。
"""

def build_user_prompt(brand: str, product_name: str) -> str:
    return f"""brand: {brand}
product_name: {product_name}

上記に対して、出力ルールに従い検索キーワードを1行で出力してください。"""

def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が見つかりません。.env か環境変数で設定してください。")
    return OpenAI(api_key=api_key)

def generate_keyword(client: OpenAI, brand: str, product_name: str) -> str:
    brand = (brand or "").strip()
    product_name = (product_name or "").strip()
    if not brand and not product_name:
        return ""

    prompt = build_user_prompt(brand, product_name)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=40,
            )
            text = (resp.choices[0].message.content or "").strip()
            return " ".join(text.split())
        except (RateLimitError, APIError) as e:
            wait = RETRY_BASE_WAIT * (2 ** (attempt - 1))
            print(f"[WARN] OpenAI API エラー {e} → {wait:.1f}s 待機 ({attempt}/{MAX_RETRIES})")
            time.sleep(wait)
        except Exception as e:
            wait = 1.0 * attempt
            print(f"[WARN] 不明なエラー {e} → {wait:.1f}s 待機 ({attempt}/{MAX_RETRIES})")
            time.sleep(wait)

    return brand  # フォールバック

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"{INPUT_CSV} が見つかりません。")

    print(f"[INFO] 読み込み中: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    if "brand" not in df.columns:
        df["brand"] = ""
    if "product_name" not in df.columns:
        df["product_name"] = ""

    client = get_client()
    keywords = []

    for idx, row in df.iterrows():
        kw = generate_keyword(client, str(row["brand"]), str(row["product_name"]))
        keywords.append(kw)
        print(f"[{idx+1}/{len(df)}] {kw}")

    df["search_keyword"] = keywords
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[✅ 完了] 出力: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
