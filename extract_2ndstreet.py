# -*- coding: utf-8 -*-
"""
extract_2ndstreet.py
──────────────────────────────
2nd STREET の商品HTMLから以下の情報を抽出：

1) images: 画像URLリスト
2) product_name: 商品名
3) model_and_category:
     - model: 型式
     - categories: パンくずリスト
4) condition: 商品の状態
5) price: 値段
6) shipping_fee: 送料
（＋ brand / notes も補助的に取得）

OpenAI Responses API（Structured Outputs）＋ BeautifulSoup 併用構成。
"""

import os
import re
import json
import html
import argparse
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from bs4 import BeautifulSoup

# --- OpenAI SDK (ver 1.x 系) ---
try:
    from openai import OpenAI
    client = OpenAI()  # OPENAI_API_KEY を環境変数から取得
except Exception as e:
    print(f"[WARN] OpenAI SDK を初期化できません: {e}")
    client = None

# ====== Structured Output 用 JSON Schema ======
EXTRACTION_SCHEMA = {
    "name": "SecondStreetProduct",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "images": {"type": "array", "items": {"type": "string", "format": "uri"}, "default": []},
            "product_name": {"type": "string", "default": ""},
            "model_and_category": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "default": ""},
                    "categories": {"type": "array", "items": {"type": "string"}, "default": []},
                },
                "required": ["model", "categories"],
            },
            "condition": {"type": "string", "default": ""},
            "price": {"type": "string", "default": ""},
            "shipping_fee": {"type": "string", "default": ""},
            "brand": {"type": "string", "default": ""},
            "notes": {"type": "string", "default": ""},
        },
        "required": ["images", "product_name", "model_and_category", "condition", "price", "shipping_fee"],
    },
}

SYSTEM_PROMPT = (
    "You are an information extraction engine. "
    "Return ONLY JSON matching the provided json_schema. Do not include markdown or prose."
)

USER_PREFIX = """以下は日本の中古販売サイト「セカンドストリート」の商品詳細HTMLです。
次の6つを厳密に抽出し、json_schemaどおりのJSONで返してください。

1) images: 画像URLの配列（代表画像を先頭に）
2) product_name: 商品名（ブランド名を含むと望ましい）
3) model_and_category: { model: 型式, categories: パンくず配列 }
4) condition: 商品の状態
5) price: 値段（表記をできる限りそのまま）
6) shipping_fee: 送料（なければ空文字）
補助: brand, notes も返してください。
必ず日本語で返し、値はページ上の表記を保持してください。

HTML:
"""

# ====== HTML ロード関数 ======
def load_html(html_path: Optional[str], url: Optional[str], use_stdin: bool) -> str:
    if html_path:
        return Path(html_path).read_text(encoding="utf-8", errors="ignore")
    if url:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    if use_stdin:
        import sys
        return html.unescape(sys.stdin.read())
    raise SystemExit("HTML入力が見つかりません (--html or --url or --stdin)。")

# ====== LLMでの抽出 ======
def extract_with_llm(html_text: str) -> Optional[Dict[str, Any]]:
    if client is None:
        return None

    # 長いHTMLは先頭＋末尾だけ残す
    if len(html_text) > 180_000:
        html_text = html_text[:120_000] + "\n<!-- TRUNCATED -->\n" + html_text[-60_000:]

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PREFIX + html_text},
            ],
            response_format={"type": "json_schema", "json_schema": EXTRACTION_SCHEMA},
            temperature=0,
        )

        parsed = getattr(resp, "output_parsed", None)
        if not parsed:
            txt = getattr(resp, "output_text", None) or getattr(resp, "text", None) or ""
            parsed = json.loads(txt) if txt else None
        return parsed if isinstance(parsed, dict) else None

    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return None

# ====== BeautifulSoupでのフォールバック抽出 ======
def extract_with_bs(html_text: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html_text, "lxml")

    # --- 画像 ---
    images: List[str] = []
    for link in soup.select('link[rel="preload"][as="image"]'):
        href = link.get("href")
        if href and href not in images:
            images.append(href)
    og = soup.find("meta", property="og:image")
    if og and og.get("content") and og["content"] not in images:
        images.append(og["content"])
    for img in soup.select("img[src]"):
        src = img.get("src")
        if src and "goods" in src and src not in images:
            images.append(src)

    # --- 商品名 ---
    title = soup.title.get_text(strip=True) if soup.title else ""
    product_name = re.sub(r"\s*\|\s*中古品の販売・通販ならセカンドストリート.*$", "", title)

    # --- ブランド / 価格 ---
    brand, price = "", ""
    m_brand = re.search(r"'brand'\s*:\s*'([^']+)'", html_text)
    if m_brand:
        brand = m_brand.group(1)
    m_price = re.search(r"'price'\s*:\s*'?(\d+)'?", html_text)
    if m_price:
        price = f"¥{int(m_price.group(1)):,}"
    if not price:
        tag_price = soup.find(string=re.compile(r"¥\s*\d[\d,]*"))
        if tag_price:
            price = re.search(r"¥\s*[\d,]+", tag_price).group(0)

    # --- パンくずリスト ---
    categories: List[str] = []
    for s in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(s.string or "")
            if isinstance(data, dict) and data.get("@type") == "BreadcrumbList":
                for el in data.get("itemListElement", []):
                    nm = el.get("name")
                    if nm and nm not in ["HOME", "買いたい", "商品詳細"]:
                        categories.append(nm)
        except Exception:
            pass

    # --- 型式（例: RB4459D） ---
    model = ""
    mm = re.search(r"\b([A-Z]{1,3}\d{3,5}[A-Z]?)\b", title)
    if mm:
        model = mm.group(1)
    if not model:
        mm2 = re.search(r"\bRB\d+[A-Z]?\b", html_text)
        if mm2:
            model = mm2.group(0)

    # --- 状態 ---
    condition = ""
    for s in soup.stripped_strings:
        if "商品の状態" in s and ("中古" in s or "未使用" in s):
            mcond = re.search(r"(中古[ABCD]|未使用品?)", s)
            if mcond:
                condition = mcond.group(1)
                break
    if not condition:
        btn = soup.find(string=re.compile(r"中古[ABCD]"))
        if btn:
            condition = re.search(r"中古[ABCD]", btn).group(0)

    # --- 送料 ---
    shipping = ""
    ship_tag = soup.find(string=re.compile(r"送料"))
    if ship_tag:
        mship = re.search(r"送料[:：]?\s*¥?\s*[\d,]+", ship_tag)
        if mship:
            shipping = mship.group(0)

    if not images:
        for m in re.finditer(r"https?://[^\"'>]+/goods/[^\"'>]+\.jpg", html_text):
            images.append(m.group(0))

    return {
        "images": images[:6],
        "product_name": product_name,
        "model_and_category": {"model": model, "categories": categories},
        "condition": condition,
        "price": price,
        "shipping_fee": shipping,
        "brand": brand,
        "notes": "BeautifulSoupフォールバックで抽出",
    }

# ====== メイン関数（run_extract） ======
def run_extract(html_text: str) -> Dict[str, Any]:
    data = extract_with_llm(html_text) or extract_with_bs(html_text)
    return {
        "1_images": data.get("images", []),
        "2_product_name": data.get("product_name", ""),
        "3_model": data.get("model_and_category", {}).get("model", ""),
        "3_categories": data.get("model_and_category", {}).get("categories", []),
        "4_condition": data.get("condition", ""),
        "5_price": data.get("price", ""),
        "6_shipping_fee": data.get("shipping_fee", ""),
        "brand": data.get("brand", ""),
        "notes": data.get("notes", ""),
    }

# ====== CLI 実行対応 ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", help="HTMLファイルパス")
    ap.add_argument("--url", help="URL入力")
    ap.add_argument("--stdin", action="store_true", help="STDINからHTML受け取り")
    ap.add_argument("--out", default="result.json", help="出力JSONパス")
    args = ap.parse_args()

    html_text = load_html(args.html, args.url, args.stdin)
    result = run_extract(html_text)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[OK] 書き出し: {args.out}")

if __name__ == "__main__":
    main()
