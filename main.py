# -*- coding: utf-8 -*-
"""
output/html/*.html をまとめて解析し、output/result.csv に出力します。
- 画像は「最初の1枚（できれば大サイズ .jpg）」のみ
- LLM(JSONモード)で抽出、ダメでもBeautifulSoupでフォールバック
"""

import os, re, csv, json, time, urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup, Comment
from openai import OpenAI
from openai import APIError, RateLimitError, BadRequestError

# ========= パス設定 =========
BASE_DIR = Path(__file__).resolve().parent
HTML_DIR = BASE_DIR / "output" / "html"
OUT_DIR  = BASE_DIR / "output"
OUT_CSV  = OUT_DIR / "result.csv"

# ========= OpenAI =========
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ========= 共通ユーティリティ =========
def normalize_url(url: str, base: str = "https://www.2ndstreet.jp/") -> str:
    if not url:
        return ""
    url = url.strip()
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return urllib.parse.urljoin(base, url)
    return url

def harvest_images(raw_html: str) -> List[str]:
    soup = BeautifulSoup(raw_html, "lxml")
    cand = set()

    # og:image / image_src
    for m in soup.select('meta[property="og:image"], meta[name="og:image"], link[rel="image_src"]'):
        u = m.get("content") or m.get("href")
        if u:
            cand.add(u)

    # preload
    for link in soup.select('link[rel="preload"][as="image"]'):
        u = link.get("href")
        if u:
            cand.add(u)

    # img[src] / lazy attrs
    for img in soup.select("img"):
        for attr in ("src", "data-src", "data-original", "data-lazy", "data-image"):
            u = img.get(attr)
            if not u:
                continue
            if ("goods" in u) or u.endswith((".jpg", ".jpeg", ".png", ".webp")):
                cand.add(u)

    # 正規化 + 重複排除
    norm, seen = [], set()
    for u in cand:
        nu = normalize_url(u)
        if nu and nu not in seen:
            seen.add(nu)
            norm.append(nu)

    # 商品画像っぽさ優先の並べ替え
    def score(u: str) -> tuple:
        s1 = 1 if "/goods/" in u or "img/pc/goods" in u else 0
        s2 = 1 if "og" in u else 0
        s3 = -len(u)
        s4 = 1 if re.search(r"[/_-]1(?:[_.-]|\.jpg|\.jpeg|\.png|\.webp)", u) else 0
        return (s1, s4, s2, s3)

    norm.sort(key=score, reverse=True)
    return norm

def clean_and_pick_block(raw_html: str) -> str:
    """冗長タグを除去し、価格/状態/送料等のキーワードにヒットする最小共通親ブロックを返す。"""
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for c in soup.find_all(string=lambda x: isinstance(x, Comment)):
        c.extract()

    root = soup.find("main") or soup.find(id=re.compile(r"(content|contents|main)", re.I)) or soup.body or soup

    PATTERNS = [
        re.compile(r"¥\s?\d[\d,]*"),
        re.compile(r"商品の状態"),
        re.compile(r"中古[ABCD]|未使用品?"),
        re.compile(r"送料[:：]?\s?¥?\s?[\d,]+"),
        re.compile(r"商品[の]?説明"),
    ]

    def find_hit_nodes(scope):
        hits = []
        for txt in scope.find_all(string=True):
            s = (txt or "").strip()
            if not s:
                continue
            if any(p.search(s) for p in PATTERNS):
                hits.append(txt.parent)
        return hits

    def lca(nodes):
        if not nodes:
            return None
        sets = []
        for n in nodes:
            s = set()
            cur = n
            while cur is not None:
                s.add(cur)
                cur = cur.parent
            sets.append(s)
        common = set.intersection(*sets) if sets else set()
        best, depth = None, -1
        for el in common:
            d, cur = 0, el
            while cur is not None:
                d += 1
                cur = cur.parent
            if d > depth:
                best, depth = el, d
        return best

    hits = find_hit_nodes(root)
    block = lca(hits) or root
    snippet = str(block)

    # 大きすぎたら縮める
    if len(snippet) > 120_000:
        candidates = [c for c in block.find_all(["section", "div", "article"], recursive=True)]
        scored = []
        for c in candidates[:300]:
            text = c.get_text(" ", strip=True)[:2000]
            score = sum(1 for p in PATTERNS if p.search(text))
            scored.append((score, len(text), c))
        scored.sort(reverse=True, key=lambda t: (t[0], -t[1]))
        if scored and scored[0][0] > 0:
            snippet = str(scored[0][2])

    MAX_CHARS = 90_000
    if len(snippet) > MAX_CHARS:
        snippet = snippet[:60_000] + "\n<!-- TRUNCATED -->\n" + snippet[-30_000:]

    return snippet

def pick_primary(imgs: List[str]) -> Optional[str]:
    if not imgs:
        return None
    # フル解像度 .jpg（_mn でない）を最優先
    for u in imgs:
        if u.endswith(".jpg") and "_mn" not in u:
            return u
    return imgs[0]

# ========= LLM 抽出 =========
SYSTEM_PROMPT = (
    "You are an information extraction engine. "
    "Return ONLY valid JSON with the required keys. No explanations."
)

def llm_extract(snippet: str) -> Dict[str, Any]:
    USER_PROMPT = """以下は中古販売サイト「セカンドストリート」の商品HTMLです。
次のキーだけを含む厳密なJSONを返してください（余計なキーは禁止）:

{
  "images": string[],
  "product_name": string,
  "model": string,
  "categories": string[],
  "condition": string,
  "price": string,
  "shipping_fee": string
}

- 値はページ表記を尊重
- 見つからない項目は空文字または空配列
- 絶対にJSON以外の出力をしない

--- HTML スニペット ---
""" + snippet

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content
    return json.loads(content)

# ========= BeautifulSoup フォールバック =========
def bs_fallback(raw_html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(raw_html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    product_name = re.sub(r"\s*\|\s*中古品の販売・通販ならセカンドストリート.*$", "", title)

    # price
    price = ""
    m = re.search(r"¥\s*[\d,]+", raw_html)
    if m:
        price = m.group(0)

    # condition
    condition = ""
    for s in soup.stripped_strings:
        if "商品の状態" in s or re.search(r"中古[ABCD]|未使用品?", s):
            mc = re.search(r"(中古[ABCD]|未使用品?)", s)
            if mc:
                condition = mc.group(1); break

    # categories（パンくずがない場合は空）
    cats = []
    for tag in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(tag.string or "")
            if isinstance(data, dict) and data.get("@type") == "BreadcrumbList":
                for el in data.get("itemListElement", []):
                    nm = el.get("name")
                    if nm and nm not in ["HOME", "買いたい", "商品詳細"]:
                        cats.append(nm)
        except Exception:
            pass

    # model
    model = ""
    mm = re.search(r"\bRB\d+[A-Z]?\b", raw_html)
    if mm:
        model = mm.group(0)

    # shipping
    ship = ""
    ms = re.search(r"送料[:：]?\s*¥?\s*[\d,]+", raw_html)
    if ms:
        ship = ms.group(0)
        # 形を合わせて値だけにする
        mprice = re.search(r"¥\s*[\d,]+", ship)
        if mprice:
            ship = mprice.group(0)

    return {
        "images": [],
        "product_name": product_name,
        "model": model,
        "categories": cats,
        "condition": condition,
        "price": price,
        "shipping_fee": ship,
    }

# ========= 1ファイル処理 =========
def process_file(html_path: Path) -> Dict[str, Any]:
    raw_html = html_path.read_text(encoding="utf-8", errors="ignore")

    # 画像候補を先取り
    pre_images = harvest_images(raw_html)

    # 商品ブロックを抽出
    snippet = clean_and_pick_block(raw_html)

    # LLMで抽出（必要に応じてリトライ）
    data: Dict[str, Any]
    try:
        data = llm_extract(snippet)
    except (RateLimitError, BadRequestError) as e:
        # 少し待って再試行（1回）
        time.sleep(2.0)
        try:
            data = llm_extract(snippet)
        except Exception:
            data = bs_fallback(raw_html)
    except Exception:
        data = bs_fallback(raw_html)

    # 画像をマージ → 1枚だけ
    images = data.get("images") or []
    if not images and pre_images:
        images = pre_images
    else:
        pool, uniq, seen = images + pre_images, [], set()
        for u in pool:
            nu = normalize_url(u)
            if nu and nu not in seen:
                seen.add(nu); uniq.append(nu)
        images = uniq
    primary = pick_primary(images)
    data["images"] = [primary] if primary else []

    # 価格の軽い補正（¥が無ければ付ける）
    if data.get("price") and re.fullmatch(r"\d[\d,]*", data["price"]):
        data["price"] = "¥" + data["price"]

    return data

# ========= メイン（一括処理→CSV） =========
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not HTML_DIR.exists():
        raise SystemExit(f"[ERROR] フォルダがありません: {HTML_DIR}")

    files = sorted(HTML_DIR.glob("*.html"))
    if not files:
        raise SystemExit(f"[WARN] HTMLが見つかりません: {HTML_DIR}")

    rows = []
    print(f"[INFO] {len(files)} 件のHTMLを処理します")
    for i, f in enumerate(files, 1):
        try:
            data = process_file(f)
            rows.append({
                "file": f.name,
                "image": (data.get("images") or [""])[0],
                "product_name": data.get("product_name", ""),
                "model": data.get("model", ""),
                "categories": " > ".join(data.get("categories", [])),
                "condition": data.get("condition", ""),
                "price": data.get("price", ""),
                "shipping_fee": data.get("shipping_fee", ""),
            })
            print(f"[OK] {i}/{len(files)} {f.name}")
            # 軽いレート対策
            time.sleep(0.4)
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")

    # CSV出力
    fieldnames = ["file", "image", "product_name", "model", "categories", "condition", "price", "shipping_fee"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[✅ 完了] {len(rows)}件を書き出し: {OUT_CSV}")

if __name__ == "__main__":
    main()
