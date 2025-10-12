# -*- coding: utf-8 -*-
"""
input/30032_c94b5b5eb0.html を読み込み、
1) 商品ブロックだけ抽出（冗長なhead/script/style除去）
2) Chat Completions(JSONモード)で主要情報を抽出
3) 画像は BeautifulSoup で先取り収集して LLM結果にマージ（最初の1枚のみ）
"""

import os, json, re, urllib.parse
from pathlib import Path
from bs4 import BeautifulSoup, Comment
from openai import OpenAI

# ===== 基本設定 =====
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "input" / "30032_c94b5b5eb0.html"
OUT_PATH  = BASE_DIR / "output.json"

if not HTML_PATH.exists():
    raise FileNotFoundError(f"HTMLが見つかりません: {HTML_PATH}")

raw_html = HTML_PATH.read_text(encoding="utf-8", errors="ignore")

# ===== 画像ハーベスト（先取り収集） =====
def normalize_url(url: str, base: str = "https://www.2ndstreet.jp/") -> str:
    if not url:
        return ""
    url = url.strip()
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return urllib.parse.urljoin(base, url)
    return url

def harvest_images(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    cand = set()

    for m in soup.select('meta[property="og:image"], meta[name="og:image"], link[rel="image_src"]'):
        u = m.get("content") or m.get("href")
        if u:
            cand.add(u)

    for link in soup.select('link[rel="preload"][as="image"]'):
        u = link.get("href")
        if u:
            cand.add(u)

    for img in soup.select("img"):
        for attr in ("src", "data-src", "data-original", "data-lazy", "data-image"):
            u = img.get(attr)
            if not u:
                continue
            if ("goods" in u) or u.endswith((".jpg", ".jpeg", ".png", ".webp")):
                cand.add(u)

    norm = []
    seen = set()
    for u in cand:
        nu = normalize_url(u)
        if nu and nu not in seen:
            seen.add(nu)
            norm.append(nu)

    # 並べ替え：商品画像っぽさ優先
    def score(u: str) -> tuple:
        s1 = 1 if "/goods/" in u or "img/pc/goods" in u else 0
        s2 = 1 if "og" in u else 0
        s3 = -len(u)
        s4 = 1 if re.search(r"[/_-]1(?:[_.-]|\.jpg|\.jpeg|\.png|\.webp)", u) else 0
        return (s1, s4, s2, s3)

    norm.sort(key=score, reverse=True)
    return norm

pre_images = harvest_images(raw_html)

# ===== HTMLを軽量化して“商品ブロック”だけ抽出 =====
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
        s = txt.strip()
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
        d = 0
        cur = el
        while cur is not None:
            d += 1
            cur = cur.parent
        if d > depth:
            best, depth = el, d
    return best

hits = find_hit_nodes(root)
block = lca(hits) or root
snippet = str(block)

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

# ===== LLM プロンプト（JSONモード） =====
SYSTEM_PROMPT = (
    "You are an information extraction engine. "
    "Return ONLY valid JSON with the required keys. No explanations."
)
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

print("[INFO] OpenAIで解析中...")

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
try:
    data = json.loads(content)
except Exception:
    data = {"images": [], "product_name": "", "model": "", "categories": [], "condition": "", "price": "", "shipping_fee": ""}

# ===== 画像をマージ → 最初の1枚だけ =====
images = data.get("images") or []
if not images and pre_images:
    images = pre_images
else:
    pool = images + pre_images
    uniq, seen = [], set()
    for u in pool:
        nu = normalize_url(u)
        if nu and nu not in seen:
            seen.add(nu)
            uniq.append(nu)
    images = uniq

# 代表画像を1枚だけ選ぶ（大きめ優先）
def pick_primary(imgs: list[str]) -> str | None:
    if not imgs:
        return None
    for u in imgs:
        if u.endswith(".jpg") and "_mn" not in u:
            return u
    return imgs[0]

primary = pick_primary(images)
data["images"] = [primary] if primary else []

# 価格に通貨記号が落ちている場合の軽い補正
if data.get("price") and re.fullmatch(r"\d[\d,]*", data["price"]):
    data["price"] = "¥" + data["price"]

# ===== 出力 =====
OUT_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("[✅ 完了] 抽出結果を output.json に保存しました。")
print(json.dumps(data, ensure_ascii=False, indent=2))
