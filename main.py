# scrape_secondstreet_stealth.py
# -*- coding: utf-8 -*-
"""
一気通貫版：
1) input/*.csv のURLから商品ページを取得し、output/html/*.html に保存
2) 保存済みHTMLを解析して、output/result.csv を出力
   - 画像は最初の1枚（できれば大サイズ .jpg）に限定
   - OpenAI(JSONモード)抽出 → 失敗時はBeautifulSoupフォールバック
使い方(最速):
  pip install playwright beautifulsoup4 lxml openai
  playwright install
  python scrape_secondstreet_stealth.py --headless
オプション:
  --phase both|html|parse        実行フェーズを選択（既定 both）
  --overwrite                    既存HTMLを上書き保存
  --delay 1.0                    各リクエスト間の待機秒（保存時）
  --retries 2                    保存失敗時のリトライ回数
  --no-llm                       LLM抽出を使わず常にBSフォールバック
"""

import os, re, csv, json, time, argparse, hashlib, urllib.parse, sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# ========== パス ==========
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUT_DIR   = BASE_DIR / "output"
HTML_DIR  = OUT_DIR / "html"
OUT_CSV   = OUT_DIR / "result.csv"

# ========== 依存 ==========
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup, Comment

# OpenAIは任意（未設定なら自動フォールバック）
_USE_LLM = True
try:
    from openai import OpenAI
    from openai import APIError, RateLimitError, BadRequestError
    _OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    if not _OPENAI_KEY:
        _USE_LLM = False
        OpenAI = None  # type: ignore
        APIError = RateLimitError = BadRequestError = Exception  # type: ignore
    else:
        client = OpenAI(api_key=_OPENAI_KEY)
except Exception:
    _USE_LLM = False

# ========== URL列名候補 ==========
URL_HEADERS = {"url","URL","Url","link","Link","LINK","商品URL","リンク","商品リンク"}

# ========== 共通ユーティリティ ==========
def read_all_csv_urls(input_dir: Path) -> List[str]:
    urls = []
    for p in input_dir.glob("*.csv"):
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        header = [c.strip() for c in rows[0]]
        url_col = None
        for i, h in enumerate(header):
            if h in URL_HEADERS or h.lower() in {x.lower() for x in URL_HEADERS}:
                url_col = i; break
        start = 1 if url_col is not None else 0
        if url_col is None:
            if rows and rows[0] and re.match(r"^https?://", (rows[0][0] or "").strip()):
                url_col = 0
            else:
                for j, c in enumerate(rows[1] if len(rows)>1 else []):
                    if re.match(r"^https?://", (c or "").strip()):
                        url_col = j; break
                if url_col is None:
                    url_col = 0
        for r in rows[start:]:
            if not r or url_col >= len(r):
                continue
            u = (r[url_col] or "").strip()
            if u.startswith("http"):
                urls.append(u)
    # 重複除去
    seen, uniq = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq

def build_html_filename(url: str) -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    tail = url.split("/")[-1] or "page"
    safe_tail = re.sub(r"[^A-Za-z0-9_-]+", "_", tail)
    return f"{safe_tail}_{h}.html"

def save_html(page, url: str, html_dir: Path, *, overwrite: bool=False, wait_selector: str="h1") -> Path:
    fname = build_html_filename(url)
    path = html_dir / fname
    if path.exists() and not overwrite:
        print(f"[SKIP] {url} → {path.name}（既存）")
        return path
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    if wait_selector:
        try:
            page.wait_for_selector(wait_selector, timeout=8000)
        except Exception:
            pass
    html = page.content()
    path.write_text(html, encoding="utf-8")
    print(f"[SAVE] {url} → {path.name}")
    return path

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
    for m in soup.select('meta[property="og:image"], meta[name="og:image"], link[rel="image_src"]'):
        u = m.get("content") or m.get("href")
        if u: cand.add(u)
    for link in soup.select('link[rel="preload"][as="image"]'):
        u = link.get("href")
        if u: cand.add(u)
    for img in soup.select("img"):
        for attr in ("src","data-src","data-original","data-lazy","data-image"):
            u = img.get(attr)
            if not u: continue
            if ("goods" in u) or u.endswith((".jpg",".jpeg",".png",".webp")):
                cand.add(u)
    normd, seen = [], set()
    for u in cand:
        nu = normalize_url(u)
        if nu and nu not in seen:
            seen.add(nu); normd.append(nu)
    def score(u: str) -> tuple:
        s1 = 1 if "/goods/" in u or "img/pc/goods" in u else 0
        s2 = 1 if "og" in u else 0
        s3 = -len(u)
        s4 = 1 if re.search(r"[/_-]1(?:[_.-]|\.jpg|\.jpeg|\.png|\.webp)", u) else 0
        return (s1, s4, s2, s3)
    normd.sort(key=score, reverse=True)
    return normd

def clean_and_pick_block(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script","style","noscript"]):
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
            if not s: continue
            if any(p.search(s) for p in PATTERNS):
                hits.append(txt.parent)
        return hits
    def lca(nodes):
        if not nodes: return None
        sets = []
        for n in nodes:
            s = set(); cur = n
            while cur is not None:
                s.add(cur); cur = cur.parent
            sets.append(s)
        common = set.intersection(*sets) if sets else set()
        best, depth = None, -1
        for el in common:
            d, cur = 0, el
            while cur is not None:
                d += 1; cur = cur.parent
            if d > depth:
                best, depth = el, d
        return best
    hits = find_hit_nodes(root)
    block = lca(hits) or root
    snippet = str(block)
    if len(snippet) > 120_000:
        candidates = [c for c in block.find_all(["section","div","article"], recursive=True)]
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
    if not imgs: return None
    for u in imgs:
        if u.endswith(".jpg") and "_mn" not in u:
            return u
    return imgs[0]

# ========== LLM 抽出 ==========
SYSTEM_PROMPT = "You are an information extraction engine. Return ONLY valid JSON with the required keys. No explanations."

def llm_extract(snippet: str) -> Dict[str, Any]:
    if not _USE_LLM:
        raise RuntimeError("LLM disabled")
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

# ========== BeautifulSoup フォールバック ==========
def bs_fallback(raw_html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(raw_html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    product_name = re.sub(r"\s*\|\s*中古品の販売・通販ならセカンドストリート.*$", "", title)
    price = ""
    m = re.search(r"¥\s*[\d,]+", raw_html)
    if m: price = m.group(0)
    condition = ""
    for s in soup.stripped_strings:
        if "商品の状態" in s or re.search(r"中古[ABCD]|未使用品?", s):
            mc = re.search(r"(中古[ABCD]|未使用品?)", s)
            if mc:
                condition = mc.group(1); break
    cats = []
    for tag in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(tag.string or "")
            if isinstance(data, dict) and data.get("@type") == "BreadcrumbList":
                for el in data.get("itemListElement", []):
                    nm = el.get("name")
                    if nm and nm not in ["HOME","買いたい","商品詳細"]:
                        cats.append(nm)
        except Exception:
            pass
    model = ""
    mm = re.search(r"\bRB\d+[A-Z]?\b", raw_html)
    if mm: model = mm.group(0)
    ship = ""
    ms = re.search(r"送料[:：]?\s*¥?\s*[\d,]+", raw_html)
    if ms:
        ship = ms.group(0)
        mprice = re.search(r"¥\s*[\d,]+", ship)
        if mprice: ship = mprice.group(0)
    return {
        "images": [],
        "product_name": product_name,
        "model": model,
        "categories": cats,
        "condition": condition,
        "price": price,
        "shipping_fee": ship,
    }

# ========== 1ファイル解析 ==========
def process_file(html_path: Path, use_llm: bool=True) -> Dict[str, Any]:
    raw_html = html_path.read_text(encoding="utf-8", errors="ignore")
    pre_images = harvest_images(raw_html)
    snippet = clean_and_pick_block(raw_html)
    data: Dict[str, Any]
    if use_llm and _USE_LLM:
        try:
            data = llm_extract(snippet)
        except (RateLimitError, BadRequestError):   # type: ignore
            time.sleep(2.0)
            try:
                data = llm_extract(snippet)
            except Exception:
                data = bs_fallback(raw_html)
        except Exception:
            data = bs_fallback(raw_html)
    else:
        data = bs_fallback(raw_html)
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
    if data.get("price") and re.fullmatch(r"\d[\d,]*", data["price"]):
        data["price"] = "¥" + data["price"]
    return data

# ========== フェーズ1: HTML保存 ==========
def run_phase_html(headless: bool, overwrite: bool, delay: float, retries: int) -> None:
    urls = read_all_csv_urls(INPUT_DIR)
    if not urls:
        print("[ERROR] input/*.csv にURLが見つかりません。"); sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0; failed = 0
    with sync_playwright() as p:
        # 軽いステルス：UA, Viewport, locale/timezone
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1366, "height": 900},
            locale="ja-JP", timezone_id="Asia/Tokyo",
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"),
        )
        page = context.new_page()
        total = len(urls)
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{total}] GET {url}")
            ok = False; err = None
            for attempt in range(retries + 1):
                try:
                    save_html(page, url, HTML_DIR, overwrite=overwrite)
                    ok = True; break
                except Exception as e:
                    err = e
                    print(f"  ↳ attempt {attempt+1} failed: {e}")
                    time.sleep(1.0)
            if ok: saved += 1
            else:
                failed += 1
                print(f"[FAIL] {url} : {err}")
            if delay > 0:
                time.sleep(delay)
        browser.close()
    print("\n===== PHASE[HTML] SUMMARY =====")
    print(f"Saved : {saved}")
    print(f"Failed: {failed}")
    print(f"Dir   : {HTML_DIR.resolve()}")
    print("================================\n")

# ========== フェーズ2: 解析→CSV ==========
def run_phase_parse(use_llm: bool=True) -> None:
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
            data = process_file(f, use_llm=use_llm)
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
            time.sleep(0.4)  # 軽いレート調整
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")
    fieldnames = ["file","image","product_name","model","categories","condition","price","shipping_fee"]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[✅ 完了] {len(rows)}件を書き出し: {OUT_CSV}")

# ========== エントリポイント ==========
def main():
    ap = argparse.ArgumentParser(description="2nd STREET: HTML保存→解析まで一気通貫")
    ap.add_argument("--phase", choices=["both","html","parse"], default="both", help="実行フェーズ（既定 both）")
    ap.add_argument("--headless", action="store_true", help="保存時にヘッドレスで実行")
    ap.add_argument("--overwrite", action="store_true", help="既存HTMLを上書き保存")
    ap.add_argument("--delay", type=float, default=0.5, help="保存フェーズの各リクエスト間待機秒（既定0.5）")
    ap.add_argument("--retries", type=int, default=2, help="保存失敗時のリトライ回数（既定2）")
    ap.add_argument("--no-llm", action="store_true", help="LLM抽出を使わず常にBSフォールバックを使用")
    args = ap.parse_args()

    use_llm = (not args.no_llm) and _USE_LLM
    if args.no_llm:
        print("[INFO] --no-llm 指定のため、BeautifulSoupフォールバックのみで解析します。")
    elif not _USE_LLM:
        print("[INFO] OPENAI_API_KEY 未設定のため、BeautifulSoupフォールバックで解析します。")

    if args.phase in ("both","html"):
        run_phase_html(headless=args.headless, overwrite=args.overwrite, delay=args.delay, retries=args.retries)
    if args.phase in ("both","parse"):
        run_phase_parse(use_llm=use_llm)

if __name__ == "__main__":
    main()
