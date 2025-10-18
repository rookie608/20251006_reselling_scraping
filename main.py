# scrape_secondstreet_pipeline.py
# -*- coding: utf-8 -*-
"""
2nd STREET 一気通貫パイプライン：
  input/*.csv にある「検索ページURL」を開いて商品詳細URL(shopsId含む)を抽出
   → 商品ページHTMLを保存
   → HTMLを解析して output/url_map_YYYYMMDDHHMM.csv へ出力（最終成果はこの1本）
   → （追加）url_map_YYYYMMDDHHMM.csv を読み込み、OpenAIで検索キーワード列 search_keyword を生成して
      output/result_with_まとめ_YYYYMMDDHHMM.csv を出力

出力：
- output/secondstreet_results_YYYYMMDDHHMM.csv : 検索ページ→商品URLの抽出結果（順位つき）
- output/url_map_YYYYMMDDHHMM.csv              : 商品URL↔HTML対応 + 解析情報（最終成果）
- output/result_with_まとめ_YYYYMMDDHHMM.csv   : url_map 上に search_keyword 列を付与したもの
- output/html/*.html                           : 保存した商品ページ（必要に応じて検索ページも）
"""

import os, re, csv, json, time, argparse, hashlib, urllib.parse, sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable, Set
from datetime import datetime

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from bs4 import BeautifulSoup, Comment

# =========（キーワード生成用 追加）=========
try:
    from dotenv import load_dotenv  # optional
except Exception:
    load_dotenv = None  # type: ignore

# ========== パス ==========
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
OUT_DIR   = BASE_DIR / "output"
HTML_DIR  = OUT_DIR / "html"

# 実行時刻サフィックス（例: 202510151000）
RUN_STAMP = datetime.now().strftime("%Y%m%d%H%M")

# タイムスタンプ付きの出力ファイル名
RESULT_SEARCH_CSV = OUT_DIR / f"secondstreet_results_{RUN_STAMP}.csv"   # 検索→商品URLの記録
OUT_MAP           = OUT_DIR / f"url_map_{RUN_STAMP}.csv"               # 最終成果（解析後もこれを上書き）
OUT_CSV_KW        = OUT_DIR / f"result_with_まとめ_{RUN_STAMP}.csv"     # 検索ワード付き最終CSV

# ========== 設定 ==========
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

SEARCH_PAGE_HINT = "www.2ndstreet.jp/search"
DETAIL_URL_RE = re.compile(r"/goods/detail/goodsId/\d+/shopsId/\d+", re.IGNORECASE)
SEARCH_RESULT_LINK_SEL = "a[href*='/goods/detail/goodsId/'][href*='/shopsId/']"
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)

# ========== OpenAI（任意・解析フェーズ用） ==========
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

# ========== 共通ユーティリティ ==========
def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)

def find_urls_in_row(cells: Iterable[Any]) -> List[str]:
    urls: List[str] = []
    for cell in cells:
        if pd.isna(cell):
            continue
        urls += URL_RE.findall(str(cell))
    return urls

def normalize_url(base: str, href: str) -> str:
    return href if href.startswith("http") else base.split("/search")[0].rstrip("/") + href

def extract_items_from_search(page) -> List[Dict[str, str]]:
    items, seen = [], set()
    anchors = page.locator(SEARCH_RESULT_LINK_SEL)
    try:
        count = anchors.count()
    except Exception:
        count = 0
    for i in range(count):
        a = anchors.nth(i)
        href = a.get_attribute("href") or ""
        if not href:
            continue
        url = normalize_url(page.url, href)
        if not DETAIL_URL_RE.search(url) or url in seen:
            continue
        seen.add(url)
        title = ""
        try:
            title = (a.inner_text() or "").strip()
            if not title:
                title = (a.locator("xpath=..").inner_text() or "").strip()
        except Exception:
            title = ""
        if len(title) > 200:
            title = title[:200] + "…"
        items.append({"title": title, "url": url})
    return items

def build_html_filename(url: str, prefix: str = "prod") -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    tail = url.split("/")[-1] or "page"
    safe_tail = re.sub(r"[^A-Za-z0-9_-]+", "_", tail)
    return f"{prefix}_{safe_tail}_{h}.html"

def save_html(page, url: str, *, wait_selector: Optional[str] = "h1", overwrite: bool=False, prefix: str="prod") -> Path:
    fname = build_html_filename(url, prefix=prefix)
    path = HTML_DIR / fname
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
    path.write_text(html, encoding="utf-8", errors="ignore")
    print(f"[SAVE] {url} → {path.name}")
    return path

def normalize_img_url(url: str, base: str = "https://www.2ndstreet.jp/") -> str:
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
        for attr in ("src", "data-src", "data-original", "data-lazy", "data-image"):
            u = img.get(attr)
            if not u: continue
            if ("goods" in u) or u.endswith((".jpg", ".jpeg", ".png", ".webp")):
                cand.add(u)
    normd, seen = [], set()
    for u in cand:
        nu = normalize_img_url(u)
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

# ========== 金額正規化ユーティリティ ==========
_MONEY_NUM_RE = re.compile(r"(\d[\d,]*)")
def _to_int_or_empty(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    if "無料" in s or "送料無料" in s:
        return "0"
    m = _MONEY_NUM_RE.search(s)
    if not m:
        return ""
    return str(int(m.group(1).replace(",", "")))

# ========== Brand helper ==========
KNOWN_BRANDS = [
    "Ray-Ban","RAYBAN","RAY-BAN","GUCCI","COACH","PRADA","FENDI","CELINE","CHANEL",
    "HERMES","LOUIS VUITTON","VUITTON","BOTTEGA VENETA","BOTTEGA","SAINT LAURENT",
    "YVES SAINT LAURENT","DIOR","BURBERRY","MIU MIU","BALENCIAGA","MONCLER","NIKE",
    "ADIDAS","THE NORTH FACE","SUPREME","NEW BALANCE","ASICS","OAKLEY","TOM FORD",
    "GENTLE MONSTER","OLIVER PEOPLES","EYEVAN","JINS","ZOFF","KATE SPADE"
]

def extract_brand(raw_html: str, soup: BeautifulSoup) -> str:
    # 1) JSON-LD Product > brand.name
    for tag in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(tag.string or "")
            if isinstance(data, dict):
                datas = [data]
            elif isinstance(data, list):
                datas = data
            else:
                datas = []
            for d in datas:
                if isinstance(d, dict) and d.get("@type") in ("Product","Offer","AggregateOffer"):
                    b = d.get("brand")
                    if isinstance(b, dict):
                        nm = b.get("name")
                        if nm: return nm.strip()
                    elif isinstance(b, str) and b.strip():
                        return b.strip()
        except Exception:
            pass

    # 2) ラベル類
    cand_nodes = soup.select("h1, h2, h3, .brand, [class*='brand'], [data-brand]")
    for n in cand_nodes:
        txt = (n.get_text(" ", strip=True) or "")
        if 1 <= len(txt) <= 40:
            for kb in KNOWN_BRANDS:
                if kb.lower() in txt.lower():
                    return kb
            if re.match(r"^[A-Za-z][\w\s\-.&']{1,38}$", txt):
                return txt

    # 3) <title> 先頭語
    title = (soup.title.get_text(strip=True) if soup.title else "") or ""
    m = re.match(r"^\s*([A-Za-z][\w\-. '&]{1,38})\b", title)
    if m:
        head = m.group(1)
        for kb in KNOWN_BRANDS:
            if kb.lower() in head.lower():
                return kb
        if len(head) >= 2:
            return head

    # 4) 既知ブランド出現
    low = raw_html.lower()
    for kb in KNOWN_BRANDS:
        if kb.lower() in low:
            return kb

    return ""

# ========== LLM抽出（解析） ==========
SYSTEM_PROMPT = "You are an information extraction engine. Return ONLY valid JSON with the required keys. No explanations."

def llm_extract(snippet: str) -> Dict[str, Any]:
    if not _USE_LLM:
        raise RuntimeError("LLM disabled")
    USER_PROMPT = """以下は中古販売サイト「セカンドストリート」の商品HTMLです。
次のキーだけを含む厳密なJSONを返してください（余計なキーは禁止）:

{
  "images": string[],
  "product_name": string,
  "brand": string,
  "model": string,
  "categories": string[],
  "condition": string,
  "price": string,
  "shipping_fee": string
}

- 値はページ表記を尊重（例：¥6,490 / 送料：¥770 / 送料無料 など）。ただし数値化は後段で行います。
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

    brand = extract_brand(raw_html, soup)

    price_raw = ""
    m = re.search(r"¥\s*[\d,]+", raw_html)
    if m: price_raw = m.group(0)

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
    mm = re.search(r"\bRB\d+[A-Z]?\b", raw_html, flags=re.IGNORECASE)
    if mm: model = mm.group(0)

    ship_raw = ""
    ms = re.search(r"(送料[^<]{0,10})?¥?\s*[\d,]+|送料無料", raw_html)
    if ms:
        ship_raw = ms.group(0)

    return {
        "images": [],
        "product_name": product_name,
        "brand": brand,
        "model": model,
        "categories": cats,
        "condition": condition,
        "price": price_raw,
        "shipping_fee": ship_raw,
    }

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
            nu = normalize_img_url(u)
            if nu and nu not in seen:
                seen.add(nu); uniq.append(nu)
        images = uniq
    primary = pick_primary(images)
    data["images"] = [primary] if primary else []

    data["price"] = _to_int_or_empty(data.get("price", ""))
    data["shipping_fee"] = _to_int_or_empty(data.get("shipping_fee", ""))

    if not data.get("brand"):
        soup = BeautifulSoup(raw_html, "lxml")
        data["brand"] = extract_brand(raw_html, soup)

    return data

# ========== CSV I/O ==========
BASE_HEADERS   = ["url","file","status","saved_at"]
ENRICH_HEADERS = ["image","product_name","brand","model","categories","condition","price","shipping_fee"]

def write_search_results_header():
    with RESULT_SEARCH_CSV.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["src_file","src_row","search_page_url","rank","title","url"]).writeheader()

def append_search_results(rows: List[Dict[str, Any]]):
    with RESULT_SEARCH_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["src_file","src_row","search_page_url","rank","title","url"])
        w.writerows(rows)

def write_url_map(rows: List[Dict[str, str]]) -> None:
    with OUT_MAP.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=BASE_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

def read_url_map() -> List[Dict[str, str]]:
    if not OUT_MAP.exists():
        return []
    with OUT_MAP.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return list(reader)

def overwrite_url_map_with_enriched(rows: List[Dict[str, str]]) -> None:
    headers = BASE_HEADERS + ENRICH_HEADERS
    with OUT_MAP.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

# ========== ★変更(重複排除): 直近の secondstreet_results_*.csv を探してURL集合を得る ==========
RESULTS_NAME_RE = re.compile(r"secondstreet_results_(\d{12})\.csv$", re.IGNORECASE)

def _find_latest_previous_results_path(exclude_path: Path) -> Optional[Path]:
    """
    output/secondstreet_results_*.csv から、末尾12桁の日時を見て最新を返す。
    今回の実行で書いている RESULT_SEARCH_CSV は除外。
    """
    cands = []
    for p in OUT_DIR.glob("secondstreet_results_*.csv"):
        if p.resolve() == exclude_path.resolve():
            continue
        m = RESULTS_NAME_RE.search(p.name)
        if m:
            cands.append((m.group(1), p))
    if not cands:
        return None
    # stampが大きい=新しい
    cands.sort(key=lambda t: t[0], reverse=True)
    return cands[0][1]

def _read_url_set_from_results(path: Path) -> Set[str]:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)  # encodingフォールバック
    if "url" not in df.columns:
        return set()
    return set(str(u) for u in df["url"].dropna().astype(str))

# ========== ここからメイン処理 ==========
def main():
    ap = argparse.ArgumentParser(description="2nd STREET: 検索URL→商品URL抽出→HTML保存→解析（shopsId付き）")
    ap.add_argument("--headless", action="store_true", help="ブラウザをヘッドレスで実行")
    ap.add_argument("--overwrite", action="store_true", help="既存HTMLを上書き保存")
    ap.add_argument("--delay", type=float, default=0.5, help="各リクエスト間待機秒（既定0.5）")
    ap.add_argument("--retries", type=int, default=2, help="保存失敗時のリトライ回数（既定2）")
    ap.add_argument("--no-llm", action="store_true", help="LLM抽出を使わず常にBSフォールバックを使用")
    ap.add_argument("--save-search-html", action="store_true", help="検索ページHTMLも保存する（デフォルトは保存しない）")
    args = ap.parse_args()

    use_llm = (not args.no_llm) and _USE_LLM
    if args.no_llm:
        print("[INFO] --no-llm 指定：BeautifulSoupフォールバックのみで解析します。")
    elif not _USE_LLM:
        print("[INFO] OPENAI_API_KEY 未設定：BeautifulSoupフォールバックで解析します。")

    ensure_dirs()

    # 1) input/*.csv 読み込み → 検索ページURLの抽出
    csv_files = list(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        print("[ERROR] inputフォルダにCSVが見つかりません。"); sys.exit(1)

    print(f"[INFO] {len(csv_files)}件のCSVを処理します。")
    write_search_results_header()

    search_urls: List[Dict[str, Any]] = []  # {src_file, src_row, url}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, dtype=str, keep_default_na=False, encoding="utf-8", engine="python")
        for idx, row in df.iterrows():
            urls = find_urls_in_row(row.values)
            for u in urls:
                if SEARCH_PAGE_HINT in u:
                    search_urls.append({"src_file": csv_file.name, "src_row": idx+1, "url": u})

    if not search_urls:
        print("[ERROR] 検索ページURLが見つかりませんでした（www.2ndstreet.jp/search が0件）。"); sys.exit(1)

    # 2) 検索ページ→商品URL抽出
    product_urls: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=UA, locale="ja-JP", timezone_id="Asia/Tokyo"
        )
        page = context.new_page()

        for i, entry in enumerate(search_urls, 1):
            s_url = entry["url"]
            print(f"[SEARCH {i}/{len(search_urls)}] {s_url}")
            try:
                page.goto(s_url, wait_until="domcontentloaded", timeout=60000)
            except PWTimeout:
                print("  [WARN] タイムアウト、続行します。")
            time.sleep(1.0)

            # （任意）検索ページHTML保存
            if args.save_search_html:
                try:
                    save_html(page, s_url, wait_selector=None, overwrite=args.overwrite, prefix="search")
                except Exception as e:
                    print(f"  [WARN] 検索HTML保存失敗: {e}")

            # 商品URL抽出
            rows_to_write = []
            try:
                items = extract_items_from_search(page)
                print(f"  [INFO] {len(items)}件検出")
                for rank, it in enumerate(items, start=1):
                    rows_to_write.append({
                        "src_file": entry["src_file"],
                        "src_row": entry["src_row"],
                        "search_page_url": s_url,
                        "rank": rank,
                        "title": it["title"],
                        "url": it["url"]
                    })
                    product_urls.append(it["url"])
            except Exception as e:
                print(f"  [ERROR] 抽出失敗: {e}")

            if rows_to_write:
                append_search_results(rows_to_write)

            if args.delay > 0:
                time.sleep(args.delay)

        browser.close()

    # ★変更(重複排除): 前回最新の secondstreet_results_*.csv の URL と比較して、新規のみ残す
    prev_path = _find_latest_previous_results_path(RESULT_SEARCH_CSV)
    prev_urls: Set[str] = set()
    if prev_path:
        prev_urls = _read_url_set_from_results(prev_path)
        print(f"[INFO] 直近の結果: {prev_path.name}（URL {len(prev_urls)}件）を読み込み、重複を除外します。")
    else:
        print("[INFO] 過去の secondstreet_results_*.csv が見つかりません（初回として全件処理）。")

    # 現在抽出の重複除去（同一URLの二重検出も排除）
    uniq_current: List[str] = []
    seen_cur = set()
    for u in product_urls:
        if u in seen_cur:
            continue
        seen_cur.add(u)
        uniq_current.append(u)

    # 過去との重複排除
    new_only: List[str] = [u for u in uniq_current if u not in prev_urls]
    removed = len(uniq_current) - len(new_only)
    print(f"[INFO] 今回抽出 {len(uniq_current)} 件 / 過去と重複 {removed} 件 → 新規 {len(new_only)} 件")

    if not new_only:
        print("[WARN] 新規URLがありません。以降のHTML保存・解析はスキップします。")
        # 何もせず正常終了とする
        print(f"[WRITE] {RESULT_SEARCH_CSV.resolve()}（抽出ログのみ更新）")
        return

    # 3) 商品ページHTML保存 → url_map_*.csv（素状態）作成（★新規URLのみ）
    print(f"[INFO] 商品ページ保存を開始（新規 {len(new_only)}件）")
    map_rows: List[Dict[str, str]] = []
    saved = failed = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(
            viewport={"width": 1366, "height": 900},
            user_agent=UA, locale="ja-JP", timezone_id="Asia/Tokyo"
        )
        page = context.new_page()

        for i, url in enumerate(new_only, 1):
            print(f"[PROD {i}/{len(new_only)}] GET {url}")
            fname = build_html_filename(url, prefix="prod")
            status = "saved"
            ok = False; err: Optional[Exception] = None
            for attempt in range(args.retries + 1):
                try:
                    save_html(page, url, wait_selector="h1", overwrite=args.overwrite, prefix="prod")
                    ok = True; break
                except Exception as e:
                    err = e
                    print(f"  ↳ attempt {attempt+1} failed: {e}")
                    time.sleep(1.0)
            if ok:
                saved += 1
            else:
                failed += 1
                status = f"fail:{type(err).__name__}" if err else "fail"
                print(f"[FAIL] {url} : {err}")
            if args.delay > 0:
                time.sleep(args.delay)

            map_rows.append({
                "url": url,
                "file": fname if ok or (HTML_DIR / fname).exists() else "",
                "status": status if (ok or status.startswith("fail")) else "skipped",
                "saved_at": now,
            })

        browser.close()

    write_url_map(map_rows)
    print("\n===== SAVE SUMMARY =====")
    print(f"Saved : {saved}")
    print(f"Failed: {failed}")
    print(f"Dir   : {HTML_DIR.resolve()}")
    print(f"Map   : {OUT_MAP.resolve()}")
    print("========================\n")

    # 4) 解析 → url_map_* に情報付与で上書き
    print("[INFO] 解析フェーズを開始します")
    RESULT_HEADERS = BASE_HEADERS + ENRICH_HEADERS

    enriched_rows: List[Dict[str, str]] = []
    proc = skip = 0
    map_rows = read_url_map()
    total = len(map_rows)
    if not map_rows:
        print(f"[WARN] 中間CSVが空です: {OUT_MAP}")

    for i, row in enumerate(map_rows, 1):
        url = row.get("url", "")
        fname = row.get("file", "")
        status = row.get("status", "")
        base = {k: row.get(k, "") for k in BASE_HEADERS}

        if not fname or status.startswith("fail"):
            skip += 1
            print(f"[SKIP] {i}/{total} {url} （fileなし/失敗行）")
            base.update({k: "" for k in ENRICH_HEADERS})
            enriched_rows.append(base)
            continue

        fpath = HTML_DIR / fname
        if not fpath.exists():
            skip += 1
            print(f"[SKIP] {i}/{total} {url} （HTML未存在: {fname}）")
            base.update({k: "" for k in ENRICH_HEADERS})
            enriched_rows.append(base)
            continue

        try:
            data = process_file(fpath, use_llm=use_llm)
            proc += 1
            base.update({
                "image": (data.get("images") or [""])[0],
                "product_name": data.get("product_name", ""),
                "brand": data.get("brand", ""),
                "model": data.get("model", ""),
                "categories": " > ".join(data.get("categories", [])),
                "condition": data.get("condition", ""),
                "price": data.get("price", ""),
                "shipping_fee": data.get("shipping_fee", ""),
            })
            enriched_rows.append(base)
            print(f"[OK] {i}/{total} {fname}")
            time.sleep(0.2)
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")
            base.update({k: "" for k in ENRICH_HEADERS})
            enriched_rows.append(base)

    overwrite_url_map_with_enriched(enriched_rows)

    print(f"\n[✅ 完了] 解析 {proc} 件 / スキップ {skip} 件")
    print(f"[WRITE] {OUT_MAP.resolve()}（商品情報 付与済）")
    print(f"[WRITE] {RESULT_SEARCH_CSV.resolve()}（検索→商品URL 抽出ログ）")

    # 5) （追加）検索キーワード生成 → result_with_まとめ_*.csv
    try:
        append_search_keywords()
    except Exception as e:
        print(f"[WARN] 検索キーワード生成に失敗しました: {e}")
        # 最低限、検索キーワード列を空で書き出す（OUT_MAPベース）
        try:
            df_tmp = pd.read_csv(OUT_MAP)
            if "search_keyword" not in df_tmp.columns:
                df_tmp["search_keyword"] = ""
            df_tmp.to_csv(OUT_CSV_KW, index=False, encoding="utf-8-sig")
            print(f"[INFO] フォールバック出力: {OUT_CSV_KW.resolve()}")
        except Exception as e2:
            print(f"[ERROR] フォールバック出力も失敗: {e2}")

# ======== （追加）検索キーワード生成ロジック ========
KW_MODEL = "gpt-4.1-mini"
KW_MAX_RETRIES = 5
KW_RETRY_BASE_WAIT = 2.0  # 秒

KW_SYSTEM_PROMPT = """あなたは中古EC（メルカリ・ラクマ）向けの検索キーワード生成アシスタントです。
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

def _kw_build_user_prompt(brand: str, product_name: str) -> str:
    return f"""brand: {brand}
product_name: {product_name}

上記に対して、出力ルールに従い検索キーワードを1行で出力してください。"""

def _get_kw_client() -> Optional["OpenAI"]:
    if 'client' in globals() and isinstance(globals()['client'], object):
        return globals().get('client')  # type: ignore
    if load_dotenv:
        try:
            load_dotenv()
        except Exception:
            pass
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

def _generate_keyword(client_obj: "OpenAI", brand: str, product_name: str) -> str:
    brand = (brand or "").strip()
    product_name = (product_name or "").strip()
    if not brand and not product_name:
        return ""

    prompt = _kw_build_user_prompt(brand, product_name)

    for attempt in range(1, KW_MAX_RETRIES + 1):
        try:
            resp = client_obj.chat.completions.create(
                model=KW_MODEL,
                messages=[
                    {"role": "system", "content": KW_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=40,
            )
            text = (resp.choices[0].message.content or "").strip()
            return " ".join(text.split())
        except Exception as e:
            wait = KW_RETRY_BASE_WAIT * (2 ** (attempt - 1))
            print(f"[WARN] OpenAI API エラー {e} → {wait:.1f}s 待機 ({attempt}/{KW_MAX_RETRIES})")
            time.sleep(wait)

    return brand

def append_search_keywords() -> None:
    """
    OUT_MAP（url_map_*.csv）を読み込み、brand/product_name を用いて
    search_keyword を生成し、result_with_まとめ_*.csv に保存。
    """
    if not OUT_MAP.exists():
        raise FileNotFoundError(f"{OUT_MAP} が見つかりません。前段の解析が成功しているか確認してください。")

    print(f"[INFO] キーワード生成: 読み込み {OUT_MAP}")
    df = pd.read_csv(OUT_MAP, dtype=str, keep_default_na=False)

    # 列の存在保証
    if "brand" not in df.columns:
        df["brand"] = ""
    if "product_name" not in df.columns:
        df["product_name"] = ""

    cli = _get_kw_client()
    if cli is None:
        print("[WARN] OPENAI_API_KEY 未設定または初期化失敗。brand をそのまま search_keyword に使用します。")
        df["search_keyword"] = df["brand"].fillna("").astype(str)
        df.to_csv(OUT_CSV_KW, index=False, encoding="utf-8-sig")
        print(f"[✅ 完了]（フォールバック）{OUT_CSV_KW.resolve()}")
        return

    keywords: List[str] = []
    for idx, row in df.iterrows():
        kw = _generate_keyword(cli, str(row.get("brand","")), str(row.get("product_name","")))
        keywords.append(kw)
        print(f"[KW {idx+1}/{len(df)}] {kw}")

    df["search_keyword"] = keywords
    df.to_csv(OUT_CSV_KW, index=False, encoding="utf-8-sig")
    print(f"[✅ 完了] キーワード付き出力: {OUT_CSV_KW.resolve()}")

if __name__ == "__main__":
    main()
