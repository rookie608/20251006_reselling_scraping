# extract_secondstreet_by_saved_html.py
from playwright.sync_api import sync_playwright
from pathlib import Path
import csv, re, sys, argparse, hashlib, time

INPUT_DIR = Path("input")
OUT_DIR   = Path("output")
HTML_DIR  = OUT_DIR / "html"

URL_HEADERS = {"url","URL","Url","link","Link","LINK","商品URL","リンク","商品リンク"}

def read_all_csv_urls(input_dir: Path):
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
                url_col = i
                break
        start = 1 if url_col is not None else 0
        if url_col is None:
            # 先頭行・2行目をざっくり走査してURL列を推測
            if rows and rows[0] and re.match(r"^https?://", rows[0][0].strip()):
                url_col = 0
            else:
                for j, c in enumerate(rows[1] if len(rows) > 1 else []):
                    if re.match(r"^https?://", (c or "").strip()):
                        url_col = j
                        break
                if url_col is None:
                    url_col = 0
        for r in rows[start:]:
            if not r or url_col >= len(r):
                continue
            u = (r[url_col] or "").strip()
            if u.startswith("http"):
                urls.append(u)
    # 重複除去（入力ファイル横断）
    seen, uniq = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq

def build_html_filename(url: str) -> str:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    tail = url.split("/")[-1] or "page"
    safe_tail = re.sub(r"[^A-Za-z0-9_-]+", "_", tail)
    return f"{safe_tail}_{h}.html"

def save_html(page, url: str, html_dir: Path, *, overwrite: bool = False, wait_selector: str = "h1") -> Path:
    """指定URLのpage.content()を保存。既存ファイルがありoverwrite=Falseならスキップ。"""
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

def main():
    ap = argparse.ArgumentParser(description="CSV内URLのページHTMLだけを保存（解析・CSV/JSON出力なし）")
    ap.add_argument("--headless", action="store_true", help="ヘッドレスで実行")
    ap.add_argument("--overwrite", action="store_true", help="既存HTMLを上書き保存する")
    ap.add_argument("--delay", type=float, default=0.5, help="各リクエスト間の待機秒（既定0.5）")
    ap.add_argument("--retries", type=int, default=2, help="保存失敗時のリトライ回数（既定2）")
    args = ap.parse_args()

    urls = read_all_csv_urls(INPUT_DIR)
    if not urls:
        print("[ERROR] input/*.csv にURLが見つかりません。")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    failed = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(
            viewport={"width": 1366, "height": 900},
            locale="ja-JP",
            timezone_id="Asia/Tokyo",
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        page = context.new_page()

        total = len(urls)
        for i, url in enumerate(urls, 1):
            print(f"[{i}/{total}] GET {url}")
            ok = False
            err = None
            for attempt in range(args.retries + 1):
                try:
                    save_html(page, url, HTML_DIR, overwrite=args.overwrite)
                    ok = True
                    break
                except Exception as e:
                    err = e
                    print(f"  ↳ attempt {attempt+1} failed: {e}")
                    time.sleep(1.0)
            if ok:
                saved += 1
            else:
                failed += 1
                print(f"[FAIL] {url} : {err}")
            if args.delay > 0:
                time.sleep(args.delay)

        browser.close()

    print("\n===== SUMMARY =====")
    print(f"Saved : {saved}")
    print(f"Failed: {failed}")
    print(f"Dir   : {HTML_DIR.resolve()}")
    print("===================\n")

if __name__ == "__main__":
    main()
