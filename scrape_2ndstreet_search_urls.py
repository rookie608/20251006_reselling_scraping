# -*- coding: utf-8 -*-
"""
inputフォルダ内のCSVをすべて読み込み、
各ファイルのURLから2nd STREETの検索ページを開き、
shopsIdを含む商品詳細URLを抽出してoutput/secondstreet_results.csvに出力します。
"""

from pathlib import Path
import pandas as pd
import csv, re, time, hashlib
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ========= パス設定 =========
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR = OUTPUT_DIR / "html"
HTML_DIR.mkdir(parents=True, exist_ok=True)

RESULT_CSV = OUTPUT_DIR / "secondstreet_results.csv"
MAP_CSV = OUTPUT_DIR / "intermediate_url_html.csv"

# ========= 設定 =========
SAVE_HTML = True
DETAIL_URL_RE = re.compile(r"/goods/detail/goodsId/\d+/shopsId/\d+", re.IGNORECASE)
SEARCH_RESULT_LINK_SEL = "a[href*='/goods/detail/goodsId/'][href*='/shopsId/']"
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# ========= 関数 =========
def find_urls_in_row(row):
    urls = []
    for cell in row:
        if pd.isna(cell): continue
        urls += URL_RE.findall(str(cell))
    return urls

def normalize_url(base, href):
    return href if href.startswith("http") else base.split("/search")[0].rstrip("/") + href

def extract_items(page):
    items, seen = [], set()
    anchors = page.locator(SEARCH_RESULT_LINK_SEL)
    for i in range(anchors.count()):
        a = anchors.nth(i)
        href = a.get_attribute("href") or ""
        if not href: continue
        url = normalize_url(page.url, href)
        if not DETAIL_URL_RE.search(url) or url in seen:
            continue
        seen.add(url)
        title = (a.inner_text() or "").strip()
        if not title:
            try:
                title = (a.locator("xpath=..").inner_text() or "").strip()
            except: title = ""
        if len(title) > 200:
            title = title[:200] + "…"
        items.append({"title": title, "url": url})
    return items

def save_html(url, html_text):
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]
    fname = f"search_{digest}.html"
    path = HTML_DIR / fname
    path.write_text(html_text, encoding="utf-8", errors="ignore")
    return fname

# ========= メイン処理 =========
def main():
    csv_files = list(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        print("[ERROR] inputフォルダにCSVが見つかりません。")
        return

    print(f"[INFO] {len(csv_files)}件のCSVを処理します。")

    # 出力初期化
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["src_file","src_row","search_page_url","rank","title","url"]).writeheader()
    if SAVE_HTML:
        with MAP_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["search_page_url","html_filename"]).writeheader()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width":1280,"height":900},user_agent=UA,locale="ja-JP",timezone_id="Asia/Tokyo")
        page = context.new_page()

        count_all = 0
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, dtype=str, keep_default_na=False, encoding="utf-8", engine="python")
            for idx, row in df.iterrows():
                urls = find_urls_in_row(row.values)
                for url in urls:
                    if "www.2ndstreet.jp/search" not in url: continue
                    count_all += 1
                    print(f"[{count_all:04}] {csv_file.name} row={idx+1} → {url}")
                    try:
                        page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    except PWTimeout:
                        print("  [WARN] タイムアウト、続行します。")
                    time.sleep(1.2)

                    if SAVE_HTML:
                        try:
                            html_name = save_html(url, page.content())
                            with MAP_CSV.open("a", newline="", encoding="utf-8") as f:
                                csv.DictWriter(f, fieldnames=["search_page_url","html_filename"]).writerow(
                                    {"search_page_url": url, "html_filename": html_name})
                        except Exception as e:
                            print(f"  [WARN] HTML保存失敗: {e}")

                    try:
                        items = extract_items(page)
                    except Exception as e:
                        print(f"  [ERROR] 抽出失敗: {e}")
                        continue

                    print(f"  [INFO] {len(items)}件検出")

                    with RESULT_CSV.open("a", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=["src_file","src_row","search_page_url","rank","title","url"])
                        for rank, it in enumerate(items, start=1):
                            writer.writerow({
                                "src_file": csv_file.name,
                                "src_row": idx+1,
                                "search_page_url": url,
                                "rank": rank,
                                "title": it["title"],
                                "url": it["url"]
                            })

        print(f"\n[✅ 完了] {count_all}件のURLを処理。結果: {RESULT_CSV}")
        if SAVE_HTML:
            print(f"[✅ HTML保存] {MAP_CSV} / {HTML_DIR}")

if __name__ == "__main__":
    main()
