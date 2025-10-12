# extract_secondstreet_by_saved_html.py
from playwright.sync_api import sync_playwright
from pathlib import Path
from bs4 import BeautifulSoup
import csv, re, json, sys, argparse, hashlib

INPUT_DIR = Path("input")
OUT_DIR   = Path("output")
HTML_DIR  = OUT_DIR / "html"
OUT_CSV   = OUT_DIR / "products.csv"
OUT_JSONL = OUT_DIR / "products.jsonl"

URL_HEADERS = {"url","URL","Url","link","Link","LINK","商品URL","リンク","商品リンク"}

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def read_all_csv_urls(input_dir: Path):
    urls = []
    for p in input_dir.glob("*.csv"):
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.reader(f))
        if not rows: continue
        header = [c.strip() for c in rows[0]]
        url_col = None
        for i, h in enumerate(header):
            if h in URL_HEADERS or h.lower() in {x.lower() for x in URL_HEADERS}:
                url_col = i; break
        start = 1 if url_col is not None else 0
        if url_col is None:
            if re.match(r"^https?://", rows[0][0].strip()):
                url_col = 0
            else:
                for j, c in enumerate(rows[1] if len(rows)>1 else []):
                    if re.match(r"^https?://", c.strip()):
                        url_col = j; break
                if url_col is None: url_col = 0
        for r in rows[start:]:
            if not r or url_col>=len(r): continue
            u = r[url_col].strip()
            if u.startswith("http"): urls.append(u)
    # 重複除去
    seen, uniq = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq

def save_html(page, url: str, html_dir: Path) -> Path:
    """h1が出るまで待ってから page.content() を保存"""
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    try:
        page.wait_for_selector("h1", timeout=8000)
    except Exception:
        pass
    html = page.content()
    # URLベースのファイル名
    h = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", url.split("/")[-1] or "page")
    fname = f"{name}_{h}.html"
    path = html_dir / fname
    path.write_text(html, encoding="utf-8")
    return path

def find_ancestor(tag, names=("section","div","main","article")):
    cur = tag
    while cur and cur.name not in names:
        cur = cur.parent
    return cur or tag

def parse_saved_html(html_path: Path):
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")

    # --- title（これまでOKな方法） ---
    title_tag = (
        soup.select_one("h1") or
        soup.select_one(".p-goodsDetail__title") or
        soup.select_one('[data-testid="goods-title"]') or
        soup.select_one(".p-goodsDetail__head")
    )
    title = norm(title_tag.get_text(" ", strip=True)) if title_tag else ""

    # --- 右枠：h1から近いコンテナ or 「商品の状態とは？」リンクの親 ---
    link_state = soup.find("a", string=re.compile("商品の状態とは"))
    right_box = find_ancestor(link_state) if link_state else find_ancestor(title_tag) if title_tag else soup.body
    right_text = norm(right_box.get_text(" ", strip=True))

    # 送料（右枠内の最初の1件）
    shipping_label = ""
    m_ship = re.search(r"送料[:：]?\s*[¥￥]\s*[0-9,]{2,}\s*税込?", right_text)
    if m_ship:
        shipping_label = m_ship.group(0).replace("￥", "¥")

    # 価格（右枠内。送料を含む行を除外して最大値）
    price_label = ""
    yen_re = re.compile(r"[¥￥]\s*([0-9,]{3,})")
    lines = [ln.strip() for ln in right_text.split() if ln.strip()]
    # 行概念が潰れている可能性があるため、一旦右枠内の要素単位で走査
    candidates = []
    for el in right_box.find_all(True):
        t = norm(el.get_text(" ", strip=True))
        if not t or "送料" in t:  # 送料行除外
            continue
        m = yen_re.search(t)
        if m:
            val = int(m.group(1).replace(",", ""))
            label = m.group(0).replace("￥", "¥")
            if "税込" in t and "税込" not in label:
                label += " 税込"
            candidates.append((val, label))
    if candidates:
        price_label = max(candidates, key=lambda x: x[0])[1]

    # 状態（右枠優先：テキスト or ボタン）
    condition = ""
    m_cond = re.search(r"商品の状態[:：]?\s*(未使用品|中古A|中古B|中古C|中古D)", right_text)
    if m_cond:
        condition = m_cond.group(1)
    if not condition:
        # 「商品の状態」を含む最寄りセクションを探してボタン文字列を拾う
        sec = None
        for c in right_box.find_all(["section","div","article","main"]):
            if "商品の状態" in norm(c.get_text(" ", strip=True)):
                sec = c; break
        if sec:
            opts = [norm(b.get_text(" ", strip=True)) for b in sec.find_all(["button","a","span","div"])]
            for k in ["未使用品","中古A","中古B","中古C","中古D"]:
                if k in opts:
                    condition = k; break

    # 左枠：『似た商品を探す』リンクの親ボックス内のimg
    image_urls = []
    link_similar = soup.find("a", string=re.compile("似た商品を探す"))
    left_box = find_ancestor(link_similar) if link_similar else soup
    imgs = left_box.find_all("img")
    for img in imgs:
        u = img.get("src") or img.get("data-src") or ""
        if u and re.search(r"\.(jpg|jpeg|png|webp)(\?|$)", u, re.I):
            # 絶対URL化（相対が来た場合）
            if u.startswith("//"):
                u = "https:" + u
            elif u.startswith("/"):
                # ベースURLをファイル名からは作れないので、そのまま相対を残すか、後段で補完
                pass
            image_urls.append(u)
    # 重複除去
    seen, uniq = set(), []
    for u in image_urls:
        if u not in seen:
            seen.add(u); uniq.append(u)
    image_urls = uniq

    return {
        "title": title,
        "price_label": price_label,
        "shipping_label": shipping_label,
        "condition": condition,
        "image_urls": image_urls
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    urls = read_all_csv_urls(INPUT_DIR)
    if not urls:
        print("[ERROR] input/*.csv にURLが見つかりません。"); sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        context = browser.new_context(
            viewport={"width": 1366, "height": 900},
            locale="ja-JP", timezone_id="Asia/Tokyo",
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36")
        )
        page = context.new_page()

        for i, url in enumerate(urls, 1):
            rec = {"url": url, "title":"", "price_label":"", "shipping_label":"", "condition":"", "image_urls":[], "error":""}
            try:
                html_path = save_html(page, url, HTML_DIR)
                data = parse_saved_html(html_path)
                rec.update(data)
                print(f"[{i}/{len(urls)}] {url} → {rec['price_label']} | {rec['shipping_label']} | {rec['condition']}")
            except Exception as e:
                rec["error"] = str(e)
                print(f"[{i}/{len(urls)}] ERROR: {e}")
            results.append(rec)

        browser.close()

    # 出力
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["url","title","price_label","shipping_label","condition","image_urls","error"])
        w.writeheader()
        for r in results:
            row = dict(r)
            if isinstance(row["image_urls"], list):
                row["image_urls"] = " ".join(row["image_urls"])
            w.writerow(row)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ HTML保存: {HTML_DIR}\n✅ 出力: {OUT_CSV} / {OUT_JSONL}")

if __name__ == "__main__":
    main()
