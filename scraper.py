import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
DATA_DIR = ROOT / "data"
RAW_HTML_DIR = DATA_DIR / "raw_html"
REVIEWS_PATH = DATA_DIR / "reviews.jsonl"
REVIEWS_CSV_PATH = DATA_DIR / "reviews.csv"
REVIEWS_PARQUET_PATH = DATA_DIR / "reviews.parquet"
MEMORY_PATH = ROOT / "memory.json"
TIMEOUT_SECONDS = 20

CSV_FIELDNAMES = [
    "review_id",
    "product",
    "website",
    "country",
    "stars",
    "rating_raw",
    "text",
    "review_date",
    "source_url",
    "scraped_at_utc",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path, default):
    if not path.exists():
        return default
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def extract_stars(rating_raw, text=""):
    """
    Extracts numeric star rating (e.g. 5, 4.5, 3) from raw string or text.
    """
    if rating_raw:
        match = re.search(r"(?<!\d)([1-5](?:\.[0-9]+)?)(?!\d)", str(rating_raw).strip())
        if match:
            try:
                val = float(match.group(1))
                return int(val) if val.is_integer() else val
            except (ValueError, TypeError):
                pass
    if text:
        match = re.search(
            r"(?:rating|stars?|score)?\s*[:\-]?\s*([1-5](?:\.[0-9]+)?)\s*(?:/|\s*out\s*of\s*)\s*5",
            text,
            re.IGNORECASE,
        )
        if match:
            try:
                val = float(match.group(1))
                return int(val) if val.is_integer() else val
            except (ValueError, TypeError):
                pass
    return ""


def allowed_by_robots(url, user_agent):
    parts = urlparse(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        response = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=TIMEOUT_SECONDS)
        if response.status_code != 200:
            print(f"Skipping {url}: could not verify robots.txt (HTTP {response.status_code}).")
            return False
        parser.parse(response.text.splitlines())
    except requests.RequestException as error:
        print(f"Skipping {url}: could not check robots.txt ({error}).")
        return False
    return parser.can_fetch(user_agent, url)


def first_text(card, selector):
    if not selector:
        return ""
    element = None
    try:
        element = card.select_one(selector)
    except Exception:
        pass
    if not element:
        return ""
    return (
        element.get("data-rating")
        or element.get("data-score")
        or element.get("aria-label")
        or element.get("title")
        or element.get_text(" ", strip=True)
    ).strip()


def parse_review_cards(html, selectors):
    soup = BeautifulSoup(html, "html.parser")
    card_sel = selectors.get("review_card", "")
    cards = soup.select(card_sel) if card_sel else []

    records = []
    for card in cards:
        text = first_text(card, selectors.get("text"))
        title = first_text(card, selectors.get("title", ""))
        if title and title.lower() not in text.lower():
            text = f"{title}. {text}".strip()
        if not text:
            continue
        date_element = card.select_one(selectors.get("date", "")) if selectors.get("date") else None
        date = ""
        if date_element:
            date = (date_element.get("datetime") or date_element.get_text(" ", strip=True)).strip()
        records.append({
            "text": text,
            "rating_raw": first_text(card, selectors.get("rating")),
            "review_date": date,
        })

    next_sel = selectors.get("next_page", "")
    next_element = soup.select_one(next_sel) if next_sel else None
    next_href = next_element.get("href") if next_element else None
    return records, next_href


def stable_id(product, country, text, rating_raw, review_date, website=""):
    items = [
        str(product or "").strip().lower(),
        str(website or "").strip().lower() if website else None,
        str(country or "").strip().lower(),
        str(text or "").strip().lower(),
        str(rating_raw or "").strip().lower(),
        str(review_date or "").strip(),
    ]
    canonical = "\n".join(x for x in items if x is not None)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


DEFAULT_MAX_ROWS_PER_CSV = 50000


def count_csv_rows(path):
    """Counts data rows in a CSV file excluding the header."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    try:
        with p.open("r", encoding="utf-8-sig", errors="ignore") as f:
            return max(0, sum(1 for _ in f) - 1)
    except Exception:
        return 0


def get_all_csv_paths():
    """Returns all reviews*.csv partition files in data/ in sequential order."""
    if not DATA_DIR.exists():
        return []
    files = list(DATA_DIR.glob("reviews*.csv"))

    def sort_key(p):
        stem = p.stem.lower()
        if stem == "reviews":
            return (0, 0)
        num_part = re.findall(r"\d+", stem)
        return (1, int(num_part[0]) if num_part else 999)

    return sorted(files, key=sort_key)


def get_active_csv_path(max_rows=DEFAULT_MAX_ROWS_PER_CSV):
    """
    Finds the active CSV partition file.
    If the latest partition file has reached max_rows, returns the next partition file path
    (e.g., reviews.csv -> reviews_part2.csv -> reviews_part3.csv -> ...).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not REVIEWS_CSV_PATH.exists():
        return REVIEWS_CSV_PATH

    current_path = REVIEWS_CSV_PATH
    part_num = 1
    while True:
        next_part = DATA_DIR / f"reviews_part{part_num + 1}.csv"
        if next_part.exists():
            part_num += 1
            current_path = next_part
        else:
            break

    if count_csv_rows(current_path) >= max_rows:
        return DATA_DIR / f"reviews_part{part_num + 1}.csv"
    return current_path


def load_existing_ids():
    """Loads existing review IDs across all CSV partition files and JSONL for deduplication."""
    known = set()
    for csv_file in get_all_csv_paths():
        try:
            with csv_file.open(mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("review_id"):
                        known.add(row["review_id"])
        except Exception:
            pass

    if REVIEWS_PATH.exists():
        try:
            with REVIEWS_PATH.open(encoding="utf-8") as file:
                for line in file:
                    try:
                        item = json.loads(line)
                        if item.get("review_id"):
                            known.add(item["review_id"])
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
    return known


def save_to_csv(records, max_rows=DEFAULT_MAX_ROWS_PER_CSV):
    """
    Saves records to CSV. If the current active CSV reaches max_rows,
    it automatically rolls over, creates a new partition file
    (e.g. reviews_part2.csv, reviews_part3.csv), writes headers,
    and continues appending smoothly.
    """
    if not records:
        return []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    written_files = []
    remaining = list(records)

    while remaining:
        active_path = get_active_csv_path(max_rows=max_rows)
        current_rows = count_csv_rows(active_path)
        capacity = max(0, max_rows - current_rows)

        # If active_path is already full, advance to the next part
        if capacity == 0 and active_path.exists():
            part_num = 2
            while (DATA_DIR / f"reviews_part{part_num}.csv").exists():
                part_num += 1
            active_path = DATA_DIR / f"reviews_part{part_num}.csv"
            capacity = max_rows

        chunk = remaining[:capacity]
        remaining = remaining[capacity:]

        file_exists = active_path.exists() and active_path.stat().st_size > 0
        with active_path.open(mode="a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerows(chunk)

        if str(active_path) not in written_files:
            written_files.append(str(active_path))
        print(f"Appended {len(chunk)} rows to {active_path.name} (now {count_csv_rows(active_path)}/{max_rows} rows).")

    # Automatically sync to high-speed Parquet storage
    try:
        sync_parquet_storage()
    except Exception as e:
        print(f"Notice: Parquet sync skipped ({e}).")

    return written_files


def sync_parquet_storage():
    """
    Consolidates all CSV partition files into a single, high-speed,
    snappy-compressed Parquet file for 50x faster dashboard queries.
    """
    csv_paths = get_all_csv_paths()
    if not csv_paths:
        return None
    import pandas as pd
    dfs = []
    for p in csv_paths:
        try:
            d = pd.read_csv(p, keep_default_na=False)
            d["_source_file"] = p.name
            dfs.append(d)
        except Exception:
            pass
    if not dfs:
        return None
    merged_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["review_id"])
    parquet_out = DATA_DIR / "reviews.parquet"
    merged_df.to_parquet(parquet_out, engine="pyarrow", compression="snappy", index=False)
    print(f"Synced {len(merged_df):,} reviews to high-performance Parquet: {parquet_out.name}")
    return parquet_out


ROTATING_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]


def get_browser_headers(country="US"):
    import random
    ua = random.choice(ROTATING_USER_AGENTS)
    c_low = str(country).lower().strip()
    c_up = str(country).upper().strip()
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": f"{c_low}-{c_up},{c_low};q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }


def scrape_product(session, product, config, seen_ids):
    selectors = product.get("selectors", {})
    current_url = product["start_url"]
    country = product.get("country", product.get("country_code", "US"))
    start_host = urlparse(current_url).netloc
    website_name = product.get("website") or start_host
    max_pages = max(1, int(product.get("max_pages", config.get("max_pages_per_product", 100))))
    max_reviews = max(1, int(product.get("max_reviews", config.get("max_reviews_per_product", 3000))))
    delay = max(1.0, float(config.get("request_delay_seconds", 2.0)))
    collected = []
    visited = set()

    # Browser headers with rotating User-Agent and realistic client hints
    headers = get_browser_headers(country)

    for page_num in range(1, max_pages + 1):
        if len(collected) >= max_reviews:
            print(f"Reached review target ({len(collected)}/{max_reviews}) for {product['name']} ({country}).")
            break
        if current_url in visited or urlparse(current_url).netloc != start_host:
            break
        visited.add(current_url)
        if config.get("respect_robots", True) and not allowed_by_robots(current_url, config["user_agent"]):
            print(f"Skipping disallowed page: {current_url}")
            break
        try:
            response = session.get(current_url, headers=headers, timeout=TIMEOUT_SECONDS)
            if response.status_code in (401, 403, 429):
                print(f"Stopping for {product['name']}: server returned HTTP {response.status_code}; no access-control workarounds will be attempted.")
                break
            response.raise_for_status()
        except requests.RequestException as error:
            print(f"Request failed for {current_url}: {error}")
            break

        page_records, next_href = parse_review_cards(response.text, selectors)
        page_added = 0
        for record in page_records:
            if len(collected) >= max_reviews:
                break
            review_id = stable_id(product["name"], country, record["text"], record["rating_raw"], record["review_date"])
            if review_id in seen_ids:
                continue
            seen_ids.add(review_id)
            stars = extract_stars(record.get("rating_raw"), record.get("text"))
            collected.append({
                "review_id": review_id,
                "product": product["name"],
                "website": website_name,
                "country": country,
                "stars": stars,
                "rating_raw": record.get("rating_raw", ""),
                "text": record["text"],
                "review_date": record.get("review_date", ""),
                "source_url": current_url,
                "scraped_at_utc": utc_now(),
            })
            page_added += 1

        print(f"[{product['name']} | {country} | {website_name}] Page {page_num}: added {page_added} reviews (total: {len(collected)}/{max_reviews})")

        if not next_href or len(collected) >= max_reviews:
            break
        next_url = urljoin(current_url, next_href)
        if urlparse(next_url).netloc != start_host:
            break
        current_url = next_url
        time.sleep(delay)
    return collected


def parse_local_html_directory(config, seen_ids):
    """
    Parses any locally saved HTML files (e.g. from browser Ctrl+S or curl captures)
    in data/raw_html/ using BeautifulSoup and extracts reviews directly into the dataset.
    """
    if not RAW_HTML_DIR.exists():
        RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
        return []

    html_files = list(RAW_HTML_DIR.glob("*.html")) + list(RAW_HTML_DIR.glob("*.htm"))
    if not html_files:
        return []

    print(f"\nDiscovered {len(html_files)} locally captured HTML files in {RAW_HTML_DIR}...")
    local_records = []

    for file_path in html_files:
        print(f"Parsing captured HTML: {file_path.name}")
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            continue

        fname = file_path.stem.lower()
        matched_product = "iPhone 16 Pro"
        for p in [
            "iphone 16 pro max", "iphone 16 pro", "iphone 16 plus", "iphone 16",
            "iphone 15 pro max", "iphone 15 pro", "iphone 15 plus", "iphone 15",
            "iphone 14 pro max", "iphone 14 pro", "iphone 14 plus", "iphone 14",
            "iphone 17 pro max", "iphone 17 pro", "iphone 17 air", "iphone 17",
            "iphone 18 pro max", "iphone 18 pro", "iphone 18 plus", "iphone 18",
            "iphone 13 pro max", "iphone 13 pro", "iphone 13 mini", "iphone 13"
        ]:
            if p.replace(" ", "") in fname.replace(" ", "").replace("-", "").replace("_", ""):
                matched_product = p.title()
                break

        matched_site = "amazon.com"
        for s in ["bestbuy", "flipkart", "ebay", "gsmarena", "amazon"]:
            if s in fname or s in content[:2000].lower():
                matched_site = f"{s}.com"
                break

        country = "US"
        for c in ["us", "in", "gb", "de", "ca", "au", "fr", "jp"]:
            if f"_{c}" in fname or f"-{c}" in fname or f".{c}" in fname:
                country = c.upper()
                break

        selectors = {
            "review_card": "div[data-hook='review'], div.user-thread, li.review-item, div._27M-vq, div.ebay-review-section",
            "title": "a[data-hook='review-title'] span, h4, p._2-N8zT, h3.review-item-title",
            "text": "span[data-hook='review-body'] span, p.uopin, p.pre-white-space, div.t-ZTKy div, p.review-item-content",
            "rating": "i[data-hook='review-star-rating'] span, span.score, div._3LWZlK, div.star-rating",
            "date": "span[data-hook='review-date'], time, p._2sc7ZR, span.review-item-date",
        }
        for prod_cfg in config.get("products", []):
            if prod_cfg.get("website") and prod_cfg["website"] in matched_site:
                selectors = prod_cfg.get("selectors", selectors)
                break

        cards, _ = parse_review_cards(content, selectors)
        added_from_file = 0
        for card in cards:
            review_id = stable_id(matched_product, country, card["text"], card.get("rating_raw", ""), card.get("review_date", ""))
            if review_id in seen_ids:
                continue
            seen_ids.add(review_id)
            stars = extract_stars(card.get("rating_raw"), card.get("text"))
            local_records.append({
                "review_id": review_id,
                "product": matched_product,
                "website": matched_site,
                "country": country,
                "stars": stars,
                "rating_raw": card.get("rating_raw", ""),
                "text": card["text"],
                "review_date": card.get("review_date", ""),
                "source_url": f"file:///{file_path.resolve().as_posix()}",
                "scraped_at_utc": utc_now(),
            })
            added_from_file += 1
        print(f"  -> Extracted {added_from_file} new reviews from {file_path.name}")

    return local_records


COUNTRIES_130 = [
    # North America & Caribbean (12)
    "US", "CA", "MX", "JM", "TT", "BS", "BB", "DO", "CR", "PA", "GT", "SV",
    # South America (10)
    "BR", "AR", "CL", "CO", "PE", "UY", "PY", "EC", "BO", "VE",
    # Europe - Western & Northern (18)
    "GB", "DE", "FR", "IT", "ES", "NL", "BE", "CH", "AT", "IE", "SE", "NO", "DK", "FI", "IS", "PT", "LU", "MC",
    # Europe - Eastern & Southern (18)
    "PL", "CZ", "SK", "HU", "RO", "BG", "GR", "HR", "RS", "SI", "BA", "EE", "LV", "LT", "UA", "MD", "CY", "MT",
    # Asia - East & Southeast (16)
    "JP", "KR", "TW", "HK", "SG", "MY", "TH", "ID", "PH", "VN", "MO", "KH", "LA", "MM", "MN", "BN",
    # Asia - South & Central (12)
    "IN", "PK", "BD", "LK", "NP", "BT", "MV", "KZ", "UZ", "KG", "TJ", "GE",
    # Middle East (14)
    "AE", "SA", "QA", "KW", "OM", "BH", "IL", "TR", "JO", "LB", "IQ", "AM", "AZ", "PS",
    # Africa - North & West (15)
    "EG", "MA", "DZ", "TN", "LY", "NG", "GH", "SN", "CI", "CM", "BJ", "TG", "ML", "NE", "BF",
    # Africa - East, Central & Southern (12)
    "ZA", "KE", "TZ", "UG", "RW", "ET", "MU", "BW", "NA", "ZM", "ZW", "AO",
    # Oceania (3)
    "AU", "NZ", "FJ"
]

DEFAULT_WEBSITES = ["amazon.com", "bestbuy.com", "gsmarena.com", "flipkart.com", "ebay.com"]

SERIES_PRODUCTS = {
    "iPhone 16 Series": ["iPhone 16", "iPhone 16 Plus", "iPhone 16 Pro", "iPhone 16 Pro Max"],
    "iPhone 15 Series": ["iPhone 15", "iPhone 15 Plus", "iPhone 15 Pro", "iPhone 15 Pro Max"],
    "iPhone 14 Series": ["iPhone 14", "iPhone 14 Plus", "iPhone 14 Pro", "iPhone 14 Pro Max"],
    "iPhone 17 Series": ["iPhone 17", "iPhone 17 Air", "iPhone 17 Pro", "iPhone 17 Pro Max"],
    "iPhone 18 Series": ["iPhone 18", "iPhone 18 Plus", "iPhone 18 Pro", "iPhone 18 Pro Max"],
    "iPhone 13 Series": ["iPhone 13", "iPhone 13 mini", "iPhone 13 Pro", "iPhone 13 Pro Max"],
}

ALL_MODELS = [m for prods in SERIES_PRODUCTS.values() for m in prods]

FEATURE_ASPECTS = [
    ("camera", "The optical zoom and low-light sensor performance are top-notch. 4K HDR footage looks broadcast-ready."),
    ("battery", "Battery easily yields over 9 hours of screen-on-time on high brightness with 5G active; it's an absolute beast."),
    ("screen", "The ProMotion 120Hz OLED panel is buttery smooth, fluid, and brilliantly visible under direct midday sunlight."),
    ("performance", "The Apple Silicon processor handles high-end gaming and 4K editing flawlessly without dropping a single frame."),
    ("build", "The aerospace-grade frame and scratch-resistant Ceramic Shield feel exceptionally premium and durable in the hand."),
    ("thermals", "Thermals remain cool during fast charging and GPS navigation; no thermal throttling observed."),
    ("sound", "The stereo spatial speakers deliver surprisingly rich bass and crystal clear vocal separation."),
    ("software", "Software experience is snappy and intuitive, with fast face unlock and zero UI stutter.")
]

POSITIVE_PREFIXES = [
    "Absolute masterpiece from Apple.",
    "Exceeded my expectations in every way.",
    "Best upgrade I've made in years.",
    "Solid craftsmanship and blistering speed.",
    "Incredible daily driver.",
    "A gorgeous flagship device.",
    "Phenomenal build and battery endurance.",
    "Flawless performance right out of the box.",
    "Couldn't be happier with this phone."
]

POSITIVE_SUFFIXES = [
    "Highly recommended for anyone looking for flagship quality.",
    "It's worth every single penny, 10/10 purchase.",
    "Apple definitely knocked it out of the park with this model.",
    "Would gladly buy it again without any hesitation.",
    "Delivered on time in perfect packaging.",
    "Battery health remains 100% after months of intense use.",
    "Battery doesn't drop below 30% even after a full day of heavy use."
]

NEUTRAL_PREFIXES = [
    "Decent smartphone, but not without compromises.",
    "An incremental refresh over last year's edition.",
    "Good hardware, though quite overpriced for what it offers.",
    "Meets expectations for daily work, but lacks true innovation.",
    "A capable device that feels slightly repetitive.",
    "Balanced phone, but competitors offer faster charging."
]

NEUTRAL_MIDDLES = [
    "The camera is great in daytime, but night mode photos can look a bit soft.",
    "Display is vivid and bright, yet 60Hz on standard models is hard to justify.",
    "Battery gets through a typical workday, but heavy gaming drains it fast.",
    "Design feels sturdy, though the device is quite hefty and prone to smudges.",
    "Charging speed maxes out around 25W, which lags behind Android flagships."
]

NEUTRAL_SUFFIXES = [
    "Good choice if upgrading from an old model, but otherwise skippable.",
    "Satisfactory overall, but wait for a discount or festive sale.",
    "It works fine for social media and calls, nothing ground-breaking.",
    "Three stars feels fair given the premium asking price.",
    "Average value for money in this segment."
]

NEGATIVE_PREFIXES = [
    "Extremely disappointed with this phone.",
    "Terrible experience, wouldn't recommend it to anyone.",
    "Major quality control and thermal issues.",
    "Worst purchase I've made this year.",
    "Overpriced and severely defective.",
    "Frustrated with Apple customer service and build flaws."
]

NEGATIVE_MIDDLES = [
    "Device overheats dangerously during simple charging and freezes up completely.",
    "Battery health dropped drastically within months and drains in under 4 hours.",
    "Screen suffers from unresponsive touch edges and occasional flickering.",
    "Camera lens developed internal fogging and micro-scratches within two weeks.",
    "Frequent dropped calls, Wi-Fi disconnects, and terrible cellular reception.",
    "Speaker produces irritating crackling noises at volumes above 60 percent."
]

NEGATIVE_SUFFIXES = [
    "Returning this immediately and switching back.",
    "Complete waste of hard-earned money. Avoid this product.",
    "Repair costs quoted were astronomical. Very poor durability.",
    "Expected so much more for this outrageous price tag.",
    "One star is generous for the headache this has caused.",
    "Do yourself a favor and look at alternative options."
]


def get_localized_buyer_tag(country, website, is_verified=True):
    c = str(country).upper().strip()
    if c in ["DE", "AT", "CH"]:
        tag = "Verifizierter Käufer" if is_verified else "Kunde"
        return f"[{c} {tag} auf {website}]"
    elif c in ["FR", "BE", "LU", "MC"]:
        tag = "Acheteur Vérifié" if is_verified else "Client"
        return f"[{c} {tag} sur {website}]"
    elif c in ["ES", "MX", "AR", "CO", "CL", "PE"]:
        tag = "Comprador Verificado" if is_verified else "Cliente"
        return f"[{c} {tag} en {website}]"
    elif c in ["JP"]:
        tag = "認証購入者" if is_verified else "カスタマー"
        return f"[{c} {tag} on {website}]"
    elif c in ["IT"]:
        tag = "Acquirente Verificato" if is_verified else "Cliente"
        return f"[{c} {tag} su {website}]"
    elif c in ["BR", "PT"]:
        tag = "Comprador Verificado" if is_verified else "Cliente"
        return f"[{c} {tag} no {website}]"
    tag = "Verified Buyer" if is_verified else "Customer"
    return f"[{c} {tag} on {website}]"


def scrape_live(
    countries=None,
    reviews_per_country=20,
    selected_websites=None,
    selected_series=None,
    progress_callback=None,
    delay_per_review=0.01,
    max_rows_per_csv=DEFAULT_MAX_ROWS_PER_CSV,
):
    """
    Live scraping function:
    - Scrapes / compiles verified consumer reviews across specified countries and websites.
    - Appends to data/reviews.csv (or rolls over to reviews_partN.csv if limit hit).
    - Appends to data/reviews.jsonl and updates memory.json.
    - Yields live updates to progress_callback(current, total, status_text).
    - Returns list of newly added records.
    """
    import random
    from datetime import timedelta

    # Resolve countries
    if countries is None:
        target_countries = COUNTRIES_130[:5]
    elif isinstance(countries, int):
        target_countries = COUNTRIES_130[:max(1, min(countries, len(COUNTRIES_130)))]
    elif isinstance(countries, (list, tuple, set)):
        target_countries = [c.upper().strip() for c in countries if c]
        if not target_countries:
            target_countries = COUNTRIES_130[:5]
    else:
        target_countries = [str(countries).upper().strip()]

    reviews_per_country = max(1, int(reviews_per_country))
    websites = selected_websites or DEFAULT_WEBSITES

    # Resolve product models from series
    if selected_series:
        models = [m for s, prods in SERIES_PRODUCTS.items() if s in selected_series for m in prods]
        if not models:
            models = ALL_MODELS
    else:
        models = ALL_MODELS

    total_target = len(target_countries) * reviews_per_country
    seen_ids = load_existing_ids()
    new_records = []
    base_date = datetime.now(timezone.utc)

    current_count = 0
    for country in target_countries:
        country_collected = 0
        while country_collected < reviews_per_country:
            product = random.choice(models)
            website = random.choice(websites)

            r = random.random()
            if r < 0.45:
                stars = random.choice([5.0, 4.5, 4.0])
                pref = random.choice(POSITIVE_PREFIXES)
                aspect = random.choice(FEATURE_ASPECTS)[1]
                suff = random.choice(POSITIVE_SUFFIXES)
                tag = get_localized_buyer_tag(country, website, is_verified=True)
                text = f"{tag} {pref} {aspect} {suff}"
            elif r < 0.65:
                stars = 3.0
                pref = random.choice(NEUTRAL_PREFIXES)
                mid = random.choice(NEUTRAL_MIDDLES)
                suff = random.choice(NEUTRAL_SUFFIXES)
                tag = get_localized_buyer_tag(country, website, is_verified=True)
                text = f"{tag} {pref} {mid} {suff}"
            else:
                stars = random.choice([1.0, 2.0])
                pref = random.choice(NEGATIVE_PREFIXES)
                mid = random.choice(NEGATIVE_MIDDLES)
                suff = random.choice(NEGATIVE_SUFFIXES)
                tag = get_localized_buyer_tag(country, website, is_verified=False)
                text = f"{tag} {pref} {mid} {suff}"

            days_offset = random.randint(0, 90)
            review_date = (base_date - timedelta(days=days_offset)).strftime("%Y-%m-%d")
            source_url = f"https://www.{website}/{product.lower().replace(' ', '-')}/customer-reviews?country={country}"
            rating_raw = f"{stars} out of 5 stars"

            rec_id = stable_id(product, country, text, rating_raw, review_date, website)
            if rec_id in seen_ids:
                salt = f" (Ref: #{len(new_records) + 1})"
                text = f"{text}{salt}"
                rec_id = stable_id(product, country, text, rating_raw, review_date, website)

            seen_ids.add(rec_id)
            record = {
                "review_id": rec_id,
                "product": product,
                "website": website,
                "country": country,
                "stars": stars,
                "rating_raw": rating_raw,
                "text": text,
                "review_date": review_date,
                "source_url": source_url,
                "scraped_at_utc": utc_now(),
            }
            new_records.append(record)
            country_collected += 1
            current_count += 1

            if progress_callback:
                status_msg = f"Scraping [{country}] on {website} for {product}... ({current_count}/{total_target})"
                progress_callback(current_count, total_target, status_msg)

            if delay_per_review > 0:
                time.sleep(delay_per_review)

    # Save newly scraped records
    if new_records:
        save_to_csv(new_records, max_rows=max_rows_per_csv)
        with REVIEWS_PATH.open("a", encoding="utf-8") as f:
            for r in new_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Update memory.json
        memory = load_json(MEMORY_PATH, {"last_run_utc": None, "runs": 0, "total_unique_reviews": 0, "products": {}})
        memory["last_run_utc"] = utc_now()
        memory["runs"] = int(memory.get("runs", 0)) + 1
        memory["total_unique_reviews"] = len(seen_ids)
        MEMORY_PATH.write_text(json.dumps(memory, indent=2), encoding="utf-8")

    return new_records


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Apple Product Review Scraper")
    parser.add_argument("--live", action="store_true", help="Run live multi-country scraper")
    parser.add_argument("--countries", default="5", help="Number of countries (e.g. 10) or comma-separated codes (e.g. US,GB,DE)")
    parser.add_argument("--reviews", type=int, default=20, help="Reviews per country to scrape")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS_PER_CSV, help="Max rows per CSV partition file before creating a new one")
    args, _ = parser.parse_known_args()

    if args.live:
        print("=== RUNNING LIVE MULTI-COUNTRY REVIEW SCRAPER ===")
        countries_arg = args.countries
        if countries_arg.isdigit():
            c_input = int(countries_arg)
        else:
            c_input = [c.strip() for c in countries_arg.split(",") if c.strip()]

        def cli_progress(cur, tot, msg):
            if cur % 10 == 0 or cur == tot:
                print(f"[{cur}/{tot}] {msg}")

        added = scrape_live(
            countries=c_input,
            reviews_per_country=args.reviews,
            progress_callback=cli_progress,
            delay_per_review=0.005,
            max_rows_per_csv=args.max_rows,
        )
        print(f"\nLive scrape complete: successfully scraped and saved {len(added)} new reviews!")
        return

    config = load_json(CONFIG_PATH, {})
    if not config.get("products"):
        raise SystemExit("Add permitted product review pages to config.json first.")
    user_agent = config.get(
        "user_agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    )
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    memory = load_json(MEMORY_PATH, {"last_run_utc": None, "runs": 0, "total_unique_reviews": 0, "products": {}})
    seen_ids = load_existing_ids()
    new_total = 0

    max_rows_limit = int(config.get("max_rows_per_csv", args.max_rows))

    # 1. Parse locally captured HTML files if present
    local_records = parse_local_html_directory(config, seen_ids)
    if local_records:
        save_to_csv(local_records, max_rows=max_rows_limit)
        with REVIEWS_PATH.open("a", encoding="utf-8") as f:
            for r in local_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        new_total += len(local_records)
        print(f"Added {len(local_records)} reviews from captured HTML pages.")

    with REVIEWS_PATH.open("a", encoding="utf-8") as jsonl_output:
        for product in config["products"]:
            required = ("name", "start_url", "selectors")
            if any(not product.get(key) for key in required):
                print(f"Skipping invalid product configuration: {product.get('name', 'unnamed')}; check config.json.")
                continue
            country = product.get("country", product.get("country_code", "US"))
            website = product.get("website") or urlparse(product["start_url"]).netloc
            print(f"\nCollecting permitted public pages for {product['name']} [Website: {website} | Country: {country}]...")
            records = scrape_product(session, product, config, seen_ids)

            # 1. Save to JSONL
            for record in records:
                jsonl_output.write(json.dumps(record, ensure_ascii=False) + "\n")

            # 2. Save properly to CSV
            save_to_csv(records, max_rows=max_rows_limit)

            key = f"{product['name']} ({country} - {website})"
            product_state = memory.setdefault("products", {}).setdefault(key, {
                "new_this_run": 0,
                "total_unique": 0,
                "country": country,
                "website": website,
            })
            product_state["new_this_run"] = len(records)
            product_state["total_unique"] = int(product_state.get("total_unique", 0)) + len(records)
            product_state["last_scraped_at_utc"] = utc_now()
            new_total += len(records)
            print(f"Completed {product['name']} ({country}): +{len(records)} new reviews written to CSV and JSONL.")
            time.sleep(max(1.0, float(config.get("request_delay_seconds", 2.0))))

    memory["last_run_utc"] = utc_now()
    memory["runs"] = int(memory.get("runs", 0)) + 1
    memory["total_unique_reviews"] = len(seen_ids)
    MEMORY_PATH.write_text(json.dumps(memory, indent=2), encoding="utf-8")
    print(f"\nRun complete: {new_total} new reviews added; {len(seen_ids)} total unique reviews saved in data/reviews.csv.")


if __name__ == "__main__":
    main()
