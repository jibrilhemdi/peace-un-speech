import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import pandas as pd
import time
import os
import sys
import urllib.request
import logging
import re
import json
import argparse
import random
import urllib.parse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://digitallibrary.un.org/search?p=creation_date%3A2023-10-07-%3E2025-12-31&cc=Speeches&c=Speeches&ln=en&as_query=JTdCJTIyZGF0ZV9zZWxlY3RvciUyMiUzQSU3QiUyMmRhdGVUeXBlJTIyJTNBJTIyY3JlYXRpb25fZGF0ZSUyMiUyQyUyMmRhdGVQZXJpb2QlMjIlM0ElMjJzcGVjaWZpY2RhdGVwZXJpb2QlMjIlMkMlMjJkYXRlRnJvbSUyMiUzQSUyMjIwMjMtMTAtMDclMjIlMkMlMjJkYXRlVG8lMjIlM0ElMjIyMDI1LTEyLTMxJTIyJTdEJTJDJTIyY2xhdXNlcyUyMiUzQSU1QiU3QiUyMnNlYXJjaEluJTIyJTNBJTIyYWxsLWZpZWxkJTIyJTJDJTIyY29udGFpbiUyMiUzQSUyMmFsbC13b3JkcyUyMiUyQyUyMnRlcm0lMjIlM0ElMjIlMjIlMkMlMjJvcGVyYXRvciUyMiUzQSUyMkFORCUyMiU3RCU1RCU3RA%3D%3D&fct__8=PALESTINE+QUESTION&fct__3=2025&fct__3=2024&fct__3=2023&jrec=1&rg=100&as=1"
# Anchored on the repo root, not the working directory — the collector runs for hours and
# must resume from the same checkpoint whichever directory it was launched from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
OUTPUT_FILE = os.path.join(RAW_DIR, "un_speeches_palestine.csv")
PDF_DIR = os.path.join(RAW_DIR, "pdfs")
PER_PAGE = 100
CHECKPOINT_FILE = os.path.join(RAW_DIR, "collection_checkpoint.json")
os.makedirs(PDF_DIR, exist_ok=True)

driver = None

def setup_driver():
    global driver
    if driver:
        try:
            driver.quit()
        except Exception:
            pass

    try:
        uc.install()
    except Exception:
        pass

    options = uc.ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--lang=en-US')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    options.add_argument('--referer=https://www.un.org/en/library')

    driver = uc.Chrome(options=options, version_main=149)
    time.sleep(2)
    return driver

def check_driver_alive():
    """Check if driver is still alive and restart if needed."""
    global driver
    try:
        driver.current_url
        return True
    except Exception as e:
        logger.warning(f"Driver check failed: {e}")
        logger.warning("Driver crashed, restarting...")
        try:
            driver = setup_driver()
            return True
        except Exception as e2:
            logger.error(f"Failed to restart driver: {e2}")
            return False

def get_reported_total(driver):
    """Extract reported total record count from pagination text like '2,070 records found'.
    Returns (count, source_url) or (None, None) if not found.
    Used only as a SANITY CHECK against what we actually collect - we do NOT
    rely on it to decide how many pages to scrape."""
    try:
        candidates = driver.find_elements(By.CSS_SELECTOR, ".pagination, .pager, .search-summary, .results-info")
        for el in candidates:
            text = el.text
            match = re.search(r'([\d,]+)\s+records?\s+found', text, re.IGNORECASE)
            if match:
                count = int(match.group(1).replace(',', ''))
                logger.info(f"Site reports total records: {count}")
                return count, driver.current_url
    except Exception as e:
        logger.warning(f"Could not extract reported total: {e}")
    return None, None

def extract_speech_data(driver, seen_ids):
    speeches = []
    try:
        links = driver.find_elements(By.CSS_SELECTOR, "div.result-title a[href*='record/']")

        for link in links:
            try:
                href = link.get_attribute("href")
                title = link.text.strip()
                record_id = href.split('/record/')[-1].split('?')[0]

                if href and len(title) > 5 and record_id not in seen_ids:
                    seen_ids.add(record_id)
                    speeches.append({
                        "title": title,
                        "url": href,
                        "record_id": record_id,
                    })
            except Exception:
                continue

    except Exception as e:
        logger.error(f"Error extracting speech data: {e}")

    return speeches

def parse_jrec(url):
    m = re.search(r'jrec=(\d+)', url or '')
    return int(m.group(1)) if m else 1

def get_next_page_url(driver):
    """Return the URL of the NEXT page link only. The site pagination contains
    multiple jrec= links (previous/numbered); we must match the explicit 'Next'
    labelled link so we never accidentally go backwards."""
    try:
        # Prefer an explicit Next link (text contains Next / » / >)
        next_candidates = driver.find_elements(By.XPATH,
            "//a[contains(translate(@href,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'jrec=') "
            "and (contains(translate(.,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'next') "
            "or .='»' or .='>' or contains(@rel,'next'))]")
        for link in next_candidates:
            href = link.get_attribute("href")
            if href and "jrec=" in href:
                return href

        # Fallback: among all jrec= links, pick the one whose offset is strictly
        # greater than the current page's offset (the smallest such = next page).
        current = parse_jrec(driver.current_url)
        best = None
        best_off = None
        all_jrec = driver.find_elements(By.XPATH, "//a[contains(@href,'jrec=')]")
        for link in all_jrec:
            href = link.get_attribute("href")
            if href and "jrec=" in href:
                off = parse_jrec(href)
                if off > current and (best_off is None or off < best_off):
                    best = href
                    best_off = off
        return best
    except Exception as e:
        logger.warning(f"Could not find next page link: {e}")
    return None

def save_checkpoint(seen_ids, total_seen, reported_total):
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({
                "total_seen": total_seen,
                "reported_total": reported_total,
                "seen_ids": sorted(seen_ids),
            }, f)
    except Exception as e:
        logger.warning(f"Could not save checkpoint: {e}")

def load_existing():
    """Load already-collected record_ids from an existing CSV so a re-run resumes
    without duplication and without re-scraping."""
    seen = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE)
            if "record_id" in df.columns:
                seen = set(df["record_id"].astype(str).tolist())
                logger.info(f"Loaded {len(seen)} existing record_ids from {OUTPUT_FILE}")
        except Exception as e:
            logger.warning(f"Could not read existing CSV: {e}")
    return seen

def is_blocked(driver):
    """Detect anti-bot / error pages that return an empty result set.
    Returns True if blocked (403/challenge), "empty" if the results container is
    missing (retryable), or False if the page genuinely has no results."""
    try:
        src = driver.page_source or ""
        title = driver.title or ""
        if "403" in title or "Forbidden" in src[:500]:
            return True
        if "access denied" in src[:2000].lower() or "are you a robot" in src[:2000].lower():
            return True
        # A real results page always contains result-title links. Absence means
        # the page didn't load results. We do NOT rely on the URL because the
        # driver can be redirected away from the search URL.
        if not driver.find_elements(By.CSS_SELECTOR, "div.result-title a[href*='record/']"):
            no_results = re.search(r'no\s+records?\s+found|0\s+records?\s+found', src[:4000], re.IGNORECASE)
            if no_results:
                return False  # genuinely empty results page
            return "empty"
        return False
    except Exception:
        return True

def polite_wait(base=2.5, jitter=2.0):
    time.sleep(base + random.uniform(0, jitter))

def navigate_pages(driver):
    """Self-terminating pagination: keep following the Next link until none exists.
    Resilient to anti-bot blocks: empty/blocked pages are retried with a fresh
    driver and backoff rather than treated as end-of-results."""
    all_speeches = []
    seen_ids = load_existing()

    # Reported total is used ONLY as a post-hoc sanity check, not for stopping.
    reported_total, _ = get_reported_total(driver)

    page = 1
    retries = 0
    MAX_RETRIES = 5
    last_offset = 0

    while True:
        logger.info(f"Scraping page {page}...")

        if not check_driver_alive():
            logger.error("Driver not responding, cannot continue")
            break

        polite_wait()

        blocked = is_blocked(driver)
        if blocked:
            retries += 1
            logger.warning(f"Page {page} looks blocked ({blocked}). "
                           f"Retry {retries}/{MAX_RETRIES} with fresh driver + backoff.")
            if retries > MAX_RETRIES:
                logger.error("Too many consecutive blocks - stopping to avoid a ban. Re-run later.")
                break
            try:
                driver = setup_driver()
            except Exception:
                pass
            time.sleep(10 * retries)  # backoff
            # Re-fetch the current offset page rather than advancing.
            cur_jrec = parse_jrec(driver.current_url)
            if cur_jrec > 1:
                base = driver.current_url.split("jrec=")[0]
                driver.get(f"{base}jrec={cur_jrec}")
            continue

        speeches = extract_speech_data(driver, seen_ids)
        logger.info(f"Page {page}: {len(speeches)} new speeches found, {len(seen_ids)} total unique")

        if speeches:
            retries = 0
            all_speeches.extend(speeches)
            # Incremental save so a crash loses minimal work.
            df_new = pd.DataFrame(all_speeches)
            if os.path.exists(OUTPUT_FILE):
                try:
                    df_old = pd.read_csv(OUTPUT_FILE)
                    df_new = pd.concat([df_old, df_new], ignore_index=True)
                    df_new = df_new.drop_duplicates(subset=["record_id"], keep="first")
                except Exception:
                    pass
            df_new.to_csv(OUTPUT_FILE, index=False)
            save_checkpoint(seen_ids, len(seen_ids), reported_total)

        # Decide whether to continue. We only treat an empty page as the genuine
        # end if there is NO next-page link. Otherwise it's a (missed) page we
        # must retry rather than stop early.
        next_url = get_next_page_url(driver)

        if not speeches:
            if next_url:
                retries += 1
                logger.warning(f"Page {page}: 0 new speeches but a next link exists "
                               f"(offset {parse_jrec(next_url)}). Retry {retries}/{MAX_RETRIES}.")
                if retries > MAX_RETRIES:
                    logger.error("Too many consecutive empty pages with a next link - stopping.")
                    break
                try:
                    driver = setup_driver()
                except Exception:
                    pass
                time.sleep(10 * retries)
                driver.get(next_url)  # jump to the next offset and retry from there
                page += 1
                continue
            else:
                logger.info(f"Page {page}: no new speeches and no next link (genuine end-of-results).")
                break

        if not next_url:
            logger.info("No next-page link found - reached the last page.")
            break

        # Sanity: ensure the next offset actually advances.
        nxt_off = parse_jrec(next_url)
        if nxt_off <= last_offset:
            logger.warning(f"Next offset {nxt_off} does not advance from {last_offset}; stopping.")
            break
        last_offset = nxt_off

        logger.info(f"Navigating to next page: {next_url}")

        if not check_driver_alive():
            break

        try:
            driver.get(next_url)
        except Exception as e:
            logger.error(f"Navigation failed: {e}")
            driver = setup_driver()
            driver.get(next_url)

        page += 1

    # Re-read reported total from the last reachable page for a better check.
    try:
        rt2, _ = get_reported_total(driver)
        if rt2:
            reported_total = rt2
    except Exception:
        pass

    # ---- Completeness reconciliation (advisory) ----
    if reported_total is not None:
        if len(seen_ids) >= reported_total:
            logger.info(f"COMPLETENESS OK: collected {len(seen_ids)} >= reported {reported_total}")
        else:
            logger.warning(
                f"COMPLETENESS GAP: collected {len(seen_ids)} but site reported {reported_total}. "
                f"Re-run to retry; the scraper now auto-retries blocks with backoff."
            )
    else:
        logger.warning("Could not read site's reported total - completeness not auto-verified. "
                       "Review the last page manually.")

    return all_speeches

def ensure_driver(driver, retries=2):
    """Return a live driver, restarting it if it has crashed. Returns (driver, ok)."""
    for _ in range(retries):
        if check_driver_alive():
            return driver, True
        try:
            driver = setup_driver()
            return driver, True
        except Exception as e:
            logger.error(f"Driver restart failed: {e}")
    return driver, False

def format_meeting_text(text):
    """Insert a single space before an opening parenthesis so e.g.
    'S/PV.9451(Resumption1)' becomes 'S/PV.9451 (Resumption1)' (idempotent)."""
    if not text:
        return text
    return re.sub(r'(\S)\(', r'\1 (', text.strip())

def get_meeting_record_url(driver, speech_url):
    """Return (driver, kind, value).
    kind == 'url'   -> value is a direct meeting-record page URL.
    kind == 'query' -> value is formatted meeting-record text to search for.
    kind is None    -> not found / error.
    When the meeting record has no link we fall back to its text so the caller
    can search the library for the top result."""
    last_err = None
    for attempt in range(2):
        driver, ok = ensure_driver(driver)
        if not ok:
            break
        try:
            driver.get(speech_url)
            time.sleep(1)

            # Preferred: a direct link to the meeting record.
            a_elems = driver.find_elements(By.XPATH, "//span[text()='Meeting record']/following-sibling::span/a")
            if a_elems:
                href = a_elems[0].get_attribute("href")
                if href:
                    return driver, "url", href

            # Fallback: read the meeting record text (e.g. 'S/PV.9451(Resumption1)')
            span_elems = driver.find_elements(By.XPATH, "//span[text()='Meeting record']/following-sibling::span")
            if span_elems:
                text = span_elems[0].text.strip()
                if text:
                    return driver, "query", format_meeting_text(text)

            # Page loaded but no meeting-record field at all.
            return driver, None, None

        except Exception as e:
            last_err = e
            logger.warning(f"Meeting record attempt {attempt + 1} failed for {speech_url}: {e}")

    if last_err:
        logger.error(f"Error extracting meeting record from {speech_url}: {last_err}")
    return driver, None, None

def search_meeting_record(driver, query):
    """Search the UN Digital Library for the meeting record text and return the
    top result's record URL (which we can then extract a PDF from)."""
    last_err = None
    for attempt in range(2):
        driver, ok = ensure_driver(driver)
        if not ok:
            break
        try:
            url = "https://digitallibrary.un.org/search?p=" + urllib.parse.quote(query) + "&ln=en&rg=10"
            driver.get(url)
            time.sleep(2)

            links = driver.find_elements(By.CSS_SELECTOR, "div.result-title a[href*='record/']")
            if links:
                href = links[0].get_attribute("href")
                if href:
                    logger.info(f"Search top result for '{query}': {href}")
                    return driver, href

            logger.warning(f"Search for '{query}' returned no results.")
            return driver, None

        except Exception as e:
            last_err = e
            logger.warning(f"Meeting record search attempt {attempt + 1} failed for '{query}': {e}")

    if last_err:
        logger.error(f"Error searching meeting record for '{query}': {last_err}")
    return driver, None

def get_pdf_url_from_meeting(driver, meeting_url):
    last_err = None
    for attempt in range(2):
        driver, ok = ensure_driver(driver)
        if not ok:
            break
        try:
            driver.get(meeting_url)
            time.sleep(1)

            meta_elements = driver.find_elements(By.XPATH, "//meta[@name='citation_pdf_url' and contains(@content, '-EN.pdf')]")
            if meta_elements:
                return driver, meta_elements[0].get_attribute("content")

            # Some record pages expose the PDF as a direct attachment link.
            pdf_links = driver.find_elements(By.XPATH, "//a[contains(@href, '-EN.pdf')]")
            if pdf_links:
                return driver, pdf_links[0].get_attribute("href")

            return driver, None

        except Exception as e:
            last_err = e
            logger.warning(f"PDF URL attempt {attempt + 1} failed for {meeting_url}: {e}")

    if last_err:
        logger.error(f"Error extracting PDF URL from {meeting_url}: {last_err}")
    return driver, None

def download_pdf(pdf_url, record_id, pdf_dir):
    try:
        os.makedirs(pdf_dir, exist_ok=True)
        filename = f"{record_id}.pdf"
        filepath = os.path.join(pdf_dir, filename)

        if os.path.exists(filepath):
            logger.info(f"PDF already exists: {filepath}")
            return filepath

        urllib.request.urlretrieve(pdf_url, filepath)
        logger.info(f"Downloaded PDF: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error downloading PDF {pdf_url}: {e}")
    return None

def process_speech_pdfs(driver, df):
    """Download stage. Dedup is enforced: skip ids already downloaded and any
    duplicate record_id in the CSV. Runs only AFTER the full CSV is collected."""
    existing_pdfs = set()
    if os.path.exists(PDF_DIR):
        for f in os.listdir(PDF_DIR):
            if f.endswith(".pdf"):
                existing_pdfs.add(f[:-4])

    seen_record_ids = set()
    pdf_urls = {}
    pdf_files = {}
    meeting_urls = {}
    meeting_texts = {}

    for idx in range(len(df)):
        row = df.iloc[idx]
        speech_url = row['url']
        speech_id = str(row['record_id'])

        # Dedup within CSV (duplicate record_id)
        if speech_id in seen_record_ids:
            logger.info(f"Skipping duplicate record_id {speech_id} at idx {idx}")
            pdf_urls[idx] = None
            pdf_files[idx] = None
            meeting_urls[idx] = None
            meeting_texts[idx] = None
            continue

        # Dedup by already-downloaded file
        if speech_id in existing_pdfs:
            logger.info(f"PDF already downloaded for {speech_id}, skipping download")
            pdf_urls[idx] = None
            pdf_files[idx] = os.path.join(PDF_DIR, f"{speech_id}.pdf")
            meeting_urls[idx] = None
            meeting_texts[idx] = None
            seen_record_ids.add(speech_id)
            continue

        logger.info(f"Processing speech {idx + 1}/{len(df)}: {speech_id}")

        try:
            driver, kind, value = get_meeting_record_url(driver, speech_url)

            if kind == "url":
                meeting_urls[idx] = value
                driver, pdf_url = get_pdf_url_from_meeting(driver, value)
                pdf_urls[idx] = pdf_url

            elif kind == "query":
                # No direct link: search the library for the top result.
                meeting_texts[idx] = value
                logger.info(f"No meeting-record link; searching library for '{value}'")
                driver, search_url = search_meeting_record(driver, value)
                meeting_urls[idx] = search_url
                if search_url:
                    driver, pdf_url = get_pdf_url_from_meeting(driver, search_url)
                    pdf_urls[idx] = pdf_url
                else:
                    pdf_urls[idx] = None

            else:
                meeting_urls[idx] = None
                pdf_urls[idx] = None

            if pdf_urls.get(idx):
                pdf_file = download_pdf(pdf_urls[idx], speech_id, PDF_DIR)
                pdf_files[idx] = pdf_file
            else:
                pdf_files[idx] = None

        except Exception as e:
            logger.error(f"Error processing speech {speech_id}: {e}")
            pdf_urls[idx] = None
            pdf_files[idx] = None

        seen_record_ids.add(speech_id)

    return pdf_urls, pdf_files, meeting_urls, meeting_texts

def main():
    parser = argparse.ArgumentParser(description="Scrape UN speeches list and optionally download PDFs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Collect the speech list + verify pagination/completeness only. "
                             "No PDF URLs are fetched and nothing is downloaded.")
    parser.add_argument("--csv-only", action="store_true",
                        help="Build/update the CSV list (STAGE 1) but skip PDF downloads.")
    parser.add_argument("--resume-pdfs", action="store_true",
                        help="Skip STAGE 1 (assume CSV exists) and only run the PDF download stage.")
    args = parser.parse_args()

    if args.dry_run:
        logger.info("=== DRY-RUN MODE: list collection + completeness check, NO downloads ===")
    elif args.resume_pdfs:
        logger.info("=== RESUME-PDFS MODE: skipping list collection, running PDF stage only ===")
    else:
        logger.info("Starting UN Speeches collection for Palestine question (Oct 7, 2023 - Dec 31, 2025)")

    d = setup_driver()
    try:
        # ---- STAGE 1: collect the speech list (CSV) ----
        if not args.resume_pdfs:
            logger.info("Using pre-filtered BASE_URL (subject + date filters already applied)")
            d.get(BASE_URL)
            time.sleep(2)
            logger.info(f"Page title: {d.title}")
            logger.info(f"Current URL: {d.current_url}")

            all_speeches = navigate_pages(d)
            logger.info(f"STAGE 1 done: {len(all_speeches)} new speeches collected this run.")

            if not os.path.exists(OUTPUT_FILE):
                logger.warning("No CSV produced; nothing to download.")
                return

            df = pd.read_csv(OUTPUT_FILE)
            logger.info(f"CSV contains {len(df)} unique speeches.")
        else:
            if not os.path.exists(OUTPUT_FILE):
                logger.error(f"{OUTPUT_FILE} not found - cannot resume PDFs without the list.")
                return
            df = pd.read_csv(OUTPUT_FILE)
            logger.info(f"Loaded {len(df)} speeches from existing {OUTPUT_FILE}.")

        # ---- DRY-RUN: stop before any PDF work ----
        if args.dry_run or args.csv_only:
            mode = "DRY-RUN" if args.dry_run else "CSV-ONLY"
            logger.info(f"=== {mode}: CSV list is complete. Skipping PDF extraction/download. ===")
            logger.info(f"=== Collected {len(df)} unique speeches. Review the CSV, then run without --dry-run to download. ===")
            return

        # ---- STAGE 2: download PDFs (resumable + deduped) ----
        logger.info("STAGE 2: extracting PDF URLs and downloading (resumable)...")
        pdf_urls, pdf_files, meeting_urls, meeting_texts = process_speech_pdfs(d, df)

        df['meeting_record_url'] = [meeting_urls.get(i) for i in range(len(df))]
        df['meeting_record_text'] = [meeting_texts.get(i) for i in range(len(df))]
        df['pdf_url'] = [pdf_urls.get(i) for i in range(len(df))]
        df['pdf_file'] = [pdf_files.get(i) for i in range(len(df))]
        df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Added PDF information to {OUTPUT_FILE}")

        downloaded = sum(1 for v in pdf_files.values() if v)
        missing = len(df) - downloaded
        logger.info(f"STAGE 2 done: {downloaded} PDFs present/downloaded, {missing} missing for {len(df)} speeches.")

    except Exception as e:
        logger.error(f"Error during scraping: {e}")
    finally:
        try:
            d.quit()
        except Exception:
            pass
        logger.info("Done")

if __name__ == "__main__":
    main()
