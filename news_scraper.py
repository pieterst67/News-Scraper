#!/usr/bin/env python3

import os
import importlib
import sqlite3
import smtplib
import ssl
import json
import html
import time
import random
import logging
import locale
import datetime as dt
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime
from collections import defaultdict
from urllib.parse import urlparse

import feedparser
import numpy as np
import hdbscan
from sklearn.preprocessing import normalize
from newspaper import Article
from openai import OpenAI
from dotenv import load_dotenv
from embeddings_utils import embed_article

# --- INITIALIZATION ---
load_dotenv()  # Optional .env support
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
locale.setlocale(locale.LC_TIME, "nl_NL.UTF-8")

# --- CONFIGURATION ---
DB_PATH = os.environ.get("DB_PATH", "news.db") # Use env var or default to local file

# Load FEEDS from environment variables
FEEDS = [u for u in os.getenv("FEEDS", "").splitlines() if u]

# Mmax words per digest (~200 wpm × 15 min = 3000)
READ_LIMIT_WORDS = 3000

# This is the new key parameter for the centroid pipeline.
# Only articles with a cosine similarity > this value to a topic's center will be included.
SIMILARITY_THRESHOLD = 0.85

# Threshold to consider a new cluster an update to an old one
CONTINUING_STORY_THRESHOLD = 0.80

# Number of days to keep old articles and briefings in the database.
DB_RETENTION_DAYS = 7

# Build paywall map {registered_domain : scraper-callable}
paywall_map = {}
for item in os.getenv("PAYWALL_MAP", "").split(","):
    item = item.strip()
    if not item:
        continue
    domain, target = item.split("=")
    module, func   = target.split(":")
    paywall_map[domain] = getattr(importlib.import_module(module), func)

# --- HELPER FUNCTIONS (from your script) ---
def fetch_article(url: str) -> str:
    """Downloads and parses an article, returning its text."""
    try:
        art = Article(url)
        domain = urlparse(url).netloc
        # pay-wall scraper present?
        if domain in paywall_map:
            art.set_html(paywall_map[domain](url))
        else:
            art.download()
        art.parse()
        return art.text or art.title
    except Exception as e:
        logging.warning(f"Could not download or parse article at {url}: {e}")
        return ""

def _as_datetime(pub: str) -> dt.datetime:
    """Converts a string to a timezone-aware datetime object."""
    try:
        d = dt.datetime.fromisoformat(pub)
    except (ValueError, TypeError):
        d = parsedate_to_datetime(pub)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)

# --- CORE FUNCTIONS ---
def init_db():
    """Initializes the database and adds new columns if they don't exist."""
    with sqlite3.connect(DB_PATH) as con:
        # Create articles table if it doesn't exist
        con.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT,
                url TEXT UNIQUE,
                title TEXT,
                full_text TEXT,
                published_date TIMESTAMP,
                embedding BLOB,
                processed_for_digest BOOLEAN DEFAULT 0
            )
        """)
        # Create briefings table if it doesn't exist
        con.execute("""
            CREATE TABLE IF NOT EXISTS briefings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_title TEXT,
                summary_content TEXT,
                source_articles TEXT,
                importance_score INTEGER DEFAULT 0,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.commit()
    logging.info(f"Database {DB_PATH} initialized.")

def cleanup_database():
    """Removes old records from the database to prevent it from growing indefinitely."""
    logging.info(f"Cleaning up database records older than {DB_RETENTION_DAYS} days...")
    cutoff_date = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=DB_RETENTION_DAYS)

    with sqlite3.connect(DB_PATH) as con:
        cursor = con.cursor()

        # Clean up old articles
        cursor.execute("DELETE FROM articles WHERE published_date < ?", (cutoff_date.isoformat(),))
        articles_deleted = cursor.rowcount

        # Clean up old briefings
        cursor.execute("DELETE FROM briefings WHERE created_at < ? AND sent_in_digest = 1", (cutoff_date.isoformat(),))
        briefings_deleted = cursor.rowcount

        con.commit()

        if articles_deleted > 0 or briefings_deleted > 0:
            logging.info(f"Deleted {articles_deleted} old articles and {briefings_deleted} old briefings.")
            logging.info("Reclaiming disk space...")
            con.execute("VACUUM")
            logging.info("Database cleanup complete.")
        else:
            logging.info("No old records to clean up.")

def collect_articles():
    """Scrapes RSS feeds for new articles and stores them with their embeddings."""
    logging.info("Starting article collection phase...")
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=48)

    with sqlite3.connect(DB_PATH) as con:
        cursor = con.cursor()
        for feed_url in FEEDS:
            logging.info(f"Processing feed: {feed_url}")
            try:
                feed = feedparser.parse(feed_url)

                # Get the main title of the feed to use as the source name
                source_name = feed.feed.get('title', 'Unknown Source')
                for entry in feed.entries:
                    url = entry.link

                    # 1. Skip if already in DB
                    cursor.execute("SELECT 1 FROM articles WHERE url=?", (url,))
                    if cursor.fetchone():
                        logging.info(f"Article already in database: {entry.title}. Skipping")
                        continue

                    # 2. Skip if too old
                    try:
                        pub_dt = _as_datetime(getattr(entry, "published", ""))
                        if pub_dt < cutoff:
                            logging.info(f"Article too old: {entry.title}. Skipping")
                            continue
                    except Exception:
                        logging.warning(f"Could not parse date for article: {entry.title}. Skipping.")
                        continue

                    # 3. Fetch full text
                    full_text = fetch_article(url)
                    len_full_text = len(full_text)
                    if not full_text or len_full_text < 300: # Skip short/empty articles
                        logging.info(f"Article too short: {entry.title} with {len_full_text} characters. Skipping")
                        continue

                    # Be a good web citizen
                    time.sleep(random.uniform(2.0, 5.0))

                    # 4. Generate token-aware chunked, nomalized embedding
                    try:
                        res = embed_article(entry.title, full_text, model="text-embedding-3-small")
                    except Exception as e:
                        logging.error("Embedding failed: %s", e)
                        continue

                    embedding_vec = res.get("doc_vector")

                    if not isinstance(embedding_vec, np.ndarray) or embedding_vec.ndim != 1 or embedding_vec.size == 0:
                        logging.error("Invalid embedding: missing/shape")
                        continue
                    if not np.isfinite(embedding_vec).all():
                        logging.error("Invalid embedding: non-finite values")
                        continue
                    n = np.linalg.norm(embedding_vec)
                    if not (0.9 <= n <= 1.1):
                        logging.error("Invalid embedding: bad norm %.3f", n)
                        continue

                    # warn if a fallback path was used
                    if res.get("used_fallback"):
                        logging.warning("Embedding fallback used (%s)", res.get("fallback_reason"))

                    # 5. Store in database
                    cursor.execute(
                        """INSERT INTO articles (source_name, url, title, full_text, published_date, embedding)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (source_name, url, entry.title, full_text, pub_dt.isoformat(), embedding_vec.tobytes())
                    )
                    logging.info(f"Stored article from {source_name}: {entry.title}")

            except Exception as e:
                logging.error(f"Failed to process feed {feed_url}: {e}")
        con.commit()
    logging.info("Article collection finished.")


def cluster_and_summarize():
    """Clusters articles using a two-pass Centroid-Based Re-clustering pipeline,
       with memory of past briefings."""
    logging.info("Starting Centroid Re-clustering and summarization phase...")
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        cursor = con.cursor()

        # Fetch all unprocessed articles, now including the source_name and URL
        cursor.execute("SELECT id, title, source_name, full_text, embedding, url FROM articles WHERE processed_for_digest = 0")
        rows = cursor.fetchall()

        if len(rows) < 20:
            logging.info(f"Not enough new articles to cluster ({len(rows)}). Need at least 20. Skipping.")
            return

        logging.info(f"Found {len(rows)} unprocessed articles.")

        article_ids = [row['id'] for row in rows]
        # Store URL and source_name in the articles_map
        articles_map = {row['id']: {'title': row['title'], 'source_name': row['source_name'], 'full_text': row['full_text'], 'url': row['url']} for row in rows}
        normalized_embeddings = np.array([np.frombuffer(row['embedding'], dtype=np.float32) for row in rows])

        # --- Find Previous Briefings for Context ---
        cutoff_date = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=3)
        cursor.execute("SELECT id, summary_content, embedding FROM briefings WHERE created_at > ?", (cutoff_date.isoformat(),))
        past_briefings = [{'id': r['id'], 'summary': r['summary_content'], 'embedding': np.frombuffer(r['embedding'], dtype=np.float32)} for r in cursor.fetchall() if r['embedding']]
        past_briefing_embeddings = np.array([b['embedding'] for b in past_briefings]) if past_briefings else np.array([])

        # --- CENTROID-BASED RE-CLUSTERING PIPELINE ---

        # 1. Perform a loose, initial clustering to discover potential topics.
        initial_clusters = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=3,
            metric="euclidean",
            cluster_selection_method="leaf",
        ).fit_predict(normalized_embeddings)
        logging.info(f"Found {len(set(initial_clusters)) - 1} initial topic groups.")

        # 2. Calculate the centroid for each initial topic cluster.
        centroids = []
        for cluster_id in set(initial_clusters):
            if cluster_id == -1: continue
            cluster_indices = np.where(initial_clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                centroid = np.mean(normalized_embeddings[cluster_indices], axis=0)
                centroids.append(normalize(centroid.reshape(1, -1))[0])

        if not centroids:
            logging.info("No stable initial clusters found to create centroids. Skipping.")
            return
        logging.info(f"Calculated {len(centroids)} topic centroids.")

        # 3. Re-cluster based on similarity to centroids.
        final_clusters = defaultdict(list)
        # Cosine similarity for normalized vectors is just the dot product.
        similarities = np.dot(normalized_embeddings, np.array(centroids).T)

        for i, article_sims in enumerate(similarities):
            best_cluster_id = np.argmax(article_sims)
            best_similarity = article_sims[best_cluster_id]

            if best_similarity >= SIMILARITY_THRESHOLD:
                final_clusters[best_cluster_id].append(i) # Use the centroid's index as the cluster ID

        logging.info(f"Formed {len(final_clusters)} pure clusters after similarity filtering.")

        # 4. Summarize the final, pure clusters
        for cluster_id, cluster_indices in final_clusters.items():
            if len(cluster_indices) < 3: # Don't summarize very small clusters
                logging.info(f"Pure cluster {cluster_id} is too small with {len(cluster_indices)} article(s). Skipping.")
                continue

            current_centroid = centroids[cluster_id]

            # --- Check for Continuing Story ---
            previous_summary = None
            if past_briefing_embeddings.size > 0:
                story_similarities = np.dot(past_briefing_embeddings, current_centroid.T)
                most_similar_past_briefing_idx = np.argmax(story_similarities)
                if story_similarities[most_similar_past_briefing_idx] >= CONTINUING_STORY_THRESHOLD:
                    previous_summary = past_briefings[most_similar_past_briefing_idx]['summary']
                    logging.info(f"Found continuing story for cluster {cluster_id}.")

            combined_text = ""
            source_info = [] # Store dicts of {'title': ..., 'url': ..., 'source_name': ...}
            for idx in cluster_indices:
                article_id = article_ids[idx]
                article_data = articles_map[article_id]
                combined_text += f"Article Title: {article_data['title']}\n\n{article_data['full_text']}\n\n---\n\n"
                source_info.append({
                    'title': article_data['title'],
                    'url': article_data['url'],
                    'source_name': article_data['source_name']
                })

            if not combined_text: continue

            logging.info(f"Summarizing pure cluster {cluster_id} with {len(cluster_indices)} articles...")
            try:
                system_prompt = (
                    "You are a world-class news analyst. Your task is to synthesize news articles into a sharp, concise briefing. "
                    "Output MUST be entirely in correct Dutch. After composing, check if the text is fully in Dutch; "
                    "if not, rewrite it entirely in correct Dutch. "
                    "Maintain a neutral, objective, fact-based tone similar to ANP or Reuters. "
                    "Do not include opinions, conclusions, observations, speculation, or value judgments. "
                    "Respond ONLY with a valid JSON object, using exactly these keys: "
                    "'title' (string), 'summary' (string), 'importance' (integer from 1 = minor news to 10 = major global event). "
                    "Do not add any extra commentary before or after the JSON object."
                )

                if previous_summary:
                    user_prompt = (
                        "Update the briefing for a developing story, focusing only on new developments compared to yesterday.\n\n"
                        f"YESTERDAY'S SUMMARY:\n{previous_summary}\n\n"
                        f"TODAY'S NEW ARTICLES:\n{combined_text[:90000]}\n\n"
                        "Produce an updated JSON object with 'title', 'summary', and 'importance'."
                    )
                else:
                    user_prompt = (
                        "Synthesize the following news articles into one compelling briefing.\n\n"
                        f"ARTICLES:\n{combined_text[:100000]}\n\n"
                        "Return a JSON object with 'title', 'summary', and 'importance'."
                    )

                response = client.chat.completions.create(
                    model="gpt-4o",
                    seed=42,               # Ensures reproducible results for identical inputs
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0,         # Deterministic, no creative drift
                    max_tokens=800,        # Enough for detailed summary without overflow
                    top_p=1,               # Keep selection deterministic with temp=0
                    frequency_penalty=0,   # Allow repetition if factual
                    presence_penalty=0,     # Avoid pushing new, non-factual content
                    response_format={"type": "json_object"}  # Enforce strict JSON output
                )
                data = json.loads(response.choices[0].message.content)
                summary_title = data.get("title", "Untitled Briefing") + (" (Update)" if previous_summary else "")
                summary_content = data.get("summary", "No summary available.")
                importance_score = data.get("importance", 5)

                cursor.execute(
                    "INSERT INTO briefings (summary_title, summary_content, source_articles, importance_score, embedding) VALUES (?, ?, ?, ?, ?)",
                    (summary_title, summary_content, json.dumps(source_info), importance_score, current_centroid.tobytes())
                )
                logging.info(f"Saved briefing: {summary_title} (Importance: {importance_score})")

            except Exception as e:
                logging.error(f"Failed to summarize cluster {cluster_id}: {e}")

        if article_ids:
            cursor.executemany("UPDATE articles SET processed_for_digest = 1 WHERE id = ?", [(id,) for id in article_ids])

        con.commit()
    logging.info("Clustering and summarization finished.")


def build_digest() -> str:
    """Builds an HTML digest from the latest generated briefings."""
    logging.info("Building HTML digest...")
    with sqlite3.connect(DB_PATH) as con:
        # Select all briefings that have not been sent yet
        rows = con.execute(
            "SELECT id, summary_title, summary_content, source_articles FROM briefings WHERE sent_in_digest = 0 ORDER BY importance_score DESC"
        ).fetchall()

    if not rows:
        logging.info("No new briefings to build a digest.")
        return "", []

    briefing_ids = [row[0] for row in rows]
    digest_parts, words = [], 0
    for row in rows:
        _, title, content, sources_str = row
        word_count = len(content.split())
        if words + word_count > READ_LIMIT_WORDS:
            break

        e_title = html.escape(title)
        e_content = html.escape(content).replace("\n", "<br>")

        block = [
            f'<h3>{e_title}</h3>',
            f"<p>{e_content}</p>",
        ]

        try:
            sources = json.loads(sources_str)
            if sources:
                # Create a list of clickable links using the source_name, URL, and Title
                source_items = "".join([f"<li>{html.escape(s.get('source_name', ''))}, <a href='{html.escape(s['url'])}'>{html.escape(s['title'])}</a></li>" for s in sources])
                block.append(f"<details><summary><em>Bronnen</em></summary><p><ul>{source_items}</ul></p></details>")
        except (json.JSONDecodeError, TypeError, KeyError):
            pass # Ignore if sources can't be parsed

        block.append("<hr>")
        digest_parts.append("\n".join(block))
        words += word_count

    if not digest_parts: return "", []

    html_body = (
        "<html><body style='line-height: 1.6;'>"
        f"<h2>Dagelijks Nieuwsoverzicht - {dt.datetime.now():%A, %d %B %Y}</h2>"
        + "\n".join(digest_parts) +
        "</body></html>"
    )
    return html_body, briefing_ids

def mark_briefings_as_sent(briefing_ids: list[int]):
    """Updates the database to mark a list of briefings as sent."""
    if not briefing_ids:
        return
    logging.info(f"Marking {len(briefing_ids)} briefings as sent...")
    with sqlite3.connect(DB_PATH) as con:
        con.executemany("UPDATE briefings SET sent_in_digest = 1 WHERE id = ?", [(id,) for id in briefing_ids])
        con.commit()
    logging.info("Briefings marked as sent.")

def send_mail(html_body: str):
    """Sends the HTML digest via email."""
    if not html_body:
        logging.info("HTML content is empty. Skipping email.")
        return

    logging.info(f"Sending digest to {os.environ['EMAIL_TO']}...")
    msg = MIMEText(html_body, "html", "utf-8")
    msg["Subject"] = "Dagelijks Nieuwsoverzicht - " + dt.datetime.now().strftime("%A, %d %B %Y")
    msg["From"] = os.environ["EMAIL_FROM"]
    msg["To"] = os.environ["EMAIL_TO"]
    msg["CC"] = os.environ["EMAIL_CC"]

    # Get BCC addresses from .env file
    bcc_string = os.environ["EMAIL_BCC"]
    bcc_list = [email.strip() for email in bcc_string.split(',')]

    # Extract all recipients for sendmail
    to_recipients = [email.strip() for email in os.environ["EMAIL_TO"].split(',')]
    cc_recipients = [email.strip() for email in os.environ["EMAIL_CC"].split(',') if os.environ["EMAIL_CC"]]
    all_recipients = to_recipients + cc_recipients + bcc_list

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ["SMTP_PORT"])) as s:
            s.starttls(context=ctx)
            s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
            s.sendmail(os.environ["EMAIL_FROM"], all_recipients, msg.as_string())

        logging.info(f"Email sent successfully to {len(all_recipients)} recipients (TO: {len(to_recipients)}, CC: {len(cc_recipients)}, BCC: {len(bcc_list)})")

    except Exception as e:
        logging.error(f"Failed to send email: {e}")

def main():
    """Main function to run the full news digest pipeline."""
    init_db()
    cleanup_database()
    collect_articles()
    cluster_and_summarize()
    digest_html, briefing_ids = build_digest()
    if digest_html:
        # For debugging, you can print the HTML
#        print(digest_html)
        send_mail(digest_html)
        mark_briefings_as_sent(briefing_ids)
    else:
        logging.info("No new digest was generated.")


if __name__ == "__main__":
    main()
