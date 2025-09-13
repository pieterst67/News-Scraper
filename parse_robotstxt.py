import re
import requests
from urllib.parse import urlsplit, urlunsplit

def _robots_url(page_url: str) -> str:
    parts = list(urlsplit(page_url))
    parts[2] = "/robots.txt"
    parts[3] = ""
    parts[4] = ""
    return urlunsplit(parts)

def _parse_robots_for_crawl_delay(robots_txt: str, user_agent: str) -> float | None:
    ua = user_agent.lower()
    groups = []
    current = {"uas": [], "crawl_delay": None}
    prev_was_ua = False

    for raw in robots_txt.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            if current["uas"] or current["crawl_delay"] is not None:
                groups.append(current)
            current = {"uas": [], "crawl_delay": None}
            prev_was_ua = False
            continue

        m = re.match(r"(?i)^\s*([a-z][a-z0-9\-]*)\s*:\s*(.+)\s*$", line)
        if not m:
            continue
        field, value = m.group(1).lower(), m.group(2).strip()

        if field == "user-agent":
            if not prev_was_ua and (current["uas"] or current["crawl_delay"] is not None):
                groups.append(current)
                current = {"uas": [], "crawl_delay": None}
            current["uas"].append(value.lower())
            prev_was_ua = True
        else:
            prev_was_ua = False
            if field == "crawl-delay":
                try:
                    current["crawl_delay"] = float(value)
                except ValueError:
                    pass

    if current["uas"] or current["crawl_delay"] is not None:
        groups.append(current)

    best_len, best_delay = -1, None
    for g in groups:
        if g["crawl_delay"] is None or not g["uas"]:
            continue
        for token in g["uas"]:
            if token == "*" or token in ua:
                tlen = 1 if token == "*" else len(token)
                if tlen > best_len:
                    best_len, best_delay = tlen, g["crawl_delay"]
                break

    return best_delay

def get_crawl_delay(page_url: str, user_agent: str, timeout: float = 10.0) -> float | None:
    """Return Crawl-delay (seconds) for the best-matching UA, or None if not set/available."""
    r = requests.get(
        _robots_url(page_url),
        headers={"User-Agent": user_agent, "Accept": "text/plain,*/*;q=0.1"},
        timeout=timeout,
        allow_redirects=True,
    )
    if r.status_code >= 400 or not r.text:
        return None
    return _parse_robots_for_crawl_delay(r.text, user_agent)

# Example:
# delay = get_crawl_delay("https://example.com/some/page", "MyCrawler/1.0 (+https://example.org/bot)")
# print(delay)
