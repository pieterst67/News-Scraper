# get_article_with_browser.py
# Requires: pip install requests
import random
import requests
import json
import re
import logging
import lxml.html
from typing import Dict, List, Optional, Any, Tuple
from trafilatura import extract
from newspaper import Article
from pathlib import Path
from urllib.parse import urljoin

USER_AGENTS = [
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/123.0.6312.87 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14; SM-S908B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Mobile Safari/537.36",
]

ADD_SCRIPT_TAGS = [
    {"content": "(() => { function clickFull(){const c=Array.from(document.querySelectorAll('button,a,[role=\"button\"]'));const b=c.find(el=>/show\\s*full\\s*article/i.test((el.textContent||'')));if(b){b.click();return true;}return false;}if(!clickFull()){const mo=new MutationObserver((_,o)=>{if(clickFull())o.disconnect()});mo.observe(document.documentElement,{childList:true,subtree:true});setTimeout(()=>mo.disconnect(),8000);}})();"},
#    {"content": "(() => { Object.defineProperty(Intl, 'DateTimeFormat', { writable: true, value: new Proxy(Intl.DateTimeFormat, { construct: (t, a) => new t('en-US', Object.assign({}, a[1])) }) }); })();"},
    {"content": "(() => { const bs = Array.from(document.querySelectorAll('button, a')).filter(el => el.textContent.toLowerCase().includes('accept') && (el.textContent.toLowerCase().includes('cookie') || el.textContent.toLowerCase().includes('cookies'))); bs.forEach(b => b.click()); })();"},
#    {"content": "(() => { const pw = Array.from(document.querySelectorAll('div, section')).filter(el => el.id.toLowerCase().includes('paywall') || el.className.toLowerCase().includes('paywall') || el.id.toLowerCase().includes('subscribe') || el.className.toLowerCase().includes('subscribe')); pw.forEach(p => p.remove()); })();"},
#    {"content": "(() => { document.querySelectorAll('script, style, iframe, .ad, .ads, .advertisement, [class*=\"social\"], [id*=\"social\"], .share, .comments, aside, nav, header:not(article header), footer').forEach(el => el.remove()); })();"},
#    {"content": "(() => { const keep=['href','src','alt','title']; document.querySelectorAll('*').forEach(el => { [...el.attributes].forEach(a => { if(!keep.includes(a.name.toLowerCase())) el.removeAttribute(a.name); }); }); })();"},
#    {"content": "(() => { function rm(){let n=0; document.querySelectorAll('div, span, p, section, article').forEach(el=>{ if(!el.hasChildNodes()||el.textContent.trim()===''){el.remove(); n++;}}); return n;} let tries=0; const i=setInterval(()=>{if(rm()===0||++tries>5)clearInterval(i);},1000); })();"},
#    {"content": "(() => { document.querySelectorAll('meta').forEach(m=>{ if(m.attributes.length<=1) m.remove(); }); })();"}
]

def get_top_image(html: str) -> Optional[str]:
    if not html:
        return None

    try:
        # Parse the HTML content using lxml
        doc = lxml.html.fromstring(html)
    except lxml.etree.ParserError:
        return None

    # 1. Check for OpenGraph 'og:image' meta tag
    og_image = doc.xpath('//meta[@property="og:image"]/@content')
    if og_image:
        return og_image[0]

    # 2. Check for Twitter Card 'twitter:image' meta tag
    twitter_image = doc.xpath('//meta[@name="twitter:image"]/@content')
    if twitter_image:
        return twitter_image[0]

    # 3. Check for 'image_src' link tag
    # Note: The provided HTML has this tag, but it points to the article URL, not an image.
    image_src_link = doc.xpath('//link[@rel="image_src"]/@href')
    if image_src_link and '.jpg' in image_src_link[0]: # Basic check to ensure it's an image
        return image_src_link[0]

    # 4. Fallback: Find the first image within the main article content
    # This is the crucial part for lc.nl, which places the main image inside the <article> tag.
    main_article_img = doc.xpath('//article//img/@src')
    if main_article_img:
        # Loop through found images to find the most suitable one
        for img_src in main_article_img:
            # We filter out small, non-descriptive images like logos or icons.
            # Here, we assume a good candidate is a non-SVG file.
            if 'logo.svg' not in img_src and '.svg' not in img_src:
                return img_src

    return None

def extract_text_from_html(html: str, url: str) -> Optional[Dict[str, Any]]:
    if not html:
        return None

    min_chars = 200

    for params in ({"favor_precision": True}, {"favor_recall": True}):
        out = extract(
            html,
            url=url,
            with_metadata=True,
            include_comments=False,
            deduplicate=True,
            output_format="json",
            **params,
        )
        if not out:
            logging.warning("Extract retry: failed first time")
            continue
        d = json.loads(out)
        text = (d.get("text") or "").strip()
        if len(text) >= min_chars:
            title = (d.get("title") or "").strip()
            image = get_top_image(html) or ""
            return {
                "title": title,
                "text": text,
                "image": image,
            }
        logging.warning("Extract retry: less than {min_chars} characters returned the first time")

    return None

def get_article_with_browser(env: Dict[str, str], url: str, timeout: int = 45) -> Tuple[str, str, str]:
    try:
        endpoint = f"https://api.cloudflare.com/client/v4/accounts/{env['CLOUDFLARE_ACCOUNT_ID']}/browser-rendering/content"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {env['CLOUDFLARE_API_TOKEN']}",
        }
        payload = {
            "url": url,
            "userAgent": random.choice(USER_AGENTS),
            "setExtraHTTPHeaders": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "DNT": "1",
                "Accept-Language": "en-US,en;q=0.5",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
            },
            "gotoOptions": {"waitUntil": "networkidle0", "timeout": 30000, "referer": "https://www.google.com/"},
            "viewport": {"width": 390, "height": 844, "deviceScaleFactor": 3, "isMobile": True, "hasTouch": True, "isLandscape": False},
            "rejectResourceTypes": ["image", "media", "font", "websocket"],
            "bestAttempt": True,
            "addScriptTag": ADD_SCRIPT_TAGS,
            "waitForSelector": {"selector": "article, .article, .content, .post, #article, main", "timeout": 5000},
        }
        r = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()

        try:
            data = r.json()
        except ValueError:
            logging.error("Non-JSON response from Cloudflare: %s", r.text[:500])
            return "", "", ""

        if not bool(data.get("success", data.get("status", False))):
            logging.error("Render failure: %s", data.get("errors", data))
            return "", "", ""

        result = data.get("result")
        if isinstance(result, str):
            html = result
        elif isinstance(result, dict):
            html = result.get("content") or result.get("html") or ""
        else:
            html = ""

        if not html:
            return "", "", ""

        art = extract_text_from_html(html, url) or {}
        return (
            art.get("text", "") if isinstance(art, dict) else "",
            art.get("title", "") if isinstance(art, dict) else "",
            art.get("image", "") if isinstance(art, dict) else "",
        )
    except Exception:
        return "", "", ""

def get_article_with_get(url: str, timeout: int = 45) -> Tuple[str, str, str]:
    try:
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Referer": "https://www.google.com/",
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()

        art = extract_text_from_html(r.text, url) or {}
        return (
            art.get("text", "") if isinstance(art, dict) else "",
            art.get("title", "") if isinstance(art, dict) else "",
            art.get("image", "") if isinstance(art, dict) else "",
        )
    except Exception:
        return "", "", ""
