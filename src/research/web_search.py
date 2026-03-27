"""
Web research pipeline:
  1. DuckDuckGo search (no API key)
  2. Trafilatura for full article text extraction
  3. Jina Reader fallback for JS-heavy pages
  4. Tavily fallback when DDG rate-limits
"""

import time
from typing import Optional


def search_and_extract(
    query: str,
    max_results: int = 5,
    max_chars: int = 2000,
    tavily_api_key: str = "",
    jina_enabled: bool = True,
) -> list[dict]:
    """
    Returns a list of dicts: [{"url": ..., "title": ..., "content": ...}]
    Falls back gracefully if any component fails.
    """
    urls = _ddg_search(query, max_results, tavily_api_key)
    results = []
    for url, title in urls:
        content = _extract_content(url, max_chars, jina_enabled)
        if content:
            results.append({"url": url, "title": title, "content": content})
        if len(results) >= max_results:
            break
    return results


def format_research_for_prompt(results: list[dict], max_chars: int = 3000) -> str:
    """Format extracted research results for injection into an agent prompt."""
    if not results:
        return ""
    lines = ["=== WEB RESEARCH ==="]
    total = 0
    for r in results:
        snippet = r["content"][:max_chars - total]
        lines.append(f"\nSource: {r['url']}\nTitle: {r['title']}\n{snippet}")
        total += len(snippet)
        if total >= max_chars:
            break
    return "\n".join(lines)


# ------------------------------------------------------------------
# DDG search
# ------------------------------------------------------------------

def _ddg_search(query: str, max_results: int, tavily_api_key: str) -> list[tuple]:
    """Returns list of (url, title). Falls back to Tavily on rate limit."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            return [(r["href"], r["title"]) for r in results if "href" in r]
    except Exception as e:
        if "ratelimit" in str(e).lower() and tavily_api_key:
            return _tavily_search(query, max_results, tavily_api_key)
        print(f"  [research] DDG search failed: {e}")
        return []


def _tavily_search(query: str, max_results: int, api_key: str) -> list[tuple]:
    """Tavily fallback — returns (url, title) pairs."""
    try:
        import requests
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"query": query, "max_results": max_results, "api_key": api_key},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return [(r["url"], r.get("title", "")) for r in data.get("results", [])]
    except Exception as e:
        print(f"  [research] Tavily fallback failed: {e}")
        return []


# ------------------------------------------------------------------
# Content extraction
# ------------------------------------------------------------------

def _extract_content(url: str, max_chars: int, jina_enabled: bool) -> str:
    """Try Trafilatura first, then Jina Reader as fallback."""
    content = _trafilatura_extract(url)
    if not content and jina_enabled:
        content = _jina_extract(url)
    return content[:max_chars] if content else ""


def _trafilatura_extract(url: str) -> str:
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            return text or ""
    except Exception as e:
        print(f"  [research] trafilatura failed for {url}: {e}")
    return ""


def _jina_extract(url: str) -> str:
    try:
        import requests
        resp = requests.get(f"https://r.jina.ai/{url}", timeout=15)
        if resp.status_code == 200:
            return resp.text
    except Exception as e:
        print(f"  [research] Jina failed for {url}: {e}")
    return ""
