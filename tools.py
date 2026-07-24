import requests
from langchain_core.tools import tool
from tavily import TavilyClient
from config import EXA_API_KEY, TAVILY_API_KEY

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


@tool
def exa_search(query: str, num_results: int = 5) -> str:
    """Search the web using Exa AI. Best for semantic/meaning-based research queries."""
    response = requests.post(
        "https://api.exa.ai/search",
        headers={
            "x-api-key": EXA_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "numResults": num_results,
            "contents": {"text": {"maxCharacters": 800}}
        }
    )
    response.raise_for_status()
    data = response.json()
    results = []
    for r in data.get("results", []):
        results.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Content: {r.get('text', '')}\n"
        )
    return "\n---\n".join(results) if results else "No results found."


@tool
def exa_find_similar(url: str, num_results: int = 3) -> str:
    """Find pages similar to a given URL using Exa AI."""
    response = requests.post(
        "https://api.exa.ai/findSimilar",
        headers={
            "x-api-key": EXA_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "url": url,
            "numResults": num_results,
            "contents": {"text": {"maxCharacters": 800}}
        }
    )
    response.raise_for_status()
    data = response.json()
    results = []
    for r in data.get("results", []):
        results.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Content: {r.get('text', '')}\n"
        )
    return "\n---\n".join(results) if results else "No results found."


@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily. Best for factual, current, and agent-optimized queries."""
    response = tavily_client.search(
        query=query,
        max_results=max_results,
        include_answer=True,
        include_raw_content=False
    )
    results = []
    if response.get("answer"):
        results.append(f"Quick Answer: {response['answer']}\n")
    for r in response.get("results", []):
        results.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Content: {r.get('content', '')}\n"
            f"Score: {r.get('score', '')}\n"
        )
    return "\n---\n".join(results) if results else "No results found."


@tool
def semantic_scholar_search(query: str, limit: int = 10) -> str:
    """
    Search Semantic Scholar for academic papers.
    Returns paper titles, abstracts, years, and open-access PDF URLs.
    """
    response = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search/bulk",
        params={
            "query": query,
            "limit": limit,
            "fields": "title,abstract,year,authors,openAccessPdf,externalIds"
        }
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for paper in data.get("data", []):
        pdf_info = paper.get("openAccessPdf")
        pdf_url = pdf_info.get("url") if pdf_info else None
        authors = ", ".join(
            a.get("name", "") for a in paper.get("authors", [])[:3]
        )
        results.append(
            f"Title: {paper.get('title', 'N/A')}\n"
            f"Year: {paper.get('year', 'N/A')}\n"
            f"Authors: {authors}\n"
            f"Abstract: {(paper.get('abstract') or 'N/A')[:200]}\n"
            f"PDF URL: {pdf_url if pdf_url else 'NOT AVAILABLE'}\n"
        )

    return "\n---\n".join(results) if results else "No papers found."


TOOLS = [exa_search, exa_find_similar, tavily_search, semantic_scholar_search]

TOOL_MAP = {
    "exa_search": exa_search,
    "exa_find_similar": exa_find_similar,
    "tavily_search": tavily_search,
    "semantic_scholar_search": semantic_scholar_search,
}