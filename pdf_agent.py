import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from llm import llm
from tools import semantic_scholar_search
from state import State
from config import MAX_PDFS
from memory_utils import get_current_user_question


PDF_QUERY_PROMPT = """
You are an academic search specialist.

Given the user's research question, relevant prior research memory,
and approved research plan,
generate the BEST possible Semantic Scholar search query
to find highly relevant academic papers with open-access PDFs.

Rules:
- Return ONLY the search query string, nothing else
- Make it specific and academic in tone
- Focus on the core research topic
- If prior memory is relevant, search for papers that verify, extend, update, or fill gaps in that memory
- Use key technical terms
- Do NOT use boolean operators or quotes
- Maximum 10 words
"""

PDF_SUMMARIZER_PROMPT = """
You are an academic paper summarizer.

Given the full text of an academic paper, extract and summarize:
1. Main research question or objective
2. Methodology used
3. Key findings and results
4. Conclusions and implications
5. Any specific data, benchmarks, or statistics mentioned

Be concise but comprehensive.
Focus on information relevant to the user's question: {user_question}
"""


def load_pdf_from_url(pdf_url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(pdf_url, headers=headers, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        os.unlink(tmp_path)

        full_text = "\n\n".join(doc.page_content for doc in documents)
        return full_text[:8000]

    except Exception as e:
        return f"ERROR loading PDF: {e}"


def summarize_pdf(pdf_text: str, user_question: str, paper_title: str) -> str:
    if pdf_text.startswith("ERROR"):
        return pdf_text

    prompt = PDF_SUMMARIZER_PROMPT.format(user_question=user_question)
    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Paper: {paper_title}\n\nFull Text:\n{pdf_text}")
    ])
    return response.content


def _parse_papers_with_pdfs(search_results: str) -> list[tuple[str, str]]:
    papers = []
    for paper_block in search_results.split("\n---\n"):
        title = ""
        pdf_url = ""
        for line in paper_block.strip().split("\n"):
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("PDF URL:") and "NOT AVAILABLE" not in line:
                pdf_url = line.replace("PDF URL:", "").strip()
        if pdf_url:
            papers.append((title, pdf_url))
    return papers


def _load_and_summarize_paper(title: str, pdf_url: str, user_question: str) -> str:
    print(f"[pdf_agent] Loading and summarizing: {title[:60]}...")
    pdf_text = load_pdf_from_url(pdf_url)
    summary = summarize_pdf(pdf_text, user_question, title)
    print(f"[pdf_agent] Summarized: {title[:60]}")
    return (
        f"### {title}\n"
        f"**Source:** {pdf_url}\n\n"
        f"{summary}"
    )


def pdf_research_agent(state: State) -> dict:
    print("\n[pdf_agent] START")

    user_question = get_current_user_question(state)

    print("[pdf_agent] Generating academic search query...")
    memory_context = state.get("long_term_context", "")
    query_response = llm.invoke([
        SystemMessage(content=PDF_QUERY_PROMPT),
        HumanMessage(content=(
            f"User question: {user_question}\n\n"
            f"Relevant prior research memory:\n{memory_context[:2000]}\n\n"
            f"Approved research plan:\n{state.get('plan', '')[:2000]}\n\n"
            f"Generate the best Semantic Scholar search query:"
        ))
    ])
    search_query = query_response.content.strip().strip('"')
    print(f"[pdf_agent] Query: {search_query}")

    print("[pdf_agent] Searching Semantic Scholar...")
    search_results = semantic_scholar_search.invoke({
        "query": search_query,
        "limit": 10
    })

    candidates = _parse_papers_with_pdfs(search_results)[:MAX_PDFS]
    if not candidates:
        print("[pdf_agent] No open-access PDFs found in search results.")
        return {
            "pdf_output": "No open-access PDFs were found for this topic via Semantic Scholar."
        }

    print(f"[pdf_agent] Processing {len(candidates)} PDF(s) in parallel (max={MAX_PDFS})...")
    pdf_summaries = []

    with ThreadPoolExecutor(max_workers=MAX_PDFS) as executor:
        futures = {
            executor.submit(_load_and_summarize_paper, title, pdf_url, user_question): title
            for title, pdf_url in candidates
        }
        for future in as_completed(futures):
            title = futures[future]
            try:
                pdf_summaries.append(future.result())
            except Exception as e:
                print(f"[pdf_agent] Failed for {title[:60]}: {e}")
                pdf_summaries.append(f"### {title}\n\nERROR: {e}")

    combined = "\n\n---\n\n".join(pdf_summaries)
    print(f"[pdf_agent] DONE — {len(pdf_summaries)} PDFs processed")
    return {"pdf_output": combined}
