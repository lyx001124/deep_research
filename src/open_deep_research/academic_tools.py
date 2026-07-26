"""Academic search, normalization, ranking, and citation validation utilities."""

import asyncio
import json
import re
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Annotated, Any, Optional

import arxiv
import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from open_deep_research.configuration import Configuration
from open_deep_research.state import Paper


PAPER_RECORDS_START = "<academic_papers>"
PAPER_RECORDS_END = "</academic_papers>"
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
ARXIV_ID_PATTERN = re.compile(
    r"(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+-]{1,}|[\u4e00-\u9fff]{2,}")


def normalize_doi(value: Optional[str]) -> Optional[str]:
    """Return a canonical DOI or None when the value is invalid."""
    if not value:
        return None
    doi = value.strip().lower()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi)
    doi = doi.rstrip(".,;)")
    return doi if DOI_PATTERN.match(doi) else None


def normalize_arxiv_id(value: Optional[str]) -> Optional[str]:
    """Return a canonical arXiv identifier without a version suffix."""
    if not value:
        return None
    match = ARXIV_ID_PATTERN.search(value)
    if not match:
        return None
    return re.sub(r"v\d+$", "", match.group(1), flags=re.IGNORECASE).lower()


def normalize_title(value: str) -> str:
    """Normalize a title for deterministic duplicate detection."""
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = re.sub(r"[^\w\u4e00-\u9fff]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _clean_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    return " ".join(str(value or "").split())


def _crossref_year(item: dict[str, Any]) -> Optional[int]:
    for field in ("published-print", "published-online", "issued", "created"):
        date_parts = item.get(field, {}).get("date-parts", [])
        if date_parts and date_parts[0]:
            try:
                return int(date_parts[0][0])
            except (TypeError, ValueError):
                pass
    return None


def _crossref_authors(item: dict[str, Any]) -> list[str]:
    authors = []
    for author in item.get("author", []):
        name = " ".join(
            part for part in (author.get("given", ""), author.get("family", "")) if part
        ).strip()
        if name:
            authors.append(name)
    return authors


def paper_to_dict(paper: Paper | dict[str, Any]) -> dict[str, Any]:
    """Validate a paper and return a JSON-compatible dictionary."""
    if not isinstance(paper, Paper):
        paper = Paper.model_validate(paper)
    return paper.model_dump(mode="json")


def serialize_papers(papers: list[Paper | dict[str, Any]]) -> str:
    """Serialize paper records inside markers that downstream nodes can parse safely."""
    payload = [paper_to_dict(paper) for paper in papers]
    return (
        f"{PAPER_RECORDS_START}\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n"
        f"{PAPER_RECORDS_END}"
    )


def extract_papers_from_text(text: str) -> list[dict[str, Any]]:
    """Extract all marked academic paper arrays from tool and research messages."""
    if not text:
        return []
    pattern = re.compile(
        re.escape(PAPER_RECORDS_START) + r"\s*(.*?)\s*" + re.escape(PAPER_RECORDS_END),
        re.DOTALL,
    )
    papers: list[dict[str, Any]] = []
    for payload in pattern.findall(text):
        try:
            records = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(records, list):
            continue
        for record in records:
            try:
                papers.append(paper_to_dict(record))
            except (TypeError, ValueError):
                continue
    return papers


def extract_papers_from_notes(notes: list[str]) -> list[dict[str, Any]]:
    """Extract paper records from a list of research notes."""
    papers: list[dict[str, Any]] = []
    for note in notes:
        papers.extend(extract_papers_from_text(str(note)))
    return papers


async def _search_arxiv(query: str, max_results: int) -> list[Paper]:
    def fetch() -> list[Paper]:
        client = arxiv.Client(page_size=max_results, delay_seconds=3.0, num_retries=2)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for result in client.results(search):
            papers.append(
                Paper(
                    title=_clean_text(result.title),
                    authors=[_clean_text(author.name) for author in result.authors],
                    abstract=_clean_text(result.summary),
                    published_year=result.published.year if result.published else None,
                    doi=normalize_doi(result.doi),
                    arxiv_id=normalize_arxiv_id(result.entry_id),
                    url=result.entry_id,
                    source="arxiv",
                    venue=_clean_text(result.journal_ref) or None,
                    research_direction=query,
                    verification_status="source_verified",
                )
            )
        return papers

    return await asyncio.to_thread(fetch)


def filter_papers_by_year(
    papers: list[Paper],
    year_start: Optional[int],
    year_end: Optional[int],
) -> list[Paper]:
    """Filter known publication years while retaining records with unknown years."""
    filtered = []
    for paper in papers:
        year = paper.published_year
        if year is not None and year_start is not None and year < year_start:
            continue
        if year is not None and year_end is not None and year > year_end:
            continue
        filtered.append(paper)
    return filtered


@tool(
    description=(
        "检索 arXiv 学术论文并返回机器可验证的结构化元数据，包括标题、作者、摘要、年份、"
        "DOI、arXiv ID 和原始页面。适合电子信息、通信、信号处理和人工智能主题。"
        "学术主题应优先使用准确的英文查询词；禁止根据返回结果之外的信息编造论文。"
    )
)
async def search_arxiv(
    query: str,
    max_results: Annotated[Optional[int], InjectedToolArg] = None,
    config: RunnableConfig = None,
) -> str:
    """Search arXiv and return marked JSON paper records."""
    configurable = Configuration.from_runnable_config(config)
    limit = min(max_results or configurable.max_papers_per_query, configurable.max_papers_per_query)
    papers = await _search_arxiv(query, limit)
    papers = filter_papers_by_year(
        papers,
        configurable.publication_year_start,
        configurable.publication_year_end,
    )
    return serialize_papers(papers)


async def _query_crossref_api(
    title: str,
    rows: int,
    client: Optional[httpx.AsyncClient] = None,
) -> list[dict[str, Any]]:
    owns_client = client is None
    client = client or httpx.AsyncClient(
        timeout=httpx.Timeout(15.0),
        headers={"User-Agent": "deep-research-academic-agent/1.0"},
    )
    try:
        response = await client.get(
            "https://api.crossref.org/works",
            params={
                "query.title": title,
                "rows": rows,
                "select": "DOI,title,author,published-print,published-online,issued,created,URL,container-title,abstract,type",
            },
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("items", [])
    finally:
        if owns_client:
            await client.aclose()


def crossref_item_to_paper(item: dict[str, Any]) -> Optional[Paper]:
    """Convert one Crossref work item to the normalized Paper schema."""
    title = _clean_text(item.get("title"))
    doi = normalize_doi(item.get("DOI"))
    url = item.get("URL") or (f"https://doi.org/{doi}" if doi else None)
    if not title or not url:
        return None
    return Paper(
        title=title,
        authors=_crossref_authors(item),
        abstract=_clean_text(item.get("abstract")),
        published_year=_crossref_year(item),
        doi=doi,
        url=url,
        source="crossref",
        venue=_clean_text(item.get("container-title")) or None,
        verification_status="source_verified",
    )


def rank_crossref_matches(
    title: str,
    items: list[dict[str, Any]],
    threshold: float = 0.82,
) -> list[Paper]:
    """Return Crossref records whose normalized titles closely match the query."""
    normalized_query = normalize_title(title)
    matches: list[tuple[float, Paper]] = []
    for item in items:
        paper = crossref_item_to_paper(item)
        if not paper:
            continue
        score = SequenceMatcher(
            None, normalized_query, normalize_title(paper.title)
        ).ratio()
        if score >= threshold:
            matches.append((score, paper))
    matches.sort(key=lambda pair: pair[0], reverse=True)
    return [paper for _, paper in matches]


@tool(
    description=(
        "通过 Crossref 查询论文的 DOI、作者、期刊或会议名称、出版年份和规范 URL。"
        "适合补全或核验已知论文标题的元数据，不用于编造论文结论。"
    )
)
async def query_crossref(
    title: str,
    max_results: Annotated[Optional[int], InjectedToolArg] = None,
    config: RunnableConfig = None,
) -> str:
    """Query Crossref and return marked JSON paper records."""
    configurable = Configuration.from_runnable_config(config)
    limit = min(max_results or 5, configurable.max_papers_per_query)
    items = await _query_crossref_api(title, limit)
    papers = rank_crossref_matches(title, items)
    papers = filter_papers_by_year(
        papers,
        configurable.publication_year_start,
        configurable.publication_year_end,
    )
    return serialize_papers(papers)


def _paper_identity(paper: Paper) -> tuple[str, str]:
    if paper.doi:
        return "doi", paper.doi
    if paper.arxiv_id:
        return "arxiv", paper.arxiv_id
    return "title", normalize_title(paper.title)


def _merge_papers(existing: Paper, incoming: Paper) -> Paper:
    """Merge two records, preferring verified and more complete metadata."""
    records = sorted(
        (existing, incoming),
        key=lambda item: (
            item.verification_status == "source_verified",
            bool(item.doi),
            len(item.abstract),
            len(item.authors),
        ),
        reverse=True,
    )
    primary, secondary = records
    data = primary.model_dump()
    for field in ("abstract", "published_year", "doi", "arxiv_id", "venue", "research_direction"):
        if not data.get(field) and getattr(secondary, field):
            data[field] = getattr(secondary, field)
    if len(secondary.authors) > len(data.get("authors", [])):
        data["authors"] = secondary.authors
    if primary.source != secondary.source:
        sources = set(primary.source.split("+")) | set(secondary.source.split("+"))
        data["source"] = "+".join(sorted(sources))
    if data.get("doi"):
        data["url"] = f"https://doi.org/{data['doi']}"
    return Paper.model_validate(data)


async def enrich_papers_with_crossref(
    records: list[Paper | dict[str, Any]],
    enabled: bool = True,
    max_concurrency: int = 3,
) -> list[Paper]:
    """Fill missing DOI metadata using high-confidence Crossref title matches."""
    papers = [record if isinstance(record, Paper) else Paper.model_validate(record) for record in records]
    if not enabled:
        return papers
    candidates = [paper for paper in papers if not paper.doi and paper.title]
    semaphore = asyncio.Semaphore(max_concurrency)

    async def enrich(paper: Paper) -> Paper:
        async with semaphore:
            try:
                items = await _query_crossref_api(paper.title, 3)
            except (httpx.HTTPError, ValueError, KeyError):
                return paper
        matches = rank_crossref_matches(paper.title, items, threshold=0.88)
        return _merge_papers(paper, matches[0]) if matches else paper

    enriched = await asyncio.gather(*(enrich(paper) for paper in candidates))
    by_title = {normalize_title(paper.title): paper for paper in enriched}
    return [by_title.get(normalize_title(paper.title), paper) for paper in papers]


def deduplicate_papers(records: list[Paper | dict[str, Any]]) -> list[Paper]:
    """Deduplicate by DOI, arXiv ID, then normalized title and merge metadata."""
    deduplicated: list[Paper] = []
    identities: dict[tuple[str, str], int] = {}
    for record in records:
        try:
            paper = record if isinstance(record, Paper) else Paper.model_validate(record)
        except (TypeError, ValueError):
            continue
        paper.doi = normalize_doi(paper.doi)
        paper.arxiv_id = normalize_arxiv_id(paper.arxiv_id)
        identity = _paper_identity(paper)
        existing_index = identities.get(identity)
        if existing_index is None and identity[0] != "title":
            existing_index = identities.get(("title", normalize_title(paper.title)))
        if existing_index is None:
            existing_index = len(deduplicated)
            deduplicated.append(paper)
        else:
            deduplicated[existing_index] = _merge_papers(deduplicated[existing_index], paper)
        merged = deduplicated[existing_index]
        keys = {_paper_identity(merged), ("title", normalize_title(merged.title))}
        if merged.doi:
            keys.add(("doi", merged.doi))
        if merged.arxiv_id:
            keys.add(("arxiv", merged.arxiv_id))
        for key in keys:
            identities[key] = existing_index
    return deduplicated


def _query_tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_PATTERN.findall(text or "")}


def score_paper(paper: Paper, research_brief: str) -> Paper:
    """Assign deterministic relevance and source-quality scores."""
    query_tokens = _query_tokens(research_brief)
    paper_tokens = _query_tokens(f"{paper.title} {paper.abstract} {paper.research_direction or ''}")
    overlap = len(query_tokens & paper_tokens) / max(1, len(query_tokens))
    title_similarity = SequenceMatcher(
        None, normalize_title(research_brief), normalize_title(paper.title)
    ).ratio()
    paper.relevance_score = round(min(1.0, 0.75 * overlap + 0.25 * title_similarity), 4)

    source_base = 0.75 if paper.doi else 0.65 if paper.arxiv_id else 0.45
    metadata = 0.05 * sum(
        bool(value) for value in (paper.authors, paper.abstract, paper.published_year, paper.venue)
    )
    recency = 0.0
    if paper.published_year:
        age = max(0, datetime.now(timezone.utc).year - paper.published_year)
        recency = max(0.0, 0.05 - min(age, 10) * 0.005)
    paper.quality_score = round(min(1.0, source_base + metadata + recency), 4)
    paper.verification_status = "verified" if paper.url and (paper.doi or paper.arxiv_id) else "source_verified"
    return paper


def normalize_rank_and_verify_papers(
    records: list[Paper | dict[str, Any]],
    research_brief: str,
    limit: int,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize, filter, deduplicate, score, and split accepted/rejected records."""
    accepted: list[Paper] = []
    rejected: list[dict[str, Any]] = []
    for record in deduplicate_papers(records):
        if record.published_year and year_start and record.published_year < year_start:
            rejected.append({**paper_to_dict(record), "rejection_reason": "before_year_start"})
            continue
        if record.published_year and year_end and record.published_year > year_end:
            rejected.append({**paper_to_dict(record), "rejection_reason": "after_year_end"})
            continue
        if not record.url or not (record.doi or record.arxiv_id):
            rejected.append({**paper_to_dict(record), "rejection_reason": "missing_stable_identifier"})
            continue
        accepted.append(score_paper(record, research_brief))
    accepted.sort(key=lambda item: (item.relevance_score, item.quality_score), reverse=True)
    rejected.extend(
        {**paper_to_dict(item), "rejection_reason": "result_limit"}
        for item in accepted[limit:]
    )
    return [paper_to_dict(item) for item in accepted[:limit]], rejected


def build_citation_context(papers: list[dict[str, Any]]) -> str:
    """Build a compact citation whitelist for the report model."""
    if not papers:
        return "未从学术工具中提取到可验证论文。不得虚构论文、DOI 或 URL。"
    lines = ["以下是唯一允许作为学术参考文献引用的论文白名单："]
    lines.append(f"共获得 {len(papers)} 篇已验证且去重后的论文。")
    for index, paper in enumerate(papers, 1):
        authors = ", ".join(paper.get("authors", [])[:6]) or "作者未知"
        identifier = paper.get("doi") or paper.get("arxiv_id")
        lines.append(
            f"[{index}] {paper['title']} | {authors} | {paper.get('published_year') or '年份未知'} | "
            f"{identifier} | {paper['url']}"
        )
    lines.append("不得增加白名单之外的论文、DOI 或学术 URL；信息不足时必须明确说明。")
    return "\n".join(lines)


def sanitize_report_citations(report: str, papers: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Replace unknown DOI/arXiv URLs in a report and return rejected URLs."""
    if not report:
        return report, []
    allowed = {paper["url"].rstrip("/").lower() for paper in papers if paper.get("url")}
    allowed.update(
        f"https://doi.org/{paper['doi']}".lower() for paper in papers if paper.get("doi")
    )
    allowed.update(
        f"https://arxiv.org/abs/{paper['arxiv_id']}".lower()
        for paper in papers
        if paper.get("arxiv_id")
    )
    allowed.update(
        f"https://arxiv.org/abs/{paper['arxiv_id']}".lower()
        for paper in papers
        if paper.get("arxiv_id")
    )
    academic_pattern = re.compile(
        r"https?://[^\s)\]>]*(?:doi\.org|arxiv\.org)[^\s)\]>]*",
        re.IGNORECASE,
    )
    rejected = []

    def replace(match: re.Match[str]) -> str:
        url = match.group(0).rstrip(".,;")
        if url.rstrip("/").lower() in allowed:
            return match.group(0)
        rejected.append(url)
        return "[未通过引用验证]"

    return academic_pattern.sub(replace, report), rejected
