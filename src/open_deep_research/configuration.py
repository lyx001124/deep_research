"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Academic research configuration
    academic_search_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否为研究智能体启用 arXiv 与 Crossref 学术检索工具"
            }
        }
    )
    crossref_enrichment_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否使用 Crossref 补充论文 DOI、期刊和出版年份"
            }
        }
    )
    semantic_scholar_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否启用 Semantic Scholar 学术检索工具"
            }
        }
    )
    semantic_scholar_enrichment_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否使用 Semantic Scholar 补充引用量、影响力引用量和开放获取信息"
            }
        }
    )
    citation_verification_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否在报告生成前后验证学术引用并移除未知链接"
            }
        }
    )
    max_papers_per_query: int = Field(
        default=10,
        ge=1,
        le=30,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10,
                "min": 1,
                "max": 30,
                "description": "单次学术检索最多返回的论文数量"
            }
        }
    )
    max_academic_papers: int = Field(
        default=20,
        ge=1,
        le=100,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 20,
                "min": 1,
                "max": 100,
                "description": "去重和排序后提供给最终报告的论文数量上限"
            }
        }
    )
    academic_impact_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.2,
                "min": 0.0,
                "max": 0.5,
                "description": "综合排序中学术影响力分数的权重"
            }
        }
    )
    local_pdf_search_enabled: bool = Field(
        default=False,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": False,
                "description": "是否启用配置目录内的本地 PDF 文献检索"
            }
        }
    )
    pdf_library_path: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "本地 PDF 文献库目录；研究工具不能访问此目录之外的文件"
            }
        }
    )
    max_local_pdf_files: int = Field(
        default=20,
        ge=1,
        le=100,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 20,
                "min": 1,
                "max": 100,
                "description": "单次建立本地检索索引时读取的 PDF 数量上限"
            }
        }
    )
    max_local_pdf_results: int = Field(
        default=6,
        ge=1,
        le=20,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 6,
                "min": 1,
                "max": 20,
                "description": "单次本地 PDF 检索返回的片段数量上限"
            }
        }
    )
    local_pdf_chunk_size: int = Field(
        default=1200,
        ge=400,
        le=4000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 1200,
                "min": 400,
                "max": 4000,
                "description": "PDF 文本分块的字符数"
            }
        }
    )
    local_pdf_chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=800,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 200,
                "min": 0,
                "max": 800,
                "description": "相邻 PDF 文本块的重叠字符数"
            }
        }
    )
    local_pdf_cache_enabled: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "是否缓存本地 PDF 分块索引；文件变化时自动失效"
            }
        }
    )
    local_pdf_cache_max_entries: int = Field(
        default=8,
        ge=1,
        le=64,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8,
                "min": 1,
                "max": 64,
                "description": "进程内最多保留的 PDF 索引版本数量"
            }
        }
    )
    local_pdf_retrieval_mode: str = Field(
        default="bm25",
        pattern="^(bm25|hybrid)$",
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "bm25",
                "description": "本地 PDF 使用纯 BM25 或 BM25 与 Embedding 的混合召回",
                "options": [
                    {"label": "BM25", "value": "bm25"},
                    {"label": "Hybrid", "value": "hybrid"},
                ],
            }
        },
    )
    local_pdf_embedding_model: str = Field(
        default="openai:text-embedding-3-small",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:text-embedding-3-small",
                "description": "混合召回使用的 LangChain Embedding 模型标识",
            }
        },
    )
    local_pdf_embedding_base_url: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "可选的 OpenAI 兼容 Embedding API 地址",
            }
        },
    )
    local_pdf_hybrid_candidate_limit: int = Field(
        default=30,
        ge=5,
        le=200,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 30,
                "min": 5,
                "max": 200,
                "description": "BM25 和语义检索各自参与 RRF 融合的候选数量",
            }
        },
    )
    local_pdf_lexical_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "RRF 融合中的 BM25 权重；语义权重为 1 减去该值",
            }
        },
    )
    publication_year_start: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "description": "可选的论文发表年份下限"
            }
        }
    )
    publication_year_end: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2100,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "description": "可选的论文发表年份上限"
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:deepseek-v4-pro",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:deepseek-v4-pro",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="openai:deepseek-v4-pro",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:deepseek-v4-pro",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:deepseek-v4-pro",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:deepseek-v4-pro",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:deepseek-v4-pro",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:deepseek-v4-pro",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True
