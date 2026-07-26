# Deep Research 中文深度研究 Agent

基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建的多智能体深度研究系统。用户输入研究问题后，系统会自动澄清需求、生成研究简报、拆分子任务、调用 Tavily 检索资料、压缩研究结果，并输出带来源引用的中文研究报告。

本仓库基于 LangChain 官方项目 [open_deep_research](https://github.com/langchain-ai/open_deep_research) 进行二次开发，增加了中文交互、DeepSeek V4 Pro 兼容、可信学术检索和本地 PDF RAG。

## 项目亮点

- **多智能体协作**：Supervisor 负责任务规划与调度，多个 Researcher 可并发完成子课题研究。
- **完整研究闭环**：覆盖需求澄清、任务规划、联网检索、资料压缩、结果汇总和报告生成。
- **Tool Calling**：模型自主决定搜索时机，通过 Tavily 获取实时网页资料，也可扩展 MCP 工具。
- **学术文献检索**：集成 arXiv、Crossref 与 Semantic Scholar，结构化提取论文元数据并自动补全 DOI、引用指标和开放获取信息。
- **引用可信控制**：按 DOI、arXiv ID 和标题去重，综合相关性、来源质量与学术影响力进行排序并生成引用白名单。
- **本地 PDF RAG**：使用 PyMuPDF 按页解析 PDF，支持 BM25 与 Embedding/RRF 混合召回，返回带文件名、页码的证据片段。
- **可量化评测**：内置 Hit@K、Recall@K、MRR、nDCG、引用准确率、延迟和缓存命中率评测脚本。
- **安全降级与缓存**：PDF 内容或 Embedding 模型变化时自动失效缓存；语义服务异常时回退到 BM25，避免中断研究流程。
- **DeepSeek V4 Pro 适配**：关闭不兼容的 Thinking Mode，并使用 Function Calling 完成结构化输出。
- **中文化交互**：核心提示词已本地化，支持中文研究任务与中文报告生成。
- **可观测性**：接入 LangSmith，可追踪节点输入输出、工具调用、Token 消耗、耗时与异常。
- **安全与稳定性**：支持调用次数限制、并发限制、结构化输出重试和上下文超限处理。

## 系统架构

```mermaid
flowchart TD
    A[用户输入研究问题] --> B{是否需要澄清}
    B -- 是 --> C[向用户提出澄清问题]
    C --> A
    B -- 否 --> D[生成结构化研究简报]
    D --> E[Supervisor 制定研究计划]
    E --> F1[Researcher 子智能体 1]
    E --> F2[Researcher 子智能体 2]
    E --> F3[Researcher 子智能体 N]
    F1 --> G[Tavily / 学术检索 / 本地 PDF / MCP]
    F2 --> G
    F3 --> G
    G --> H[压缩并整理研究资料]
    H --> E
    E --> I{资料是否充分}
    I -- 否 --> F1
    I -- 是 --> J[论文去重、评分与引用验证]
    J --> K[生成带引用的最终报告]

    L[LangSmith] -. 链路追踪 .-> B
    L -. 链路追踪 .-> E
    L -. 链路追踪 .-> G
    L -. 链路追踪 .-> K
```

## 技术栈

| 类型 | 技术 |
| --- | --- |
| 开发语言 | Python 3.10+ |
| Agent 编排 | LangGraph、LangChain |
| 大语言模型 | DeepSeek V4 Pro（也保留多模型提供商支持） |
| 联网搜索 | Tavily Search API |
| 学术检索 | arXiv、Crossref、Semantic Scholar |
| 本地检索 | PyMuPDF、BM25、Embedding、RRF |
| 工具协议 | Tool Calling、MCP |
| 数据校验 | Pydantic |
| 异步并发 | asyncio |
| 调试追踪 | LangSmith、LangGraph Studio |
| 测试评估 | pytest、Deep Research Bench |

## 工作流程

1. `clarify_with_user` 判断研究范围是否清晰，必要时向用户追问。
2. `write_research_brief` 将对话转换成结构化研究简报。
3. `research_supervisor` 拆分任务并通过 `ConductResearch` 调度子智能体。
4. `researcher` 使用 ReAct 与 Tool Calling 循环调用 Tavily、arXiv、Crossref、本地 PDF 或 MCP 工具。
5. `compress_research` 去重、压缩资料并保留来源信息。
6. Supervisor 判断资料是否充分；不足时继续研究，充分时结束调度。
7. `normalize_academic_sources` 解析论文记录，使用 Crossref 和 Semantic Scholar 补全元数据，执行去重、综合评分和引用验证。
8. `final_report_generation` 根据研究笔记和学术引用白名单生成最终报告。

## 项目结构

```text
deep_research/
├── src/
│   ├── open_deep_research/
│   │   ├── configuration.py     # Studio 配置与运行参数
│   │   ├── academic_tools.py    # 学术检索、元数据补全、论文评分和引用验证
│   │   ├── local_pdf_tools.py   # PDF 解析、混合召回、缓存和本地引用验证
│   │   ├── retrieval_evaluation.py # 标准检索指标计算
│   │   ├── deep_researcher.py   # LangGraph 主图和子图
│   │   ├── model_compat.py      # DeepSeek V4 兼容层
│   │   ├── prompts.py           # 中文提示词
│   │   ├── state.py             # Graph 状态和结构化输出模型
│   │   └── utils.py             # 搜索、MCP、Token 与模型工具
│   ├── legacy/                  # 早期工作流和多智能体实现
│   └── security/                # API 鉴权逻辑
├── tests/                       # 评估脚本与评测器
├── scripts/                     # 本地 PDF 检索评测 CLI
├── eval/                        # 评测集样例；真实结果默认不提交
├── data/papers/                 # 本地 PDF 语料库；默认被 Git 忽略
├── examples/                    # 示例研究报告
├── langgraph.json               # LangGraph 服务入口
├── pyproject.toml               # Python 依赖与项目配置
└── .env.example                 # 环境变量模板
```

## 快速开始

### 1. 克隆仓库

```powershell
git clone https://github.com/lyx001124/deep_research.git
cd deep_research
git switch feat/electronic-literature-agent
```

如果功能分支已经合并到 `main`，可省略最后一条命令。

### 2. 创建 Python 环境

使用 Conda：

```powershell
conda create -n agent python=3.11 -y
conda activate agent
python -m pip install -e .
```

确认开发服务器版本满足 Studio 追踪要求：

```powershell
python -m pip install --upgrade "langgraph-api>=0.11.0" "langgraph-cli[inmem]"
python -m pip check
```

### 3. 配置环境变量

在项目根目录创建 `.env`：

```dotenv
# DeepSeek：两个变量可使用同一个 API Key
OPENAI_API_KEY=your_deepseek_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# 搜索
TAVILY_API_KEY=your_tavily_api_key

# 四个阶段使用的模型
SUMMARIZATION_MODEL=openai:deepseek-v4-pro
RESEARCH_MODEL=openai:deepseek-v4-pro
COMPRESSION_MODEL=openai:deepseek-v4-pro
FINAL_REPORT_MODEL=openai:deepseek-v4-pro

# 本地开发
GET_API_KEYS_FROM_CONFIG=false

# 学术检索与引用验证
ACADEMIC_SEARCH_ENABLED=true
CROSSREF_ENRICHMENT_ENABLED=true
CITATION_VERIFICATION_ENABLED=true
MAX_PAPERS_PER_QUERY=10
MAX_ACADEMIC_PAPERS=20
SEMANTIC_SCHOLAR_ENABLED=true
SEMANTIC_SCHOLAR_ENRICHMENT_ENABLED=true
ACADEMIC_IMPACT_WEIGHT=0.2
# 可选，配置后可提高 Semantic Scholar 限流额度
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
# PUBLICATION_YEAR_START=2023
# PUBLICATION_YEAR_END=2026

# 可选：本地 PDF 文献库（默认关闭）
LOCAL_PDF_SEARCH_ENABLED=true
PDF_LIBRARY_PATH=D:/agent/deep_research/data/papers
MAX_LOCAL_PDF_FILES=20
MAX_LOCAL_PDF_RESULTS=6
LOCAL_PDF_CHUNK_SIZE=1200
LOCAL_PDF_CHUNK_OVERLAP=200
LOCAL_PDF_CACHE_ENABLED=true
LOCAL_PDF_CACHE_MAX_ENTRIES=8
# 默认 bm25；启用 hybrid 后需要单独的 Embedding 模型与密钥
LOCAL_PDF_RETRIEVAL_MODE=bm25
LOCAL_PDF_EMBEDDING_MODEL=openai:text-embedding-3-small
# LOCAL_PDF_EMBEDDING_BASE_URL=https://api.openai.com/v1
# LOCAL_PDF_EMBEDDING_API_KEY=your_embedding_api_key
LOCAL_PDF_HYBRID_CANDIDATE_LIMIT=30
LOCAL_PDF_LEXICAL_WEIGHT=0.5

# 可选：LangSmith 链路追踪
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=deep-research-deepseek-v4
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

`.env` 已加入 `.gitignore`，请勿使用 `git add -f .env`，也不要把真实 API Key 写入代码或 README。

### 4. 启动服务

Windows CMD：

```cmd
set PYTHONUTF8=1
langgraph dev --allow-blocking
```

Windows PowerShell：

```powershell
$env:PYTHONUTF8 = "1"
langgraph dev --allow-blocking
```

启动后可访问：

- API：<http://127.0.0.1:2024>
- API 文档：<http://127.0.0.1:2024/docs>
- Studio：终端输出的 LangGraph Studio URL

## Studio 配置

首次测试建议使用低成本配置：

| 配置项 | 建议值 |
| --- | --- |
| Allow Clarification | `False` |
| Max Concurrent Research Units | `1` |
| Search API | `tavily` |
| Max Researcher Iterations | `2` |
| Max React Tool Calls | `3` |
| Summarization Model | `openai:deepseek-v4-pro` |
| Research Model | `openai:deepseek-v4-pro` |
| Compression Model | `openai:deepseek-v4-pro` |
| Final Report Model | `openai:deepseek-v4-pro` |
| MCP Config | 留空 |
| Recursion Limit | `25` |

学术研究配置还提供 `Academic Search Enabled`、`Crossref Enrichment Enabled`、`Semantic Scholar Enabled`、`Semantic Scholar Enrichment Enabled`、`Citation Verification Enabled`、`Academic Impact Weight`、论文数量上限和可选年份范围。

### 本地 PDF RAG

1. 在仓库中创建 `data/papers`，把需要分析的文本型 PDF 放入该目录；该目录已被 Git 忽略。
2. 在 `.env` 设置 `LOCAL_PDF_SEARCH_ENABLED=true` 和绝对路径 `PDF_LIBRARY_PATH`。
3. 重启 `langgraph dev --allow-blocking`。Researcher 会在需要时调用 `search_local_pdfs`，并以 `local-pdf://文件名#page=页码` 标注证据。

工具只能读取配置目录中的 PDF，不接受模型传入的磁盘路径；单文件限制 50 MB、500 页，加密、损坏和无法提取文本的文件会跳过。默认使用本地 BM25 词法检索，不需要 Embedding API 或向量数据库；扫描版 PDF 仍需要后续接入 OCR。

将 `LOCAL_PDF_RETRIEVAL_MODE` 改为 `hybrid` 后，系统分别执行 BM25 与余弦语义召回，再用加权 Reciprocal Rank Fusion（RRF）融合两份排名。Embedding 模型通过 LangChain 的标准模型标识配置，不要求与 DeepSeek 报告模型来自同一供应商；Embedding 服务不可用时自动降级为 BM25，不中断研究流程。

PDF 分块索引与文档向量默认缓存在进程内。缓存键包含文件内容指纹、Embedding 模型和分块参数，因此文献增删改或模型变化后自动失效；缓存数量有上限，避免长期运行无限占用内存。工具结果会返回缓存命中状态、索引耗时、检索模式、降级原因和检索耗时。

#### 本地验证语料

开发阶段使用 20 篇 arXiv 公开论文进行本地验证，每个方向 5 篇。PDF 仅用于个人研究与测试，不随仓库分发；`data/papers/` 已加入 `.gitignore`。

| 方向 | 数量 | arXiv ID |
| --- | ---: | --- |
| OFDM 信道估计 | 5 | `2107.07161`、`2306.13761`、`2210.06588`、`2401.02035`、`2305.13487` |
| 深度学习 MIMO 检测 | 5 | `1901.05647`、`1907.09439`、`2105.05044`、`1706.01151`、`2205.10620` |
| 深度学习频谱感知 | 5 | `1909.06020`、`2003.08359`、`2504.07427`、`2307.14985`、`2401.04805` |
| 无线通信大模型 | 5 | `2505.22320`、`2501.09631`、`2307.07319`、`2507.21524`、`2408.02944` |

本地可读性检查结果：

| 指标 | 结果 |
| --- | ---: |
| PDF 数量 | 20 |
| 总页数 | 228 |
| 可提取字符数 | 958,056 |
| 生成分块数 | 1,027 |
| 无法解析文件 | 0 |

以上结果只证明 PDF 解析、分块和基础召回链路可用，不代表 Hybrid 已获得确定的质量提升。质量提升必须在同一人工标注评测集上对 BM25 和 Hybrid 进行 A/B 测试后报告。

### 本地 PDF 检索评测

复制 [评测样例](eval/local_pdf_cases.example.json)，为每个查询标注相关 PDF 的相对路径、页码和相关性等级：

```json
{
  "id": "ofdm-channel-estimation",
  "query": "OFDM channel estimation pilot neural network",
  "relevant": [
    {"relative_path": "ofdm_survey.pdf", "page": 3, "grade": 2}
  ]
}
```

运行离线评测，不需要 LangSmith 或模型 API：

```powershell
python scripts/evaluate_local_pdf_retrieval.py `
  --library D:/agent/deep_research/data/papers `
  --cases eval/local_pdf_cases.json `
  --output eval/results/bm25.json `
  --k 5
```

使用同一标注集评测混合召回，才能与 BM25 基线做有效对比：

```powershell
$env:LOCAL_PDF_EMBEDDING_API_KEY = "your_embedding_api_key"
python scripts/evaluate_local_pdf_retrieval.py `
  --library D:/agent/deep_research/data/papers `
  --cases eval/local_pdf_cases.json `
  --output eval/results/hybrid.json `
  --retrieval-mode hybrid `
  --embedding-model openai:text-embedding-3-small `
  --k 5
```

脚本输出 `Hit@K`、`Recall@K`、`Precision@K`、引用准确率、`MRR`、`nDCG@K`、平均/P95 延迟和缓存命中率。建议使用至少 30–50 条人工标注查询分别生成 `bm25.json` 和 `hybrid.json`；简历中只填写真实评测数据，不要使用示例集结果。

### 完整测试

运行无需外部服务的完整测试套件：

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m pytest -q -p no:cacheprovider `
  --ignore=src/legacy/tests/test_report_quality.py
```

当前开发环境验证结果为 `43 passed`。被忽略的旧版报告质量测试需要访问 LangSmith 外部数据集，不属于本地离线回归。

如果 Windows 系统临时目录拒绝访问，可以显式指定仓库内的测试临时目录：

```powershell
New-Item -ItemType Directory -Force .pytest_cache/tmp | Out-Null
python -m pytest -q -p no:cacheprovider `
  --basetemp .pytest_cache/tmp `
  --ignore=src/legacy/tests/test_report_quality.py
```

评测结果保存在 `eval/results/bm25.json` 和 `eval/results/hybrid.json`。比较时重点关注：

- `Recall@K`：相关页面被召回的比例。
- `MRR`：第一个相关页面出现的位置。
- `nDCG@K`：高相关页面是否排在前面。
- `Citation Accuracy`：返回引用中正确引用的比例。
- `P95 Latency`：最慢一批查询的响应时间。

质量指标越高越好，延迟指标越低越好。仓库不提供虚构的提升百分比，只有使用真实论文和人工标注得到的数据才适合写入简历。

Studio 中保存的 Assistant 配置会参与运行时模型选择，因此需要确认四个模型字段均已改为 DeepSeek V4 Pro。

## 测试问题

简单连通性测试：

```text
什么是信道估计？请搜索两个可靠来源，并用中文简要说明其基本原理、常用方法和工程应用。
```

完整研究流程测试：

```text
请调研近三年大语言模型在无线通信信号处理中的应用，重点比较信道估计、信号检测和频谱感知三个方向的技术路线。至少引用 5 个可靠来源，说明各方案的优势、局限性及工程落地难点，并给出未来研究趋势。
```

本地 PDF 与引用验证测试：

```text
请结合本地 PDF 文献库，比较深度学习在 OFDM 信道估计、MIMO 信号检测和频谱感知中的技术路线。说明模型结构、性能指标和工程局限，并为每项主要结论提供文件名及页码引用；不要引用检索结果中不存在的论文。
```

运行后在 Studio 中确认 `search_local_pdfs` 的工具结果包含：

```xml
requested_mode="hybrid"
effective_mode="hybrid"
fallback_reason=""
```

如果 `effective_mode="bm25"`，说明 Embedding 服务不可用，系统已自动降级；可以根据 `fallback_reason` 检查 API Key、模型名称或 Base URL。

## DeepSeek V4 Pro 兼容说明

DeepSeek V4 Pro 在部分 OpenAI 兼容调用中不支持默认 `response_format`，Thinking Mode 也可能与强制 `tool_choice` 冲突。本项目仅对模型名包含 `deepseek-v4` 的模型应用以下适配：

```python
extra_body = {"thinking": {"type": "disabled"}}
structured_output = {"method": "function_calling"}
```

其他模型提供商仍保持 LangChain 的默认行为。兼容逻辑位于 `src/open_deep_research/model_compat.py`。

## LangSmith 调试

开启 `LANGSMITH_TRACING=true` 后，可在 LangSmith 项目中查看：

- 每个 LangGraph 节点的输入和输出
- Supervisor 的任务拆分与 Researcher 调度过程
- DeepSeek 的 Tool Calling 参数
- Tavily 搜索请求与返回结果
- Token 消耗、延迟和异常堆栈

请勿使用包含隐私、密码或敏感资料的问题进行公开追踪测试。

## 后续计划

- [x] 接入 arXiv、Crossref 与 Semantic Scholar 学术检索
- [x] 增加论文去重、来源质量、引用影响力评分与引用白名单检查
- [x] 支持配置本地 PDF 文献库，并结合本地文档与网络资料研究
- [x] 增加可选 Embedding 语义召回与 BM25/RRF 混合检索
- [ ] 增加上传接口、OCR、持久化向量数据库和 Cross-encoder 重排
- [x] 增加本地 PDF 索引缓存和标准检索指标评测
- [ ] 增加网络搜索缓存、成本统计和端到端生成质量评测
- [ ] 导出 Markdown、PDF 和 Word 格式研究报告
- [ ] 使用 Docker 与 CI/CD 完成部署

## 致谢与许可

本项目基于 LangChain 团队的 [open_deep_research](https://github.com/langchain-ai/open_deep_research) 二次开发，感谢原项目作者和贡献者。

项目沿用原仓库的 [MIT License](LICENSE)。
