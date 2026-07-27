# 本地 PDF 检索基准

## 目的

本基准验证电子信息文献 Agent 的本地 PDF 检索链路，比较纯 BM25 与真实 Embedding 驱动的 Hybrid 方案。所有数值由同一脚本、同一语料和同一标签实际运行得到，不使用模拟结果。

## 数据与标注

- 语料：20 篇文本型 PDF，共 228 页、958,056 个可提取字符和 1,027 个页面分块。
- 主题：OFDM 信道估计、MIMO 检测、频谱感知、无线通信大模型，每类 5 篇。
- 查询：`eval/local_pdf_cases.json` 中 40 条中英文查询，每篇论文 1 条英文和 1 条中文查询。
- 标签：人工核对论文首页标题和摘要，将相关论文第 1 页作为相关结果。
- 指标：Hit@5、Recall@5、Precision@5、引用准确率、MRR、nDCG@5、平均延迟、P95 延迟和缓存命中率。

标注集曾用于定位页面级语义召回问题，因此它是人工核对的开发基准，不是独立、盲测的测试集。结果适合证明工程链路和当前方案改进，不应外推为通用学术结论。

## 对照方案

### BM25

对 1,027 个页面分块执行词法召回，直接返回页面级结果和引用。

### Hybrid

1. 对页面分块执行 BM25，保留精确术语和页码定位能力。
2. 将每篇 PDF 首页的标题与摘要合并为一条论文级语义记录。
3. 使用 SiliconFlow 的 `BAAI/bge-m3` 生成真实向量并做余弦召回。
4. 使用加权 RRF 融合页面级 BM25 与论文级语义排名。
5. Embedding 服务不可用时自动降级为 BM25，并在工具元数据中记录原因。

语义索引只包含 20 条论文级记录，而不是 1,027 个页面分块。这一设计减少向量计算，并缓解页面级语义检索“论文正确但页码错误”的问题。

## 真实结果

测试条件：相同的 40 条查询、20 篇论文、`K=5`，Hybrid 使用真实远程 Embedding 服务。

| 指标 | BM25 | Hybrid | Hybrid 相对变化 |
| --- | ---: | ---: | ---: |
| Hit@5 | 0.550 | 0.800 | +45.5% |
| Recall@5 | 0.550 | 0.800 | +45.5% |
| Precision@5 | 0.110 | 0.160 | +45.5% |
| 引用准确率 | 0.110 | 0.160 | +45.5% |
| MRR | 0.379 | 0.654 | +72.6% |
| nDCG@5 | 0.421 | 0.691 | +64.1% |
| 平均延迟 | 142.35 ms | 280.90 ms | +97.3% |
| P95 延迟 | 121.10 ms | 280.94 ms | +132.0% |

Hybrid 明显改善了主题级中英文查询的召回和排序，但平均延迟约翻倍。实际部署应通过持久化向量缓存、批量 Embedding 和更严格的独立测试集继续验证成本收益。

## 复现

BM25：

```powershell
python scripts/evaluate_local_pdf_retrieval.py `
  --library data/papers `
  --cases eval/local_pdf_cases.json `
  --output eval/results/bm25.json `
  --retrieval-mode bm25 `
  --k 5
```

Hybrid（API Key 仅放在被忽略的 `.env`）：

```powershell
python scripts/evaluate_local_pdf_retrieval.py `
  --library data/papers `
  --cases eval/local_pdf_cases.json `
  --output eval/results/hybrid.json `
  --retrieval-mode hybrid `
  --embedding-model openai:BAAI/bge-m3 `
  --embedding-base-url https://api.siliconflow.cn/v1 `
  --k 5
```

`data/papers/`、`.env` 和 `eval/results/` 均被 Git 忽略。仓库只提交标注、评测代码和汇总结果，不分发论文、密钥或个人报告。

## 端到端验收

完整 LangGraph 使用本地 Hybrid 检索生成了 18,307 字符的中文报告：

- 覆盖 OFDM 信道估计、MIMO 信号检测和频谱感知三个方向。
- 包含 149 次 `local-pdf://文件#page=页码` 引用。
- 包含 29 个唯一文件页码引用，覆盖 15 篇 PDF。
- 所有引用文件存在，页码均在相应 PDF 的有效范围内。
- 未出现报告生成错误或“未通过本地文档引用验证”标记。

最终报告保存在本地 `eval/results/final_report.md`，该目录不进入 Git。

## 局限与下一步

- 数据量小，且查询集中在四个无线通信主题。
- 同一论文的中英文查询不是完全独立样本。
- 当前标签主要衡量主题级论文首页召回，不覆盖细粒度方法段和实验表格定位。
- 当前结果是开发基准；后续应按论文划分互不重叠的开发集和盲测集，并冻结盲测标签。
- 延迟数据受本机、网络和远程 Embedding 服务状态影响，需要多轮重复测量并报告方差。
