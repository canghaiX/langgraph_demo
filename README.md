# LangGraph + LightRAG Demo

一个面向中文场景的多 Agent RAG 项目示例，基于 `LangGraph / LangChain / LightRAG / MCP` 搭建。

它不是单纯的“LLM + 一个检索接口” Demo，而是把知识库检索、本地代码库分析、联网搜索拆成不同专家能力，再由 Router Agent 按问题类型进行分流和汇总。

## 项目亮点

- 多 Agent 架构：将系统拆分为 `Router / Knowledge Specialist / Filesystem Specialist / Web Specialist`
- RAG 检索链路：集成 LightRAG 做知识库召回，并支持 CrossEncoder rerank
- 非结构化文档处理：支持 `.txt / .md / .pdf` 入库，PDF 优先直抽文本，失败时自动回退 OCR
- 可观测性：支持 trace 输出，方便查看 Router 分发、专家调用和底层工具调用轨迹
- MCP 工具集成：通过本地 MCP Server 接入文件系统检索与网页搜索能力
- CLI 友好：支持 `ingest / ask / chat` 三类命令，方便本地调试和演示

## 适用场景

- 课程设计或个人项目中的 Agent + RAG 系统实践
- 中文知识库问答 Demo
- 面向简历展示的 LLM 应用工程项目
- 研究 LangGraph、MCP、多工具路由和 LightRAG 集成方式的参考项目

## 项目结构

```text
.
├─ src/langgraph_lightrag_demo/
│  ├─ cli.py                  # CLI 入口
│  ├─ config.py               # 统一配置管理
│  ├─ graph.py                # 多 Agent 编排与工具路由
│  ├─ lightrag_client.py      # LightRAG 接入、文档解析与入库
│  ├─ reranker.py             # CrossEncoder 重排
│  └─ mcp_servers/
│     ├─ filesystem_server.py # 本地文件 MCP Server
│     └─ web_search_server.py # 网页搜索 MCP Server
├─ skills/                    # 项目内 skill 规则，动态拼入 prompt
├─ tests/                     # 基础测试
├─ data/                      # 示例数据目录
├─ main.py                    # 极薄启动入口
└─ .env.example               # 环境变量示例
```

## 系统架构

```text
User Question
    |
    v
Router Agent
    |----> Knowledge Specialist
    |         |----> LightRAG retrieval
    |         |----> Reranker
    |
    |----> Filesystem Specialist
    |         |----> Filesystem MCP tools
    |
    |----> Web Specialist
              |----> Web Search MCP tools
```

### 路由思路

- 如果问题不需要工具，Router 直接回答
- 如果问题依赖已入库资料，优先交给 Knowledge Specialist
- 如果问题依赖当前工作区文件，优先交给 Filesystem Specialist
- 如果问题依赖公开网页或最新信息，优先交给 Web Specialist
- 默认避免一次性调用所有专家，优先做最小必要路由

## 核心能力

### 1. 知识库问答

- 使用 LightRAG 管理知识库存储与召回
- 当前实现让 LightRAG 只返回检索上下文，不直接输出最终答案
- 最终回答由上层 Agent 结合上下文进行组织，便于和其他工具结果汇总

### 2. 检索后重排

- 使用 `sentence-transformers` 的 CrossEncoder 作为 reranker
- 对 LightRAG 已召回上下文做二次排序，提升相关片段靠前概率
- reranker 采用懒加载，避免程序启动即加载本地大模型

### 3. PDF 入库与 OCR 回退

- 对电子版 PDF，优先用 `pypdf` 直接抽取文本
- 如果抽取得到的正文过少，自动回退到 `PyMuPDF + RapidOCR`
- 内置基础文本清洗逻辑，包括：
  - 页码过滤
  - 噪声行过滤
  - 图注/表注保留
  - 跨页重复页眉页脚清理
  - 页码标记插入，便于后续溯源

### 4. 来源标注与 trace

- Prompt 中统一要求基于工具证据回答，并尽量标注来源
- 支持在 CLI 中通过 `--trace` 查看跨层级调用轨迹
- trace 会记录 Router 分发、专家调用、工具调用与结果预览

## 环境要求

- Python 3.10+
- 可用的 OpenAI-compatible 聊天模型接口
- 可用的 OpenAI-compatible Embedding 接口
- 如果要使用 reranker，需要本地 `bge-reranker-large` 或兼容模型目录
- 如果要处理扫描版 PDF，需要安装 OCR 相关依赖

## 安装

### 1. 创建虚拟环境

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

## 配置

复制环境变量模板：

```bash
copy .env.example .env
```

然后按你的模型服务修改 `.env`。核心配置如下：

```env
CHAT_API_KEY=EMPTY
CHAT_BASE_URL=http://127.0.0.1:8000/v1
CHAT_MODEL=Qwen3-8B

EMBEDDING_API_KEY=EMPTY
EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024

RERANKER_MODEL_PATH=/path/to/bge-reranker-large
ENABLE_MCP_WEB_SEARCH=true
ENABLE_MCP_FILESYSTEM=true
MCP_FILESYSTEM_ROOT=.
```

### 关键配置说明

- `CHAT_*`：聊天模型配置，供 Router 和各专家 Agent 使用
- `EMBEDDING_*`：Embedding 模型配置，供 LightRAG 建库与检索使用
- `LIGHTRAG_*`：LightRAG 工作目录、查询模式、响应类型等配置
- `RERANKER_*`：本地 reranker 模型路径和推理设备
- `ENABLE_MCP_*`：控制是否启用 Web / Filesystem 专家对应的 MCP 工具
- `SYSTEM_PROMPT`：全局系统提示词

## 使用方式

### 1. 导入知识库

默认导入 `data/knowledge` 目录下的 `.txt / .md / .pdf` 文件：

```bash
python main.py ingest
```

也可以手动指定文件或目录：

```bash
python main.py ingest --path data/knowledge
```

### 2. 单轮问答

```bash
python main.py ask "这个项目的多 agent 架构是怎么设计的？"
```

打开 trace：

```bash
python main.py ask "README 里提到了哪些功能？" --trace
```

### 3. 多轮对话

```bash
python main.py chat
```

打开 trace：

```bash
python main.py chat --trace
```

### 4. 通过安装后的命令运行

```bash
rag-chat ingest
rag-chat ask "请解释 Router Agent 的职责"
rag-chat chat --trace
```

## 支持的文件类型

- `.txt`
- `.md`
- `.pdf`

其中 PDF 支持：

- 文本层直接提取
- 扫描版 OCR 回退
- 基础页眉页脚清洗
- 页码来源标记

## 测试

运行测试：

```bash
python -m pytest -q
```

当前测试主要覆盖：

- 配置校验
- CLI 文件收集
- graph 中的文本收敛与 trace 逻辑
- PDF 文本清洗中的部分关键规则

## 性能指标评估

为了避免项目只停留在“能跑”的层面，建议从检索效果、路由效果、响应性能和资源消耗四个维度进行评估。即使当前仓库里还没有完整评测脚本，也建议在项目文档和简历中明确你的评估方法与后续指标补充计划。

### 1. 检索效果评估

适用于 Knowledge Specialist 的 LightRAG + rerank 链路。

建议关注这些指标：

- `Recall@K`：正确证据是否出现在前 K 个召回结果中
- `MRR`：正确证据是否排在更靠前位置
- `Hit Rate`：一次检索是否命中至少一个有效证据片段
- `Rerank Gain`：加入 reranker 前后，正确片段排序是否明显提升

建议做法：

1. 准备一组问答评测集，每条样本包含：
   - 问题
   - 标准答案
   - 对应证据文件名或页码
2. 分别测试：
   - 只用 LightRAG 检索
   - LightRAG 检索 + CrossEncoder rerank
3. 统计前 `K=3/5` 时的命中情况，比较 rerank 是否带来提升

示例表格：

| 评估项 | 数据集规模 | Recall@3 | Recall@5 | MRR | 备注 |
| --- | --- | --- | --- | --- | --- |
| LightRAG baseline | 50 questions | 待补充 | 待补充 | 待补充 | 未启用 rerank |
| LightRAG + rerank | 50 questions | 待补充 | 待补充 | 待补充 | CrossEncoder 重排 |

### 2. 路由效果评估

适用于 Router Agent 的专家分流能力。

建议关注这些指标：

- `Routing Accuracy`：问题是否被路由到正确专家
- `Over-routing Rate`：是否出现不必要的多专家调用
- `Tool-free Accuracy`：无需工具的问题是否能直接回答，而不是误触发工具

建议把问题分成三类：

- 知识库问题
- 本地代码库 / 文件问题
- 联网搜索问题

每类各准备若干样本，人工标注期望路由目标，再结合 `--trace` 输出统计 Router 实际选择是否正确。

示例表格：

| 问题类型 | 样本数 | 目标专家 | 路由准确率 | 平均专家调用数 |
| --- | --- | --- | --- | --- |
| 知识库问答 | 20 | Knowledge Specialist | 待补充 | 待补充 |
| 本地文件问答 | 20 | Filesystem Specialist | 待补充 | 待补充 |
| 联网问答 | 20 | Web Specialist | 待补充 | 待补充 |

### 3. 响应性能评估

适用于端到端问答延迟测量。

建议关注这些指标：

- `P50 / P95 Latency`
- 首 token 时间或首条有效输出时间
- 平均总响应时长
- 不同模式下的耗时拆分：
  - 无工具直接回答
  - 知识库检索回答
  - 检索 + rerank 回答
  - Web 搜索回答

建议测试方法：

1. 固定模型服务与硬件环境
2. 对同一组问题连续运行多轮
3. 分别记录总耗时、是否调用 rerank、是否触发 MCP
4. 统计 P50 / P95，而不是只看单次结果

示例表格：

| 场景 | 样本数 | P50 延迟 | P95 延迟 | 平均延迟 | 备注 |
| --- | --- | --- | --- | --- | --- |
| Router 直接回答 | 20 | 待补充 | 待补充 | 待补充 | 无工具 |
| Knowledge 检索 | 20 | 待补充 | 待补充 | 待补充 | 仅 LightRAG |
| Knowledge 检索 + rerank | 20 | 待补充 | 待补充 | 待补充 | 包含 CrossEncoder |
| Web 搜索 | 20 | 待补充 | 待补充 | 待补充 | 受网络波动影响较大 |

### 4. 资源消耗评估

适用于展示项目的工程可落地性。

建议关注这些指标：

- 索引构建耗时
- 文档入库吞吐量，例如 `pages/s` 或 `files/min`
- OCR 模式下的额外耗时
- reranker 模型加载耗时
- GPU / CPU / 内存占用峰值

如果你要把这个项目写进简历，资源维度的数字即使不多，也会比纯功能描述更有说服力。

示例表格：

| 任务 | 数据规模 | 总耗时 | 吞吐量 | 资源备注 |
| --- | --- | --- | --- | --- |
| 文本文件入库 | 待补充 | 待补充 | 待补充 | CPU / 内存待补充 |
| PDF 直抽入库 | 待补充 | 待补充 | 待补充 | 不含 OCR |
| 扫描版 PDF OCR 入库 | 待补充 | 待补充 | 待补充 | OCR 开销较高 |
| reranker 首次加载 | 1 次 | 待补充 | 不适用 | 模型冷启动 |

### 5. 推荐的评测集组织方式

可以在仓库后续补充一个 `data/eval/` 目录，按如下方式组织：

```text
data/
└─ eval/
   ├─ retrieval_eval.jsonl
   ├─ routing_eval.jsonl
   └─ latency_eval.jsonl
```

每条样本建议包含：

- `question`
- `expected_answer`
- `expected_source`
- `expected_specialist`
- `difficulty`

这样后续无论是写脚本自动评测，还是手工对照 trace 做分析，都会更方便。

### 6. 简历中如何写“性能评估”

如果后面补齐了真实数据，简历里建议写成这种风格：

- 构建多 Agent 中文知识库问答系统评测集，从检索命中率、路由准确率、端到端延迟三个维度评估系统效果
- 对比 LightRAG baseline 与 CrossEncoder rerank 方案，量化召回排序质量提升
- 基于 trace 链路统计专家路由正确率与平均工具调用次数，优化不必要的多工具开销

如果暂时还没有完整数据，也可以先在 README 中说明：

> 当前仓库已明确评估维度与指标设计，后续将补充检索命中率、路由准确率、响应延迟和资源消耗等定量结果。

## 简历/面试可强调的点

如果你把这个项目写进简历，建议重点突出这些关键词：

- 多 Agent 路由编排
- RAG 检索与 rerank
- MCP 工具集成
- PDF 解析与 OCR 回退
- 可观测性与 trace 调试
- 面向中文知识库问答的工程化落地

一个更像简历风格的描述示例：

> 基于 LangGraph、LightRAG 与 MCP 搭建多 Agent 中文知识库问答系统，设计 Router 对知识库检索、本地代码分析与联网搜索三类专家能力进行分流；实现 PDF 直抽文本与 OCR 回退、CrossEncoder 检索重排以及 trace 调试链路，提升复杂文档问答的可用性与可解释性。

## 当前局限

- README 之外还没有完整的部署文档或效果评测报告
- 测试目前以单元测试为主，缺少更完整的端到端验证
- Web 搜索使用轻量 HTML 页面解析方案，适合 Demo 与本地实验，不是生产级搜索方案
- Filesystem MCP 主要面向工作区内文本检索，不适合作为通用文件平台

## 后续可扩展方向

- 增加端到端评测集与检索质量指标
- 支持更多文档格式，如 `docx / html / csv`
- 为不同专家配置不同模型
- 增加会话记忆与用户级上下文隔离
- 提供 Web UI 或 API 服务层
- 增加更细粒度的权限控制与错误恢复机制

## License

如需开源发布，建议补充明确的 License 文件。
