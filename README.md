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
