# 项目接入说明

## 相关文件

- `src/langgraph_lightrag_demo/graph.py`：组装 LightRAG、本地工具和 MCP 工具给 agent
- `src/langgraph_lightrag_demo/mcp_servers/web_search_server.py`：网页搜索 MCP
- `src/langgraph_lightrag_demo/mcp_servers/filesystem_server.py`：本地文件系统 MCP
- `src/langgraph_lightrag_demo/lightrag_client.py`：LightRAG 查询入口

## 当前行为

- agent 启动时会加载项目内 skills 正文
- agent 会根据问题类型决定是用 LightRAG、MCP 网页还是 MCP 文件系统
- 如果检索结果冗长或噪声多，再进入 rerank

## 实践建议

- 当工具调用过多时，优先调整这个 skill 的路由规则
- 当某类问题总是走错工具时，先补充这里的边界定义
