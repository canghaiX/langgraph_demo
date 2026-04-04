# 项目接入说明

## 相关文件

- `src/langgraph_lightrag_demo/lightrag_client.py`：创建并缓存 `LightRAG` 实例
- `src/langgraph_lightrag_demo/graph.py`：暴露 `search_knowledge_base` 工具，并组装 agent 提示词
- `src/langgraph_lightrag_demo/config.py`：保存 `LIGHTRAG_*`、embedding、chat 和 chunk 参数

## 当前行为

- `LightRAG` 在 `LightRAGService.get_rag()` 中懒加载初始化
- 文档入库通过 `rag.ainsert(...)` 完成
- 查询通过 `rag.aquery(..., only_need_context=True)` 获取上下文
- 分块参数由 `CHUNK_TOKEN_SIZE` 和 `CHUNK_OVERLAP_TOKEN_SIZE` 控制
- agent 会先判断是否需要检索，再调用工具

## 实践建议

- 如果检索过于频繁或过于保守，优先调整 `graph.py` 里的 agent 提示词
- 如果要改检索、embedding 或 chunk 参数，编辑 `config.py` 和 `.env.example`
- 修改 chunk 参数后，重新构建知识库
