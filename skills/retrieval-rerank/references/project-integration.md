# 项目接入说明

## 相关文件

- `src/langgraph_lightrag_demo/reranker.py`：加载本地 `bge-reranker-large` 并给片段打分
- `src/langgraph_lightrag_demo/graph.py`：向 agent 暴露 `rerank_retrieved_context` 工具
- `src/langgraph_lightrag_demo/config.py`：保存 `RERANKER_MODEL_PATH`、`RERANKER_DEVICE` 和 `RERANKER_TOP_K`

## 当前行为

- reranker 通过 `RerankerService._get_model()` 懒加载
- 检索上下文会先拆成片段再打分
- 工具返回带分数的高排名片段
- agent 只会在检索之后决定是否重排，不会把重排当成第一步

## 实践建议

- 当检索结果很长或主题混杂时启用重排
- top-k 保持适中，在提升精度的同时不要丢掉关键证据
- 优先通过调整 agent 提示词来控制行为，再考虑增加更复杂的重排逻辑
