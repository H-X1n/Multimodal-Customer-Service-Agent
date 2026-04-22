# 接口定义

## 输入对象

### `UserInput`

字段：

- `text: str` 用户问题
- `image_paths: list[str]` 用户上传图片路径
- `session_id: str` 会话 ID
- `metadata: dict` 扩展字段

## 输出对象

### `AgentResponse`

字段：

- `answer: str` 最终回答
- `retrievals: list[RetrievalResult]` 每个子问题的检索结果
- `referenced_images: list[str]` 汇总后的相关图片路径
- `intent: IntentAnalysis | None` 意图分析结果

### `RetrievalResult`

字段：

- `question: str` 子问题
- `chunks: list[EvidenceChunk]` 命中的证据片段

### `EvidenceChunk`

字段：

- `question: str` 对应子问题
- `content: str` 证据内容
- `metadata: dict` 文档元数据
- `image_ids: list[str]` 关联图片 ID
- `image_paths: list[str]` 关联图片实际路径

## 核心服务

### `CustomerSupportAgent.answer(user_input: UserInput) -> AgentResponse`

说明：

- 智能体主入口
- 负责意图识别、对话管理、RAG 检索、回复生成

### `KnowledgeService.retrieve(query: str) -> list[EvidenceChunk]`

说明：

- 根据单个子问题召回手册证据
- 解析图片 ID 并返回图片路径

### `DialogueManager.plan_questions(...) -> list[QuestionTask]`

说明：

- 将复合问题拆分为多个可回答单元

## 前端/接口集成建议

如需接入 Web 或 API 层，建议返回如下 JSON 结构：

```json
{
  "answer": "最终回复文本",
  "images": ["绝对路径1", "绝对路径2"],
  "retrievals": [
    {
      "question": "子问题1",
      "chunks": [
        {
          "content": "证据片段",
          "image_ids": ["Camera_01"]
        }
      ]
    }
  ]
}
```
