# 多模态微调规划

## 目标

当基础通用模型对产品故障图片、多轮客服语气或复合问题拆解不够稳定时，可引入多模态微调增强以下能力：

- 文本 + 图片联合意图识别
- 故障现象分类
- 多问题拆解与逐项回答
- 更自然的客服回复风格
- 更严格的 grounded response generation

## 推荐数据格式

### 样本结构

```json
{
  "messages": [
    {"role": "system", "content": "你是专业产品客服"},
    {"role": "user", "content": "这个指示灯一直闪是什么意思？", "images": ["sample_01.jpg"]},
    {"role": "assistant", "content": "根据手册，该闪烁表示设备处于充电中。"}
  ],
  "product": "相机",
  "intent": "troubleshooting",
  "sub_questions": ["这个指示灯一直闪是什么意思"],
  "evidence_ids": ["Camera_05"]
}
```

## 数据来源建议

- 已标注的客服问答记录
- `data/question_public.csv` 中的问题样本
- 手册片段与插图自动合成的指令数据
- 人工补充的故障图片问答数据

## 训练任务拆分

### 阶段 1：理解增强

- 输入：用户问题、用户图片、历史轮次
- 输出：意图标签、子问题拆解结果、是否需要检索

### 阶段 2：生成增强

- 输入：用户问题、检索证据、图片说明
- 输出：自然客服回复

### 阶段 3：对齐优化

- 使用高质量人工偏好数据，降低幻觉和答非所问

## 接入位置

当前项目建议替换或增强以下模块：

- `agent/multimodal/understanding.py`
- `agent/services/customer_support_agent.py`

## 评估指标

- 意图识别准确率
- 多问题拆解准确率
- 图文证据召回率
- 回复事实一致性
- 幻觉率
- 用户满意度
