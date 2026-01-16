# vLLM OpenAI Chat 接口 Reasoning 长度控制参数分析报告

## 概述

本报告详细分析 vLLM 项目在调用 OpenAI Chat Completion 接口时，可以使用哪些参数来控制思考（reasoning）的长度和行为。

## 主要控制参数

### 1. `reasoning_effort` - 思考强度参数

**位置**: `vllm/entrypoints/openai/protocol.py` (第 561 行)

```python
reasoning_effort: Literal["low", "medium", "high"] | None = None
```

**功能描述**:
- 控制模型思考过程的深度和详细程度
- 可选值: `"low"`, `"medium"`, `"high"`
- 默认值: `None` (不指定)

**工作原理**:
- 该参数会被传递到 `harmony_utils.py` 中的 `get_system_message()` 函数
- 通过 `REASONING_EFFORT` 字典映射到 `ReasoningEffort` 枚举值:
  - `"high"` → `ReasoningEffort.HIGH` (高强度思考)
  - `"medium"` → `ReasoningEffort.MEDIUM` (中等强度思考)
  - `"low"` → `ReasoningEffort.LOW` (低强度思考)
- 最终通过 `SystemContent.with_reasoning_effort()` 方法注入到系统消息中

**实现位置**:
- 定义: `vllm/entrypoints/openai/protocol.py:561`
- 映射: `vllm/entrypoints/openai/parser/harmony_utils.py:53-57`
- 使用: `vllm/entrypoints/openai/serving_chat.py:1862`

**使用示例**:
```python
# Chat Completion API
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "解释量子计算"}],
    reasoning_effort="high"  # 使用高强度思考
)

# Responses API
response = await client.responses.create(
    model=model,
    input="解释量子计算",
    reasoning={"effort": "high"}  # 在 reasoning 对象中指定
)
```

### 2. `max_completion_tokens` - 最大生成 Token 数

**位置**: `vllm/entrypoints/openai/protocol.py` (第 543 行)

```python
max_completion_tokens: int | None = None
```

**功能描述**:
- 限制模型生成的最大 token 数量（包括思考内容和最终回答）
- 替代已弃用的 `max_tokens` 参数
- 可以间接控制思考内容的长度

**工作原理**:
- 该参数限制了总的输出长度
- 由于思考内容（reasoning）和最终回答（content）都计入这个限制，较小的值会迫使模型产生更简洁的思考
- 在 `serving_engine.py` 中被处理和验证

**使用示例**:
```python
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "9.11 和 9.8 哪个更大?"}],
    max_completion_tokens=100,  # 限制总输出为 100 tokens
    reasoning_effort="medium"
)
```

### 3. `include_reasoning` - 是否包含思考内容

**位置**: `vllm/entrypoints/openai/protocol.py` (第 562 行)

```python
include_reasoning: bool = True
```

**功能描述**:
- 控制响应中是否包含思考过程（reasoning）
- 默认值: `True`（包含思考内容）
- 设置为 `False` 时只返回最终答案，不返回思考过程

**工作原理**:
- 在流式和非流式响应中都生效
- 在 `serving_chat.py` 和 `serving_chat_stream_harmony.py` 中被处理
- 即使模型生成了思考内容，如果设置为 `False` 也不会返回给客户端

**使用示例**:
```python
# 包含思考内容（默认）
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "解决这个问题"}],
    include_reasoning=True
)
print(response.choices[0].message.reasoning)  # 可以访问思考内容

# 不包含思考内容
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "解决这个问题"}],
    include_reasoning=False
)
# response.choices[0].message.reasoning 将为空
```

## 服务器端配置

### `--reasoning-parser` - 指定推理解析器

**命令行参数**:
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --reasoning-parser deepseek_r1
```

**功能描述**:
- 指定用于解析模型输出中思考内容的解析器
- 必须配置才能启用 reasoning 功能
- 不同模型需要不同的解析器

**支持的解析器**:
- `deepseek_r1` - DeepSeek R1 系列
- `deepseek_v3` - DeepSeek V3.1
- `qwen3` - Qwen3 系列
- `granite` - IBM Granite 3.2
- `glm45` - GLM-4.5 系列
- `holo2` - Holo2 系列
- `hunyuan_a13b` - Hunyuan A13B 系列
- `ernie45` - ERNIE-4.5 系列
- `minimax_m2_append_think` - MiniMax-M2
- 以及其他支持推理的模型

### `--default-chat-template-kwargs` - 默认模板参数

**命令行参数**:
```bash
# 为 Qwen3 禁用思考模式
vllm serve Qwen/Qwen3-8B \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'

# 为 Granite 启用思考模式
vllm serve ibm-granite/granite-3.2-2b-instruct \
    --reasoning-parser granite \
    --default-chat-template-kwargs '{"thinking": true}'
```

**功能描述**:
- 为所有请求设置默认的聊天模板参数
- 可以在服务器级别控制思考模式的开关
- 请求级别的参数可以覆盖服务器默认设置

## 参数优先级和组合

### 优先级顺序

1. **请求级别参数** > **服务器默认配置**
2. `max_completion_tokens` 限制总输出长度
3. `reasoning_effort` 控制思考深度
4. `include_reasoning` 决定是否返回思考内容

### 推荐组合

**场景 1: 深度分析任务**
```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    reasoning_effort="high",           # 高强度思考
    max_completion_tokens=2000,        # 较大的输出限制
    include_reasoning=True             # 包含思考过程
)
```

**场景 2: 快速响应任务**
```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    reasoning_effort="low",            # 低强度思考
    max_completion_tokens=500,         # 较小的输出限制
    include_reasoning=False            # 不返回思考过程
)
```

**场景 3: 平衡的中等任务**
```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    reasoning_effort="medium",         # 中等强度思考
    max_completion_tokens=1000,        # 中等输出限制
    include_reasoning=True             # 包含思考过程
)
```

## 响应结构

### 非流式响应

```python
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "9.11 和 9.8 哪个更大?"}],
    reasoning_effort="medium"
)

# 访问思考内容
reasoning = response.choices[0].message.reasoning
# 访问最终答案
content = response.choices[0].message.content

print(f"思考过程: {reasoning}")
print(f"最终答案: {content}")
```

### 流式响应

```python
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "解释相对论"}],
    reasoning_effort="high",
    stream=True
)

for chunk in stream:
    # 检查是否有思考内容
    reasoning = getattr(chunk.choices[0].delta, "reasoning", None)
    # 检查是否有最终答案内容
    content = getattr(chunk.choices[0].delta, "content", None)
    
    if reasoning:
        print(f"思考: {reasoning}", end="", flush=True)
    if content:
        print(f"答案: {content}", end="", flush=True)
```

## 实现架构

### 核心模块

1. **协议定义** (`vllm/entrypoints/openai/protocol.py`)
   - 定义 `ChatCompletionRequest` 类
   - 包含所有 API 参数定义

2. **服务层** (`vllm/entrypoints/openai/serving_chat.py`)
   - 处理聊天完成请求
   - 调用 `get_system_message()` 注入 reasoning_effort

3. **Harmony 工具** (`vllm/entrypoints/openai/parser/harmony_utils.py`)
   - 提供 `get_system_message()` 函数
   - 实现 reasoning_effort 到 ReasoningEffort 枚举的映射

4. **推理解析器** (`vllm/reasoning/`)
   - 包含多种模型的推理解析器
   - 从模型输出中提取思考内容

### 数据流

```
客户端请求 (reasoning_effort="high")
    ↓
ChatCompletionRequest (协议层)
    ↓
ServingChat.create_chat_completion (服务层)
    ↓
get_system_message(reasoning_effort="high") (工具层)
    ↓
SystemContent.with_reasoning_effort(ReasoningEffort.HIGH)
    ↓
渲染为 Prompt Token IDs
    ↓
发送给模型生成
    ↓
ReasoningParser.extract_reasoning (解析层)
    ↓
返回带有 reasoning 字段的响应
```

## 注意事项

### 1. 模型兼容性
- 并非所有模型都支持推理功能
- 必须使用支持推理的模型并配置正确的 reasoning-parser
- 参考文档 `docs/features/reasoning_outputs.md` 查看支持的模型列表

### 2. Token 计数
- `max_completion_tokens` 限制包括思考内容和最终答案的总 token 数
- 使用高 reasoning_effort 可能导致思考内容占用大量 tokens
- 需要平衡思考深度和输出长度的需求

### 3. 性能考虑
- `reasoning_effort="high"` 会生成更长的思考内容，增加推理时间
- 对于延迟敏感的应用，建议使用 `"low"` 或 `"medium"`
- `include_reasoning=False` 可以减少响应大小，但不会减少推理时间

### 4. 结构化输出支持
- 大多数推理模型支持 JSON 和 regex 结构化输出
- 工具调用（tool calling）仅从 content 字段解析，不从 reasoning 解析
- 某些模型（如 DeepSeek-V3.1）在工具调用模式下不支持推理

## 相关文件

### 主要实现文件
- `vllm/entrypoints/openai/protocol.py` - API 协议定义
- `vllm/entrypoints/openai/serving_chat.py` - 聊天服务实现
- `vllm/entrypoints/openai/parser/harmony_utils.py` - Harmony 工具函数
- `vllm/entrypoints/openai/serving_chat_stream_harmony.py` - 流式响应处理

### 示例文件
- `examples/online_serving/openai_chat_completion_with_reasoning.py` - 基本推理示例
- `examples/online_serving/openai_chat_completion_with_reasoning_streaming.py` - 流式推理示例
- `examples/online_serving/openai_chat_completion_tool_calls_with_reasoning.py` - 工具调用与推理结合示例

### 测试文件
- `tests/entrypoints/openai/test_response_api_with_harmony.py` - reasoning_effort 测试
- `tests/entrypoints/openai/test_serving_chat_stream_harmony.py` - include_reasoning 测试

### 文档
- `docs/features/reasoning_outputs.md` - 推理输出功能完整文档

## 总结

vLLM 提供了三个主要参数来控制 OpenAI Chat 接口的思考（reasoning）长度和行为:

1. **`reasoning_effort`** (`"low"` | `"medium"` | `"high"`) - 直接控制思考深度
2. **`max_completion_tokens`** (整数) - 间接通过限制总输出长度来控制
3. **`include_reasoning`** (布尔值) - 控制是否在响应中包含思考内容

这些参数配合使用，可以灵活地控制模型的思考行为，适应不同的应用场景需求。对于需要深度分析的任务，使用 `reasoning_effort="high"` 和较大的 `max_completion_tokens`；对于快速响应场景，使用 `reasoning_effort="low"` 和较小的 token 限制。

---

**文档版本**: 1.0  
**生成日期**: 2026-01-16  
**基于仓库**: vLLM (x22x22/vllm)
