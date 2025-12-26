# Responses API 代理模式

## 概述

vLLM 现在支持一种代理模式，允许你在不运行本地模型的情况下使用 Responses API (`/v1/responses`)。服务器充当代理角色：

1. 接收客户端的 Responses API 格式请求
2. 将请求转换为 chat/completions 格式
3. 转发到远程的 OpenAI 兼容服务
4. 将响应转换回 Responses API 格式

## 使用场景

这个功能在以下情况下很有用：
- 你的客户端应用程序只支持 Responses API
- 你的大模型供应商只支持 chat/completions API
- 你想要使用远程模型而不进行本地部署

## 快速开始

### 启动代理服务器

使用以下命令启动 vLLM 的 Responses API 代理模式：

```bash
python -m vllm.entrypoints.openai.api_server \
  --responses-proxy-mode \
  --responses-proxy-base-url <远程服务的Base URL> \
  --responses-proxy-api-key <远程服务的API Key> \
  --port 8000
```

### 命令行参数

- `--responses-proxy-mode`: 启用代理模式（必需）
- `--responses-proxy-base-url`: 远程 OpenAI 兼容服务的 Base URL（必需）
  - 示例: `https://api.openai.com/v1`
  - 示例: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `--responses-proxy-api-key`: 用于远程服务认证的 API Key（必需）

### 阿里云灵积（DashScope）示例

```bash
python -m vllm.entrypoints.openai.api_server \
  --responses-proxy-mode \
  --responses-proxy-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --responses-proxy-api-key sk-98e55d42763e4e2fa9253e35783aba08 \
  --port 8000
```

### OpenAI 示例

```bash
python -m vllm.entrypoints.openai.api_server \
  --responses-proxy-mode \
  --responses-proxy-base-url https://api.openai.com/v1 \
  --responses-proxy-api-key sk-your-openai-api-key \
  --port 8000
```

## 客户端使用

启动代理服务器后，像使用普通的 vLLM Responses API 端点一样使用它：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="empty"  # 代理会处理与远程服务的认证
)

# 使用远程服务支持的模型名称
response = client.responses.create(
    model="qwen-turbo",  # 对于 DashScope
    # model="gpt-4",     # 对于 OpenAI
    input=[
        {"role": "user", "content": "你好！请介绍一下你自己。"}
    ]
)

print(response.output_text)
```

### 流式输出示例

```python
stream = client.responses.create(
    model="qwen-turbo",
    input=[
        {"role": "user", "content": "从1数到5。"}
    ],
    stream=True
)

for event in stream:
    if hasattr(event, 'type'):
        print(f"事件: {event.type}")
    if hasattr(event, 'response') and event.response:
        if hasattr(event.response, 'output'):
            for msg in event.response.output:
                if hasattr(msg, 'content'):
                    for content in msg.content:
                        if hasattr(content, 'text'):
                            print(content.text)
```

### 完整的 Python 示例

查看 `examples/online_serving/openai_responses_proxy_example.py` 获取完整的使用示例。

## 参数映射

代理会自动在 Responses API 和 chat/completions 之间映射参数：

| Responses API | chat/completions | 说明 |
|---------------|------------------|------|
| `model` | `model` | 直接映射 |
| `input` | `messages` | 消息对象数组 |
| `temperature` | `temperature` | 可选 |
| `top_p` | `top_p` | 可选 |
| `max_tokens` | `max_tokens` | 可选 |
| `presence_penalty` | `presence_penalty` | 可选 |
| `frequency_penalty` | `frequency_penalty` | 可选 |
| `stop` | `stop` | 可选 |
| `stream` | `stream` | 是否流式输出 |

## 限制

在代理模式下运行时：

1. **无本地模型**: 服务器不会加载或运行任何本地模型
2. **有限的端点**: 只支持 `/v1/responses` 端点
3. **响应检索**: 不支持 `GET /v1/responses/{id}`（无状态代理）
4. **响应取消**: 不支持 `POST /v1/responses/{id}/cancel`
5. **模型列表**: `/v1/models` 端点不会返回有意义的结果

## 支持的远程服务

代理应该可以与任何支持 chat/completions 端点的 OpenAI 兼容服务配合使用，包括：

- OpenAI API
- Azure OpenAI Service
- 阿里云灵积（DashScope）
- Anthropic（带兼容层）
- 其他 OpenAI 兼容服务

请确保：
1. 使用正确的服务 Base URL
2. 提供有效的 API Key
3. 使用远程服务支持的模型名称

## 故障排除

### 连接错误

如果看到连接错误，请验证：
- Base URL 正确且可访问
- API Key 有效
- 你的网络允许出站 HTTPS 连接

### 无效模型错误

如果收到"模型未找到"错误：
- 检查模型名称是否被远程服务支持
- 不同供应商使用不同的模型名称（例如 `qwen-turbo` vs `gpt-4`）

### 认证错误

如果收到认证错误：
- 验证你的 API Key 正确
- 检查你的账户是否有权访问该模型
- 确保你的 API Key 没有过期

## 架构

```
客户端（Responses API）
       ↓
vLLM 代理服务器
       ↓ (转换为 chat/completions)
远程大模型服务
       ↓ (返回 chat/completions 响应)
vLLM 代理服务器
       ↓ (转换回 Responses API)
客户端（接收 Responses API 响应）
```

代理执行格式转换，但不修改模型的实际内容或行为。

## 测试示例

你可以使用提供的测试端点进行测试：

```bash
# 启动服务器
python -m vllm.entrypoints.openai.api_server \
  --responses-proxy-mode \
  --responses-proxy-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --responses-proxy-api-key sk-98e55d42763e4e2fa9253e35783aba08 \
  --port 8000

# 在另一个终端运行示例
python examples/online_serving/openai_responses_proxy_example.py
```

## 贡献

如果你在使用代理模式时遇到问题或有改进建议，请：
1. 查看现有的 GitHub Issues
2. 提交新的 Issue 并详细描述你的用例
3. 考虑贡献代码改进

## 许可证

Apache 2.0 - 详见项目根目录的 LICENSE 文件
