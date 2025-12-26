#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example: Using vLLM as a Responses API proxy

This example demonstrates how to use vLLM in proxy mode to convert
Responses API requests to chat/completions and forward them to a
remote OpenAI-compatible service.

Prerequisites:
1. Start the vLLM proxy server:
   python -m vllm.entrypoints.openai.api_server \
     --responses-proxy-mode \
     --responses-proxy-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
     --responses-proxy-api-key YOUR_API_KEY \
     --port 8000

2. Run this example:
   python openai_responses_proxy_example.py
"""

from openai import OpenAI


def main():
    """Run the example."""
    # Create OpenAI client pointing to the vLLM proxy
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="empty",  # Not needed for proxy, it uses its own key
    )

    print("=" * 60)
    print("vLLM Responses API Proxy Example")
    print("=" * 60)

    # Use a model name that your remote service supports
    # For Alibaba DashScope: qwen-turbo, qwen-plus, qwen-max, etc.
    # For OpenAI: gpt-4, gpt-3.5-turbo, etc.
    model = "qwen-turbo"

    print(f"\n1. Simple request with model: {model}")
    print("-" * 60)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": "Hello! Please introduce yourself in one sentence.",
            }
        ],
    )

    print(f"Response ID: {response.id}")
    print(f"Model: {response.model}")
    print(f"Status: {response.status}")
    print("\nOutput:")
    for item in response.output:
        if hasattr(item, "content"):
            for content_item in item.content:
                if hasattr(content_item, "text"):
                    print(f"  {content_item.text}")

    if response.usage:
        print(f"\nUsage:")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")

    print("\n" + "=" * 60)
    print("2. Streaming request")
    print("-" * 60)

    stream = client.responses.create(
        model=model,
        input=[{"role": "user", "content": "Count from 1 to 5."}],
        stream=True,
    )

    print("Receiving streaming events...")
    for event in stream:
        if hasattr(event, "type"):
            print(f"  Event: {event.type}")

        # Print final output from completed event
        if hasattr(event, "response") and event.response:
            resp = event.response
            if hasattr(resp, "status") and resp.status == "completed":
                print("\n  Final output:")
                if hasattr(resp, "output"):
                    for item in resp.output:
                        if hasattr(item, "content"):
                            for content_item in item.content:
                                if hasattr(content_item, "text"):
                                    print(f"    {content_item.text}")

    print("\n" + "=" * 60)
    print("3. Multi-turn conversation")
    print("-" * 60)

    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]

    # First turn
    response = client.responses.create(model=model, input=messages)

    print("User: What is the capital of France?")
    first_response = ""
    for item in response.output:
        if hasattr(item, "content"):
            for content_item in item.content:
                if hasattr(content_item, "text"):
                    first_response = content_item.text
                    print(f"Assistant: {first_response}")

    # Continue conversation
    messages.append({"role": "assistant", "content": first_response})
    messages.append({"role": "user", "content": "What is its population?"})

    response = client.responses.create(model=model, input=messages)

    print("User: What is its population?")
    for item in response.output:
        if hasattr(item, "content"):
            for content_item in item.content:
                if hasattr(content_item, "text"):
                    print(f"Assistant: {content_item.text}")

    print("\n" + "=" * 60)
    print("✅ All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
