# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Proxy implementation for OpenAI Responses API that forwards requests
to a remote OpenAI-compatible chat/completions endpoint.

This allows clients that only support the Responses API to work with
LLM providers that only support the chat/completions API.
"""

import time
import uuid
from collections.abc import AsyncGenerator
from http import HTTPStatus

from fastapi import Request
from openai import AsyncOpenAI

from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import (
    ErrorResponse,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseInputOutputMessage,
    ResponsesRequest,
    ResponsesResponse,
    ResponseUsage,
    StreamingResponsesResponse,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class OpenAIServingResponsesProxy:
    """
    Proxy for Responses API that converts requests to chat/completions format
    and forwards them to a remote OpenAI-compatible service.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        request_logger: RequestLogger | None = None,
    ) -> None:
        """
        Initialize the proxy.

        Args:
            base_url: Base URL of the remote OpenAI-compatible service
            api_key: API key for the remote service
            request_logger: Optional request logger
        """
        self.base_url = base_url
        self.api_key = api_key
        self.request_logger = request_logger
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        logger.info(
            "Initialized Responses API proxy with base_url=%s",
            base_url,
        )

    def _convert_responses_to_chat_request(
        self, request: ResponsesRequest
    ) -> dict:
        """
        Convert a Responses API request to a chat/completions request.

        Args:
            request: The Responses API request

        Returns:
            A dictionary containing the chat/completions request parameters
        """
        # Convert input messages to chat format
        messages = []
        
        # Handle input field which can be a list of messages or a single message
        input_data = request.input
        if isinstance(input_data, list):
            for msg in input_data:
                if isinstance(msg, dict):
                    # Standard message format
                    messages.append(msg)
                elif hasattr(msg, "model_dump"):
                    # Pydantic model
                    messages.append(msg.model_dump(exclude_none=True))
        elif isinstance(input_data, dict):
            messages.append(input_data)
        elif hasattr(input_data, "model_dump"):
            messages.append(input_data.model_dump(exclude_none=True))

        # Build chat completion request
        chat_request = {
            "model": request.model,
            "messages": messages,
            "stream": request.stream if request.stream is not None else False,
        }

        # Map optional parameters
        if request.temperature is not None:
            chat_request["temperature"] = request.temperature
        if request.top_p is not None:
            chat_request["top_p"] = request.top_p
        if request.max_tokens is not None:
            chat_request["max_tokens"] = request.max_tokens
        if request.presence_penalty is not None:
            chat_request["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            chat_request["frequency_penalty"] = request.frequency_penalty
        if request.stop is not None:
            chat_request["stop"] = request.stop

        logger.debug("Converted Responses request to chat request: %s", chat_request)
        return chat_request

    def _convert_chat_to_responses_response(
        self,
        chat_response: dict,
        response_id: str,
        created_at: int,
    ) -> ResponsesResponse:
        """
        Convert a chat/completions response to a Responses API response.

        Args:
            chat_response: The chat/completions response
            response_id: The response ID
            created_at: Creation timestamp

        Returns:
            A ResponsesResponse object
        """
        # Extract the assistant message
        choice = chat_response.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")

        # Build output message
        output_message = ResponseInputOutputMessage(
            role="assistant",
            content=content,
        )

        # Build usage information
        usage_data = chat_response.get("usage", {})
        usage = ResponseUsage(
            total_tokens=usage_data.get("total_tokens", 0),
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )

        return ResponsesResponse(
            id=response_id,
            object="response",
            created=created_at,
            model=chat_response.get("model", "unknown"),
            output=[output_message],
            status="completed",
            usage=usage,
        )

    async def _convert_chat_stream_to_responses_stream(
        self,
        chat_stream: AsyncGenerator,
        response_id: str,
        created_at: int,
        model: str,
    ) -> AsyncGenerator[StreamingResponsesResponse, None]:
        """
        Convert a streaming chat/completions response to Responses API streaming format.

        Args:
            chat_stream: The streaming chat/completions response
            response_id: The response ID
            created_at: Creation timestamp
            model: The model name

        Yields:
            StreamingResponsesResponse events
        """
        # Send initial created event
        yield ResponseCreatedEvent(
            type="response.created",
            response=ResponsesResponse(
                id=response_id,
                object="response",
                created=created_at,
                model=model,
                output=[],
                status="in_progress",
                usage=None,
            ),
        )

        # Send in-progress event
        yield ResponseInProgressEvent(type="response.in_progress")

        # Accumulate content and tokens for final response
        accumulated_content = ""
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0

        # Process stream chunks
        async for chunk in chat_stream:
            if hasattr(chunk, "model_dump"):
                chunk_dict = chunk.model_dump()
            else:
                chunk_dict = chunk

            choices = chunk_dict.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content_delta = delta.get("content", "")
                
                if content_delta:
                    accumulated_content += content_delta
                    # Send text delta event (using ResponseInProgressEvent as placeholder)
                    # In a full implementation, you would send appropriate delta events

            # Check for usage information (usually in the last chunk)
            usage = chunk_dict.get("usage")
            if usage:
                total_tokens = usage.get("total_tokens", 0)
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

        # Send completed event with final response
        output_message = ResponseInputOutputMessage(
            role="assistant",
            content=accumulated_content,
        )

        usage_obj = ResponseUsage(
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ) if total_tokens > 0 else None

        final_response = ResponsesResponse(
            id=response_id,
            object="response",
            created=created_at,
            model=model,
            output=[output_message],
            status="completed",
            usage=usage_obj,
        )

        yield ResponseCompletedEvent(
            type="response.completed",
            response=final_response,
        )

    async def create_responses(
        self,
        request: ResponsesRequest,
        raw_request: Request,
    ) -> ResponsesResponse | AsyncGenerator[StreamingResponsesResponse, None] | ErrorResponse:
        """
        Handle a Responses API request by proxying to chat/completions.

        Args:
            request: The Responses API request
            raw_request: The raw FastAPI request

        Returns:
            Either a ResponsesResponse, a streaming generator, or an ErrorResponse
        """
        try:
            # Generate response ID and timestamp
            response_id = f"resp_{uuid.uuid4().hex}"
            created_at = int(time.time())

            # Convert request
            chat_request = self._convert_responses_to_chat_request(request)

            # Log the request if logger is available
            if self.request_logger is not None:
                self.request_logger.log_request(
                    request_id=response_id,
                    request=request,
                )

            # Make request to remote service
            if chat_request.get("stream", False):
                # Streaming request
                stream = await self.client.chat.completions.create(**chat_request)
                return self._convert_chat_stream_to_responses_stream(
                    stream,
                    response_id,
                    created_at,
                    request.model,
                )
            else:
                # Non-streaming request
                chat_response = await self.client.chat.completions.create(**chat_request)
                
                # Convert response to dict if needed
                if hasattr(chat_response, "model_dump"):
                    chat_response_dict = chat_response.model_dump()
                else:
                    chat_response_dict = chat_response

                response = self._convert_chat_to_responses_response(
                    chat_response_dict,
                    response_id,
                    created_at,
                )

                # Log the response if logger is available
                if self.request_logger is not None:
                    self.request_logger.log_response(
                        request_id=response_id,
                        response=response,
                    )

                return response

        except Exception as e:
            logger.error("Error in Responses API proxy: %s", str(e), exc_info=True)
            return ErrorResponse(
                message=f"Proxy error: {str(e)}",
                type="proxy_error",
                code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            )

    async def retrieve_responses(
        self,
        response_id: str,
        *,
        starting_after: int | None = None,
        stream: bool | None = False,
    ) -> ResponsesResponse | AsyncGenerator[StreamingResponsesResponse, None] | ErrorResponse:
        """
        Retrieve a response by ID. Not supported in proxy mode.
        """
        logger.warning("retrieve_responses called but not supported in proxy mode")
        return ErrorResponse(
            message="Response retrieval is not supported in proxy mode",
            type="not_supported",
            code=HTTPStatus.NOT_IMPLEMENTED.value,
        )

    async def cancel_responses(
        self,
        response_id: str,
    ) -> ResponsesResponse | ErrorResponse:
        """
        Cancel a response by ID. Not supported in proxy mode.
        """
        logger.warning("cancel_responses called but not supported in proxy mode")
        return ErrorResponse(
            message="Response cancellation is not supported in proxy mode",
            type="not_supported",
            code=HTTPStatus.NOT_IMPLEMENTED.value,
        )
