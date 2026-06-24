"""A streaming Ollama client that surfaces the model's *thinking* separately.

The stock ``OllamaChatCompletionClient.create_stream`` only reads
``chunk.message.content`` and ignores Ollama's dedicated ``thinking`` field, so
reasoning never reaches the UI. This subclass:

- sends ``think=True`` to Ollama (only when the model supports it), and
- pushes each thinking delta straight onto the active SSE queue (the same
  ``research_sink`` ContextVar the web-research tool uses), tagged with the agent
  name, while still **yielding** answer-content deltas so AutoGen emits them as
  ``ModelClientStreamingChunkEvent`` (the real-time message channel).

NB: this re-implements the base ``create_stream`` loop (autogen_ext 0.4.x). It is
intentionally kept close to the original so it is easy to diff on upgrades — see the
fallback noted in the project plan if the internal API shifts.
"""
import asyncio
import json
import os
import sys
from typing import Any, AsyncGenerator, List, Literal, Mapping, Optional, Sequence, Union

from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import CreateResult, LLMMessage, RequestUsage
from autogen_core.tools import Tool, ToolSchema
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.ollama._ollama_client import (
    _add_usage,
    normalize_name,
    normalize_stop_reason,
)

from backend.research import research_sink


class StreamingOllamaChatCompletionClient(OllamaChatCompletionClient):
    """Ollama client that streams answer tokens and emits thinking out-of-band."""

    def __init__(self, *, agent_name: str = "", enable_thinking: bool = False, **kwargs: Any) -> None:
        # These are ours, not Ollama config — strip before the base parses kwargs.
        self._agent_name = agent_name
        self._enable_thinking = enable_thinking
        super().__init__(**kwargs)

    def _emit_thinking(self, delta: str) -> None:
        """Push a thinking delta onto the active SSE queue, if one is set."""
        if not delta:
            return
        queue = research_sink.get(None)
        if queue is not None:
            queue.put_nowait({"type": "thinking_delta", "agent": self._agent_name, "content": delta})

    def _emit_usage(self, prompt_tokens: int, completion_tokens: int, thinking_tokens: int) -> None:
        """Push one combined per-model-call usage event onto the active SSE queue, if one is set.

        prompt/completion/thinking are all derived from the SAME Ollama eval (see ``create_stream``):
        `completion_tokens` is `eval_count` (thinking INCLUDED) and `thinking_tokens` is the clamped
        estimate of its thinking share (always ≤ completion). Emitting all three together — per call,
        not per final message — keeps them consistent so answer-only (completion − thinking) can't go
        negative, and records completion even on empty-answer (thinking-only) turns. Skipped only when
        there's nothing to record.
        """
        if prompt_tokens <= 0 and completion_tokens <= 0:
            return
        queue = research_sink.get(None)
        if queue is not None:
            queue.put_nowait(
                {
                    "type": "usage",
                    "agent": self._agent_name,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "thinking_tokens": thinking_tokens,
                }
            )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | Literal["auto", "required", "none"] = "auto",
        json_output: Optional[bool | type] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        create_params = self._process_create_args(
            messages, tools, tool_choice, json_output, extra_create_args
        )
        # Opt-in vision diagnostics: confirm whether image bytes actually reach the
        # wire (and with what `think` flag) for this inference. Off unless VISION_DEBUG set.
        if os.environ.get("VISION_DEBUG"):
            img_count = sum(len(getattr(m, "images", None) or []) for m in create_params.messages)
            print(
                f"[VISION_DEBUG] agent={self._agent_name!r} vision={self.model_info['vision']} "
                f"think={bool(self._enable_thinking)} images_on_wire={img_count}",
                file=sys.stderr, flush=True,
            )
        stream_future = asyncio.ensure_future(
            self._client.chat(  # type: ignore[arg-type]
                messages=create_params.messages,
                tools=create_params.tools if len(create_params.tools) > 0 else None,
                stream=True,
                think=self._enable_thinking or None,
                format=create_params.format,
                **create_params.create_args,
            )
        )
        if cancellation_token is not None:
            cancellation_token.link_future(stream_future)
        stream = await stream_future

        chunk = None
        stop_reason = None
        content_chunks: List[str] = []
        full_tool_calls: List[FunctionCall] = []
        completion_tokens = 0
        thinking_chars = 0  # length of streamed reasoning text, for the thinking-token estimate
        while True:
            try:
                chunk_future = asyncio.ensure_future(anext(stream))
                if cancellation_token is not None:
                    cancellation_token.link_future(chunk_future)
                chunk = await chunk_future

                stop_reason = chunk.done_reason if chunk.done and stop_reason is None else stop_reason

                # Reasoning → out-of-band SSE queue (separate "thinking" channel).
                thinking = getattr(chunk.message, "thinking", None)
                if thinking:
                    thinking_chars += len(thinking)
                    self._emit_thinking(thinking)

                # Answer content → yielded so AutoGen streams it as a chunk event.
                if chunk.message.content is not None:
                    content_chunks.append(chunk.message.content)
                    if len(chunk.message.content) > 0:
                        yield chunk.message.content

                if chunk.message.tool_calls is not None:
                    full_tool_calls.extend(
                        FunctionCall(
                            id=str(self._tool_id),
                            arguments=json.dumps(x.function.arguments),
                            name=normalize_name(x.function.name),
                        )
                        for x in chunk.message.tool_calls
                    )
            except StopAsyncIteration:
                break

        prompt_tokens = chunk.prompt_eval_count if (chunk and chunk.prompt_eval_count) else 0

        # eval_count covers every generated token of this response — answer text, reasoning, AND
        # tool-call arguments — so count it regardless of which the model produced this call.
        completion_tokens = chunk.eval_count if (chunk and chunk.eval_count) else 0

        content: Union[str, List[FunctionCall]]
        thought: Optional[str] = None
        if len(content_chunks) > 0 and len(full_tool_calls) > 0:
            content = full_tool_calls
            thought = "".join(content_chunks)
        elif len(content_chunks) >= 1:
            content = "".join(content_chunks)
        else:
            content = full_tool_calls

        # Ollama's `eval_count` (completion_tokens) bundles thinking + answer with no split, so
        # estimate the thinking share by character proportion of the streamed reasoning vs answer
        # text. Exact (0) for non-thinking turns; approximate otherwise. Clamped to the total.
        answer_chars = len("".join(content_chunks))
        if completion_tokens > 0 and thinking_chars > 0:
            thinking_tokens = round(completion_tokens * thinking_chars / (thinking_chars + answer_chars))
            thinking_tokens = max(0, min(thinking_tokens, completion_tokens))
        else:
            thinking_tokens = 0
        self._emit_usage(prompt_tokens, completion_tokens, thinking_tokens)

        usage = RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        result = CreateResult(
            finish_reason=normalize_stop_reason(stop_reason),
            content=content,
            usage=usage,
            cached=False,
            logprobs=None,
            thought=thought,
        )
        self._total_usage = _add_usage(self._total_usage, usage)
        self._actual_usage = _add_usage(self._actual_usage, usage)
        yield result
