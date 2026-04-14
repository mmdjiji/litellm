import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices, Usage


# Create a simple test
class MockCompletionStream:
    def __init__(self):
        self.responses = [
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(content="Hello"), index=0, finish_reason=None
                    )
                ],
            ),
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(content=" World"), index=0, finish_reason=None
                    )
                ],
            ),
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(content=""), index=0, finish_reason="stop"
                    )
                ],
            ),
        ]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.responses):
            raise StopIteration
        response = self.responses[self.index]
        self.index += 1
        return response


def test_anthropic_sse_wrapper_format():
    """Test that the SSE wrapper produces proper event and data formatting"""
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(), model="claude-3"
    )

    # Get the first chunk from the SSE wrapper
    first_chunk = next(wrapper.anthropic_sse_wrapper())

    # Verify it's bytes
    assert isinstance(first_chunk, bytes)

    # Decode and check format
    chunk_str = first_chunk.decode("utf-8")

    # Should have event line and data line
    lines = chunk_str.split("\n")
    assert len(lines) >= 3  # event line, data line, empty line (+ possibly more)
    assert lines[0].startswith("event: ")
    assert lines[1].startswith("data: ")
    assert lines[2] == ""  # Empty line to end the SSE chunk


def test_anthropic_sse_wrapper_event_types():
    """Test that different chunk types produce correct event types"""
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(), model="claude-3"
    )

    chunks = []
    for chunk in wrapper.anthropic_sse_wrapper():
        chunks.append(chunk.decode("utf-8"))
        if len(chunks) >= 3:  # Get first few chunks
            break

    # First chunk should be message_start
    assert "event: message_start" in chunks[0]
    assert '"type": "message_start"' in chunks[0]

    # Second chunk should be content_block_start
    assert "event: content_block_start" in chunks[1]
    assert '"type": "content_block_start"' in chunks[1]

    # Third chunk should be content_block_delta
    assert "event: content_block_delta" in chunks[2]
    assert '"type": "content_block_delta"' in chunks[2]


@pytest.mark.asyncio
async def test_async_anthropic_sse_wrapper():
    """Test the async version of the SSE wrapper"""

    class AsyncMockCompletionStream:
        def __init__(self):
            self.responses = [
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content="Hello"), index=0, finish_reason=None
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content=" World"), index=0, finish_reason=None
                        )
                    ],
                ),
            ]
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.responses):
                raise StopAsyncIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    wrapper = AnthropicStreamWrapper(
        completion_stream=AsyncMockCompletionStream(), model="claude-3"
    )

    # Get the first chunk from the async SSE wrapper
    first_chunk = None
    sse_stream = wrapper.async_anthropic_sse_wrapper()
    try:
        first_chunk = await anext(sse_stream)
    finally:
        await sse_stream.aclose()

    # Verify it's bytes and properly formatted
    assert first_chunk is not None
    assert isinstance(first_chunk, bytes)

    chunk_str = first_chunk.decode("utf-8")
    assert "event: message_start" in chunk_str
    assert '"type": "message_start"' in chunk_str


def test_anthropic_stream_wrapper_preserves_trigger_delta_on_block_transition():
    """The trigger delta on text -> thinking -> text transitions should not be dropped."""

    class MockTransitionStream:
        def __init__(self):
            self.responses = [
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content="Lead text"),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(
                                reasoning_content="Thinking through the answer",
                                thinking_blocks=[
                                    {
                                        "type": "thinking",
                                        "thinking": "Thinking through the answer",
                                        "signature": None,
                                    }
                                ],
                                content="",
                                role="assistant",
                            ),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content="Final answer"),
                            index=0,
                            finish_reason=None,
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content=""),
                            index=0,
                            finish_reason="stop",
                        )
                    ],
                    usage=Usage(prompt_tokens=12, completion_tokens=7, total_tokens=19),
                ),
            ]
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.responses):
                raise StopIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockTransitionStream(), model="claude-compatible"
    )

    chunks = list(wrapper)

    assert [chunk["type"] for chunk in chunks] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert chunks[1]["content_block"]["type"] == "text"
    assert chunks[2]["delta"] == {"type": "text_delta", "text": "Lead text"}
    assert chunks[4]["content_block"]["type"] == "thinking"
    assert chunks[5]["delta"] == {
        "type": "thinking_delta",
        "thinking": "Thinking through the answer",
    }
    assert chunks[7]["content_block"]["type"] == "text"
    assert chunks[8]["delta"] == {"type": "text_delta", "text": "Final answer"}
