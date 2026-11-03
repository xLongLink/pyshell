"""Use native llama.cpp completion streaming without a chat template."""

import json
import httpx2
from datetime import UTC, datetime
from contextlib import asynccontextmanager
from dataclasses import field, dataclass
from collections.abc import AsyncIterator
from pydantic_ai.usage import RequestUsage
from pydantic_ai.models import Model, StreamedResponse
from pydantic_ai.messages import TextPart, ModelRequest, UserPromptPart
from dataset.benchmark.src.serialization import CODE, TASK, FORMAT, CONSOLE, END_CODE, serialize


def render_history(messages):
    turns = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if not isinstance(part, UserPromptPart) or not isinstance(part.content, str):
                    raise ValueError("Raw worker supports text task and console messages only")
                turns.append({"role": "user", "content": part.content})
        else:
            for part in message.parts:
                if not isinstance(part, TextPart):
                    raise ValueError("Raw worker supports Python text responses only")
                turns.append({"role": "assistant", "content": part.content})
    return serialize(turns)


@dataclass
class RawStream(StreamedResponse):
    lines: AsyncIterator[str]
    name: str
    prompt: str
    url: str
    _timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    async def _get_event_iterator(self):
        complete = False
        async for line in self.lines:
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"Native completion error: {event['error']}")
            content = event.get("content", "")
            if content:
                for part in self._parts_manager.handle_text_delta(vendor_part_id="code", content=content):
                    yield part
            if event.get("stop"):
                complete = True
                self._usage = RequestUsage(input_tokens=event.get("tokens_evaluated", 0), output_tokens=event.get("tokens_predicted", 0))
                ended = event.get("stopping_word") == END_CODE
                self.finish_reason = "stop" if ended and not event.get("truncated") else "length"
                self.provider_details = {
                    "format": FORMAT,
                    "end_marker_seen": ended,
                    "stopping_word": event.get("stopping_word"),
                    "raw_prompt": self.prompt,
                }
        if not complete:
            raise RuntimeError("Native completion stream ended without a final result")

    @property
    def model_name(self):
        return self.name

    @property
    def provider_name(self):
        return "llama.cpp-raw"

    @property
    def provider_url(self):
        return self.url

    @property
    def timestamp(self):
        return self._timestamp


class RawCompletionModel(Model):
    """Adapt the worker's history to task/code/console native completions."""

    def __init__(self, name, *, base_url, api_key, settings):
        super().__init__(settings=settings)
        self.name = name
        self.client = httpx2.AsyncClient(
            base_url=base_url.rstrip("/").removesuffix("/v1"), headers={"Authorization": f"Bearer {api_key}"}, timeout=120, trust_env=False
        )

    @property
    def model_name(self):
        return self.name

    @property
    def system(self):
        return "llama.cpp-raw"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        await self.client.aclose()

    async def request(self, messages, model_settings, model_request_parameters):
        async with self.request_stream(messages, model_settings, model_request_parameters) as stream:
            async for _ in stream:
                pass
            return stream.get()

    @asynccontextmanager
    async def request_stream(self, messages, model_settings, model_request_parameters, run_context=None):
        settings = {**(self.settings or {}), **(model_settings or {})}
        prompt = render_history(messages)
        payload = {
            "prompt": prompt,
            "stream": True,
            "n_predict": settings.get("max_tokens", 2048),
            "temperature": settings.get("temperature", 0),
            "top_p": settings.get("top_p", 0.8),
            "stop": [END_CODE, TASK, CONSOLE, CODE],
            "cache_prompt": True,
        }
        async with self.client.stream("POST", "/completion", json=payload) as response:
            response.raise_for_status()
            yield RawStream(
                model_request_parameters, lines=response.aiter_lines(), name=self.name, prompt=prompt, url=str(self.client.base_url)
            )
