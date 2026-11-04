"""Exercise native streaming and stop handling without paid model requests."""

import json
import httpx2
import pytest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.messages import ModelRequest, UserPromptPart
from dataset.benchmark.src.raw_model import RawCompletionModel
from dataset.benchmark.src.serialization import CODE, TASK, FORMAT, CONSOLE, END_CODE


class TestRawModel:
    async def completion(self, events):
        def handle(request):
            self.payload = json.loads(request.content)
            assert request.url.path == "/completion"
            return httpx2.Response(200, text="\n\n".join("data: " + json.dumps(event) for event in events))

        async with RawCompletionModel("test", base_url="http://testserver/v1", api_key="local-only", settings={"temperature": 0}) as model:
            await model.client.aclose()
            model.client = httpx2.AsyncClient(base_url="http://testserver", transport=httpx2.MockTransport(handle))
            messages = [ModelRequest(parts=[UserPromptPart("Print 3")], instructions="DO NOT SEND SYSTEM")]
            return await model.request(messages, None, ModelRequestParameters())

    @pytest.mark.anyio
    async def test_native_request_has_no_chat_template_or_system(self):
        response = await self.completion(
            [
                {"content": "print(3)", "stop": False},
                {"content": "", "stop": True, "stopping_word": END_CODE, "tokens_evaluated": 12, "tokens_predicted": 8},
            ]
        )
        assert response.text == "print(3)"
        assert response.finish_reason == "stop"
        assert response.usage.input_tokens == 12
        assert response.usage.output_tokens == 8
        assert response.provider_details == {"format": FORMAT, "end_marker_seen": True, "stopping_word": END_CODE}
        assert self.payload["prompt"] == TASK + "\nPrint 3\n" + CODE + "\n"
        assert "DO NOT SEND SYSTEM" not in self.payload["prompt"]
        assert "messages" not in self.payload
        assert self.payload["stop"] == [END_CODE, TASK, CONSOLE, CODE]
        assert self.payload["prompt"].endswith(CODE + "\n")

    @pytest.mark.anyio
    async def test_eos_without_end_code_is_not_accepted(self):
        response = await self.completion([{"content": "print(3)", "stop": True, "stopping_word": "", "stopped_eos": True}])
        assert response.finish_reason != "stop"

    @pytest.mark.anyio
    async def test_stream_without_terminal_event_is_rejected(self):
        with pytest.raises(RuntimeError, match="without a final result"):
            await self.completion([{"content": "print(3)", "stop": False}])

    @pytest.mark.anyio
    async def test_boundary_metadata_and_finish_reasons_do_not_require_a_raw_prompt(self, subtests):
        for marker, truncated in ((END_CODE, False), (END_CODE, True), (TASK, False), (CONSOLE, False), (CODE, False)):
            with subtests.test(marker=marker, truncated=truncated):
                response = await self.completion([{"content": "print(3)", "stop": True, "stopping_word": marker, "truncated": truncated}])
                assert response.text == "print(3)"
                assert response.finish_reason == ("stop" if marker == END_CODE and (not truncated) else "length")
                assert response.provider_details == {"format": FORMAT, "end_marker_seen": marker == END_CODE, "stopping_word": marker}
