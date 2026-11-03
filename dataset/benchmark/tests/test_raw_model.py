"""Exercise native streaming and stop handling without paid model requests."""

import json
import httpx2
import unittest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.messages import ModelRequest, UserPromptPart
from dataset.benchmark.src.raw_model import RawCompletionModel
from dataset.benchmark.src.serialization import CODE, END_CODE


class RawModelTests(unittest.IsolatedAsyncioTestCase):
    async def completion(self, events):
        def handle(request):
            self.payload = json.loads(request.content)
            self.assertEqual(request.url.path, "/completion")
            return httpx2.Response(200, text="\n\n".join("data: " + json.dumps(event) for event in events))

        async with RawCompletionModel("test", base_url="http://testserver/v1", api_key="local-only", settings={"temperature": 0}) as model:
            await model.client.aclose()
            model.client = httpx2.AsyncClient(base_url="http://testserver", transport=httpx2.MockTransport(handle))
            messages = [ModelRequest(parts=[UserPromptPart("Print 3")], instructions="DO NOT SEND SYSTEM")]
            return await model.request(messages, None, ModelRequestParameters())

    async def test_native_request_has_no_chat_template_or_system(self):
        response = await self.completion(
            [
                {"content": "print(3)", "stop": False},
                {"content": "", "stop": True, "stopping_word": END_CODE, "tokens_evaluated": 12, "tokens_predicted": 8},
            ]
        )
        self.assertEqual(response.text, "print(3)")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.usage.input_tokens, 12)
        self.assertNotIn("DO NOT SEND SYSTEM", self.payload["prompt"])
        self.assertNotIn("messages", self.payload)
        self.assertTrue(self.payload["prompt"].endswith(CODE + "\n"))

    async def test_eos_without_end_code_is_not_accepted(self):
        response = await self.completion([{"content": "print(3)", "stop": True, "stopping_word": "", "stopped_eos": True}])
        self.assertNotEqual(response.finish_reason, "stop")

    async def test_stream_without_terminal_event_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "without a final result"):
            await self.completion([{"content": "print(3)", "stop": False}])
