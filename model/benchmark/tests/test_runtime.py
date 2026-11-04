"""Check shared runtime policy without CUDA or model loading."""

import pytest
from unittest.mock import call, patch
from model.benchmark.src import runtime


class TestRuntime:
    def test_device_policy(self, subtests):
        for device, available in (("cpu", False), ("cpu", True), ("cuda", True)):
            with subtests.test(device=device, available=available), patch.object(runtime, "torch") as torch:
                torch.cuda.is_available.return_value = available
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

                runtime.configure_device(device)

                assert not torch.backends.cuda.matmul.allow_tf32
                assert not torch.backends.cudnn.allow_tf32
                if device == "cuda":
                    torch.cuda.is_available.assert_called_once_with()
                else:
                    torch.cuda.is_available.assert_not_called()
                torch.set_num_threads.assert_not_called()
                torch.set_num_interop_threads.assert_not_called()

    def test_unavailable_cuda_fails_before_changing_policy(self):
        with patch.object(runtime, "torch") as torch:
            torch.cuda.is_available.return_value = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
                runtime.configure_device("cuda")

            assert torch.backends.cuda.matmul.allow_tf32
            assert torch.backends.cudnn.allow_tf32
            torch.set_num_threads.assert_not_called()
            torch.set_num_interop_threads.assert_not_called()

    def test_threads_are_configured_on_every_call_and_interop_errors_propagate(self):
        error = RuntimeError("interop threads already initialized")
        with patch.object(runtime, "torch") as torch:
            torch.set_num_interop_threads.side_effect = [None, error]
            runtime.configure_threads(3)

            with pytest.raises(RuntimeError) as raised:
                runtime.configure_threads(7)

            assert raised.value is error
            assert torch.mock_calls == [
                call.set_num_threads(3),
                call.set_num_interop_threads(1),
                call.set_num_threads(7),
                call.set_num_interop_threads(1),
            ]
