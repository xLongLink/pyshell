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

                assert runtime.configure_device(device) == device

                assert not torch.backends.cuda.matmul.allow_tf32
                assert not torch.backends.cudnn.allow_tf32
                if device == "cuda":
                    torch.cuda.is_available.assert_called_once_with()
                else:
                    torch.cuda.is_available.assert_not_called()
                torch.set_num_threads.assert_not_called()
                torch.set_num_interop_threads.assert_not_called()

    @pytest.mark.parametrize(("available", "expected"), [(False, "cpu"), (True, "cuda")])
    def test_auto_selects_available_device(self, available, expected):
        with patch.object(runtime, "torch") as torch:
            torch.cuda.is_available.return_value = available
            assert runtime.configure_device("auto") == expected
            torch.cuda.is_available.assert_called_once_with()
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32

    def test_unknown_device_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported device"):
            runtime.configure_device("unknown")

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

    def test_thread_configuration_can_be_reused_for_sequential_models(self):
        with patch.object(runtime, "torch") as torch:
            torch.get_num_interop_threads.side_effect = [4, 1]
            runtime.configure_threads(3)
            runtime.configure_threads(3)
            assert torch.set_num_threads.call_args_list == [call(3), call(3)]
            torch.set_num_interop_threads.assert_called_once_with(1)

    def test_interop_errors_still_propagate(self):
        error = RuntimeError("interop threads already initialized")
        with patch.object(runtime, "torch") as torch:
            torch.get_num_interop_threads.return_value = 4
            torch.set_num_interop_threads.side_effect = error
            with pytest.raises(RuntimeError) as raised:
                runtime.configure_threads(3)
            assert raised.value is error
