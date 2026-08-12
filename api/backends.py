import os

import torch
import torch.nn as nn


INPUT_DIM = 1024
HIDDEN_DIM = 4096
OUTPUT_DIM = 1000


class InferenceMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM)
        )

    def forward(self, x):
        return self.net(x)


class PyTorchBackend:
    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = (
            InferenceMLP()
            .eval()
            .to(self.device)
        )

        if self.device == "cuda":
            self.model = self.model.half()

    @property
    def name(self):
        return "pytorch"

    def infer(self, inputs):

        dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        x = torch.tensor(
            inputs,
            dtype=dtype,
            device=self.device
        )

        if self.device == "cuda":
            torch.cuda.synchronize()

        with torch.inference_mode():
            output = self.model(x)

        if self.device == "cuda":
            torch.cuda.synchronize()

        return (
            output
            .float()
            .cpu()
            .tolist()
        )


class TensorRTBackend:
    def __init__(self):

        if not torch.cuda.is_available():
            raise RuntimeError(
                "TensorRT backend requires an NVIDIA GPU"
            )

        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT is not installed"
            ) from exc

        self.trt = trt
        self.device = "cuda"

        engine_path = os.getenv(
            "ACCELSERVE_ENGINE_PATH",
            "inference/accelserve_mlp_fp16_native_io.engine"
        )

        logger = trt.Logger(
            trt.Logger.WARNING
        )

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(logger)

        engine = runtime.deserialize_cuda_engine(
            engine_bytes
        )

        if engine is None:
            raise RuntimeError(
                "Failed to load TensorRT engine"
            )

        context = engine.create_execution_context()

        if context is None:
            raise RuntimeError(
                "Failed to create TensorRT context"
            )

        self.runtime = runtime
        self.engine = engine
        self.context = context

        self.stream = torch.cuda.Stream()

    @property
    def name(self):
        return "tensorrt"

    def infer(self, inputs):

        batch_size = len(inputs)

        input_tensor = torch.tensor(
            inputs,
            dtype=torch.float16,
            device="cuda"
        )

        output_tensor = torch.empty(
            batch_size,
            OUTPUT_DIM,
            dtype=torch.float16,
            device="cuda"
        )

        input_name = "input"
        output_name = "output"

        engine_shape = tuple(
            self.engine.get_tensor_shape(
                input_name
            )
        )

        if (
            len(engine_shape) > 0
            and engine_shape[0] != -1
            and engine_shape[0] != batch_size
        ):
            raise RuntimeError(
                f"TensorRT engine expects batch "
                f"{engine_shape[0]}, got {batch_size}"
            )

        if engine_shape[0] == -1:
            self.context.set_input_shape(
                input_name,
                (
                    batch_size,
                    INPUT_DIM
                )
            )

        self.context.set_tensor_address(
            input_name,
            input_tensor.data_ptr()
        )

        self.context.set_tensor_address(
            output_name,
            output_tensor.data_ptr()
        )

        with torch.cuda.stream(
            self.stream
        ):

            ok = self.context.execute_async_v3(
                stream_handle=
                    self.stream.cuda_stream
            )

        if not ok:
            raise RuntimeError(
                "TensorRT inference failed"
            )

        self.stream.synchronize()

        return (
            output_tensor
            .float()
            .cpu()
            .tolist()
        )


def create_backend():

    backend_name = os.getenv(
        "ACCELSERVE_BACKEND",
        "pytorch"
    ).lower()

    if backend_name == "pytorch":
        return PyTorchBackend()

    if backend_name == "tensorrt":
        return TensorRTBackend()

    raise RuntimeError(
        f"Unknown backend: {backend_name}"
    )