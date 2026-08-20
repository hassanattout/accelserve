import json
import os
import threading
from pathlib import Path

import torch

from inference.model import (
    INPUT_DIM,
    OUTPUT_DIM,
    create_model,
    model_fingerprint,
)


class PyTorchBackend:
    def __init__(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = create_model().eval()
        self.model_fingerprint = model_fingerprint(self.model)
        self.model = self.model.to(self.device)

        if self.device == "cuda":
            self.model = self.model.half()

        self.maximum_batch_size = 256

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

        engine_path = Path(os.getenv(
            "ACCELSERVE_ENGINE_PATH",
            "inference/accelserve_mlp_fp16.engine"
        ))
        manifest_path = Path(f"{engine_path}.json")

        if not engine_path.is_file():
            raise RuntimeError("TensorRT engine file was not found")
        if not manifest_path.is_file():
            raise RuntimeError("TensorRT engine manifest was not found")

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("TensorRT engine manifest is invalid") from exc

        required_manifest = {
            "model_fingerprint",
            "input_dimension",
            "output_dimension",
            "minimum_batch_size",
            "maximum_batch_size",
        }
        if not required_manifest.issubset(manifest):
            raise RuntimeError("TensorRT engine manifest is incomplete")
        if manifest["input_dimension"] != INPUT_DIM:
            raise RuntimeError("TensorRT engine input dimension is incompatible")
        if manifest["output_dimension"] != OUTPUT_DIM:
            raise RuntimeError("TensorRT engine output dimension is incompatible")

        expected_fingerprint = os.getenv(
            "ACCELSERVE_EXPECTED_MODEL_FINGERPRINT"
        )
        if (
            expected_fingerprint
            and manifest["model_fingerprint"] != expected_fingerprint
        ):
            raise RuntimeError("TensorRT engine model identity is incompatible")

        self.model_fingerprint = manifest["model_fingerprint"]
        self.minimum_batch_size = int(manifest["minimum_batch_size"])
        self.maximum_batch_size = int(manifest["maximum_batch_size"])

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
        self._lock = threading.Lock()

    @property
    def name(self):
        return "tensorrt"

    def infer(self, inputs):

        batch_size = len(inputs)

        if not self.minimum_batch_size <= batch_size <= self.maximum_batch_size:
            raise RuntimeError(
                "TensorRT batch size is outside the engine profile"
            )

        with self._lock:
            return self._infer_locked(inputs, batch_size)

    def _infer_locked(self, inputs, batch_size):

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
            shape_ok = self.context.set_input_shape(
                input_name,
                (
                    batch_size,
                    INPUT_DIM
                )
            )
            if not shape_ok:
                raise RuntimeError("TensorRT rejected the input shape")

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
