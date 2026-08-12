import os
import torch
import torch.nn as nn

INPUT_DIM = 1024
HIDDEN_DIM = 4096
OUTPUT_DIM = 1000
BATCH_SIZE = 256

ONNX_PATH = os.getenv(
    "ACCELSERVE_ONNX_PATH",
    "inference/accelserve_mlp.onnx"
)

ENGINE_PATH = os.getenv(
    "ACCELSERVE_ENGINE_PATH",
    "inference/accelserve_mlp_fp16.engine"
)


class InferenceMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        )

    def forward(self, x):
        return self.net(x)


def export_onnx():
    print("Creating PyTorch model...")

    model = InferenceMLP().eval()

    dummy_input = torch.randn(
        BATCH_SIZE,
        INPUT_DIM,
        dtype=torch.float32,
    )

    print(f"Exporting ONNX model to: {ONNX_PATH}")

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )

    print("ONNX export complete.")


def build_tensorrt_engine():
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT is required to build the engine."
        ) from exc

    print(f"Building TensorRT engine from: {ONNX_PATH}")

    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger)

    network = builder.create_network(
        1 << int(
            trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
        )
    )

    parser = trt.OnnxParser(network, logger)

    with open(ONNX_PATH, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("ONNX parsing failed:")

            for i in range(parser.num_errors):
                print(parser.get_error(i))

            raise RuntimeError(
                "Failed to parse ONNX model."
            )

    config = builder.create_builder_config()

    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        1 << 30,
    )

    print("Compiling TensorRT engine...")

    serialized_engine = builder.build_serialized_network(
        network,
        config,
    )

    if serialized_engine is None:
        raise RuntimeError(
            "TensorRT engine build failed."
        )

    with open(ENGINE_PATH, "wb") as engine_file:
        engine_file.write(serialized_engine)

    engine_size_mb = (
        os.path.getsize(ENGINE_PATH)
        / (1024 * 1024)
    )

    print(f"Engine saved to: {ENGINE_PATH}")
    print(f"Engine size: {engine_size_mb:.2f} MB")


if __name__ == "__main__":
    os.makedirs("inference", exist_ok=True)

    export_onnx()
    build_tensorrt_engine()

    print()
    print("AccelServe TensorRT build complete.")