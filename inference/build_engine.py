import os

import torch

from inference.model import (
    INPUT_DIM,
    create_model,
)


BATCH_SIZE = 256

ONNX_PATH = os.getenv(
    "ACCELSERVE_ONNX_PATH",
    "inference/accelserve_mlp_fp16.onnx",
)

ENGINE_PATH = os.getenv(
    "ACCELSERVE_ENGINE_PATH",
    "inference/accelserve_mlp_fp16.engine",
)


def export_onnx():
    print("Creating deterministic PyTorch model...")

    model = (
        create_model()
        .eval()
        .half()
    )

    dummy_input = torch.randn(
        BATCH_SIZE,
        INPUT_DIM,
        dtype=torch.float16,
    )

    print(
        f"Exporting FP16 ONNX model to: "
        f"{ONNX_PATH}"
    )

    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
    )

    print("FP16 ONNX export complete.")


def build_tensorrt_engine():
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT is required to build the engine."
        ) from exc

    print(
        f"Building TensorRT engine from: "
        f"{ONNX_PATH}"
    )

    logger = trt.Logger(
        trt.Logger.WARNING
    )

    builder = trt.Builder(logger)

    network = builder.create_network(
        1 << int(
            trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
        )
    )

    parser = trt.OnnxParser(
        network,
        logger,
    )

    with open(
        ONNX_PATH,
        "rb",
    ) as model_file:
        if not parser.parse(
            model_file.read()
        ):
            print("ONNX parsing failed:")

            for i in range(
                parser.num_errors
            ):
                print(
                    parser.get_error(i)
                )

            raise RuntimeError(
                "Failed to parse ONNX model."
            )

    config = (
        builder.create_builder_config()
    )

    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        1 << 30,
    )

    print(
        "Compiling TensorRT engine..."
    )

    serialized_engine = (
        builder.build_serialized_network(
            network,
            config,
        )
    )

    if serialized_engine is None:
        raise RuntimeError(
            "TensorRT engine build failed."
        )

    with open(
        ENGINE_PATH,
        "wb",
    ) as engine_file:
        engine_file.write(
            serialized_engine
        )

    engine_size_mb = (
        os.path.getsize(
            ENGINE_PATH
        )
        / (1024 * 1024)
    )

    print(
        f"Engine saved to: "
        f"{ENGINE_PATH}"
    )

    print(
        f"Engine size: "
        f"{engine_size_mb:.2f} MB"
    )


if __name__ == "__main__":
    os.makedirs(
        "inference",
        exist_ok=True,
    )

    export_onnx()
    build_tensorrt_engine()

    print()
    print(
        "AccelServe TensorRT build complete."
    )
