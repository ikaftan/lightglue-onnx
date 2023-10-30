import argparse
from pathlib import Path
from typing import List

from onnx_runner import LightGlueRunner, load_image, rgb_to_grayscale, viz2d
from sev_cli.argparser import SevArgParser


def parse_args() -> argparse.Namespace:
    parser = SevArgParser(description="Match SuperPoint features with LightGlue.")

    parser.add_argument(
        "--lightglue-path",
        type=Path,
        help="Path to the LightGlue ONNX model.",
        required=True)

    parser.add_argument(
        "--extractor-type",
        type=str,
        help="Type of the feature extractor.",
        required=True,
        choices=["superpoint", "disk"])

    parser.add_argument(
        "--extractor-path",
        type=Path,
        help="Path to the feature extractor ONNX model.",
        default=None)

    parser.add_argument(
        "--img-size",
        type=int,
        nargs="+",
        help="Sample image size for ONNX tracing.",
        default=[540, 720])

    parser.add_argument("--kaki", action="store_true", help="Run LightGlue for kaki.")

    parser.add_argument("--trt", action="store_true", help="Use TensorRT.")

    return parser.parse_args()


def infer(lightglue_path: Path, extractor_type: str, extractor_path=None,
          img_size=[540, 720], kaki=True, trt=False):
    if isinstance
    if isinstance(img_size, List):
        if len(img_size) == 1:
            size0 = size1 = img_size[0]
        elif len(img_size) == 2:
            size0 = size1 = img_size
        elif len(img_size) == 4:
            size0, size1 = img_size[:2], img_size[2:]
        else:
            raise ValueError("Invalid img_size. Please provide 1, 2, or 4 integers.")
    else:
        size0 = size1 = img_size

    image0, scales0 = load_image(img0_path, resize=size0)
    image1, scales1 = load_image(img1_path, resize=size1)

    extractor_type = extractor_type.lower()
    if extractor_type == "superpoint":
        image0 = rgb_to_grayscale(image0)
        image1 = rgb_to_grayscale(image1)
    elif extractor_type == "disk":
        pass
    else:
        raise NotImplementedError(
            f"Unsupported feature extractor type: {extractor_type}."
        )

    # load ONNX models
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if trt:
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "weights/cache",
                },
            )
        ] + providers

    runner = LightGlueRunner(
        extractor_path=extractor_path,
        lightglue_path=lightglue_path,
        providers=providers,
    )

    # run inference
    m_kpts0, m_kpts1 = runner.run(image0, image1, scales0, scales1)

    return m_kpts0, m_kpts1


if __name__ == "__main__":
    args = parse_args()
    m_kpts0, m_kpts1 = infer(**vars(args))
    print(m_kpts0, m_kpts1)
