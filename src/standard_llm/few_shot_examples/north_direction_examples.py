# ruff: noqa: E501
import os

from dotenv import load_dotenv

load_dotenv()

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_default_north_dir = os.path.join(_project_root, "examples", "north_direction")
north_dir_img_dir = os.getenv("NORTH_DIR_EXAMPLES_PATH", _default_north_dir)

# Expected JSON for north/scale detection
NORTH_DETECTED = {"NorthDirectionSymbol": "Detected", "ScaleIndicator": "Detected"}

# Few-shot north direction exemplars
NORTH_DIR_EXAMPLES = [
    {
        "image_path": os.path.join(north_dir_img_dir, "north_dir1.png"),
        "description": "Full compass rose having north direction symbol.",
        "expected_response": NORTH_DETECTED,
    },
    {
        "image_path": os.path.join(north_dir_img_dir, "north_dir2.png"),
        "description": "A valid Geographical North Direction symbol pointing upwards.",
        "expected_response": NORTH_DETECTED,
    },
    {
        "image_path": os.path.join(north_dir_img_dir, "north_dir3.png"),
        "description": "A valid Geographical Direction symbol pointing upwards.",
        "expected_response": NORTH_DETECTED,
    },
]
