from pathlib import Path

# Auto detect USD files that can be in two forms
# - standalone USD file in "../models"
# - composite USD files in a directory in "../models"
MODELS_DIR = Path(__file__).parent / "models"
USD_PATHS = {}
SCENE_PATHS = {}

for robot_dir in MODELS_DIR.iterdir():
    robot_paths = {}
    for path in (robot_dir / "usd").iterdir():
        if path.is_file():
            robot_paths[path.name.removesuffix(".usd")] = str(path)
        else:
            usd_file = next(file for file in path.iterdir() if file.name.endswith(".usd"))
            robot_paths[usd_file.name.removesuffix(".usd")] = str(usd_file)
    USD_PATHS[robot_dir.name] = robot_paths

    scene_paths = {}
    for path in (robot_dir / "scene").iterdir():
        if path.name.startswith("scene"):
            scene_paths[path.name.removeprefix("scene_").removesuffix(".xml")] = str(path)
    SCENE_PATHS[robot_dir.name] = scene_paths
