from pathlib import Path

# Auto detect USD files that can be in two forms
# - standalone USD file in "../models"
# - composite USD files in a directory in "../models"
MODELS_DIR = Path(__file__).parent / "models"
USD_PATHS = {}
SCENE_PATHS = {}

for robot_dir in MODELS_DIR.iterdir():
    robot_path = {}
    for path in (robot_dir / "usd").iterdir():
        if path.is_file():
            robot_path[path.name.removesuffix(".usd")] = str(path)
        else:
            usd_file = next(file for file in path.iterdir() if file.name.endswith(".usd"))
            robot_path[usd_file.name.removesuffix(".usd")] = str(usd_file)
    USD_PATHS[robot_dir.name] = robot_path

    SCENE_PATHS[robot_dir.name] = robot_dir / "scene" / "scene.xml"
