import yaml


def parse(file_path):
    with open(file_path, 'r') as f:
        return yaml.load(f, Loader=yaml.UnsafeLoader)


if __name__ == "__main__":
    yaml_data = parse("logs/clean_rl/h12_12dof_flat/2025-06-15_14-41-20/params/env.yaml")
    print("History length:", yaml_data["observations"]["policy"]["history_length"])
    
    # If the scale is null, it is considered equal to 1.
    print("Base angular velocity observation:", yaml_data["observations"]["policy"]["base_ang_vel"])
