import json
from pathlib import Path

from safety_checker import SafetyChecker
from tabulate import tabulate


def extract_contact_force(data):
    max_force = 0
    total_force = 0
    count = 0
    for entry in data:
        left_force = entry["foot_contact_forces"].get("left_ankle_roll_link", 0)
        right_force = entry["foot_contact_forces"].get("right_ankle_roll_link", 0)

        if left_force == 0 and right_force == 0:
            continue

        sum_force = left_force + right_force
        total_force += sum_force

        if sum_force > max_force:
            max_force = total_force

        count += 1

    average_force = total_force / count if count > 0 else 0
    return (max_force, average_force)


def extract_rate(data, rate_name):
    max_rate = {}
    total_rate = {}
    count = 0

    for entry in data:
        for joint, rate in entry[rate_name].items():
            abs_rate = abs(rate)

            if joint not in total_rate:
                total_rate[joint] = 0
            total_rate[joint] += abs_rate

            if joint not in max_rate:
                max_rate[joint] = 0
            if abs_rate > max_rate[joint]:
                max_rate[joint] = abs_rate

        count += 1

    for joint in total_rate:
        total_rate[joint] /= count

    return (max_rate, total_rate)


def extract_data(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)[1:]  # First entry contains the limits

        max_force, avg_force = extract_contact_force(data)
        max_action_rate, avg_action_rate = extract_rate(data, "action_rate")
        max_pos_rate, avg_pos_rate = extract_rate(data, "joint_pos_rate")

        extracted_data = {
            "max_force": max_force,
            "avg_force": avg_force,
            "max_action_rate": max_action_rate,
            "avg_action_rate": avg_action_rate,
            "max_pos_rate": max_pos_rate,
            "avg_pos_rate": avg_pos_rate,
        }

    return extracted_data


def check_position_limits(safety_file_path):
    with open(safety_file_path, "r") as file:
        safety_data = json.load(file)

    violation_counts = {}

    for entry in safety_data:
        if entry["check_type"] == "position":
            joint_name = entry["joint_name"]

            if joint_name in violation_counts:
                violation_counts[joint_name] += 1
            else:
                violation_counts[joint_name] = 1

    return violation_counts


def generate_safety_file(json_file):
    log_dir = json_file.parent

    safety_checker = SafetyChecker()
    safety_checker.check_safety(json_file)
    safety_checker.save_data(log_dir=log_dir)


def prepare_action_rate_table(experiment_names, data):
    avg_action_rate_data = []
    avg_action_rate_headers = ["Joint Name"] + experiment_names

    joint_names = data[experiment_names[0]]["Avg Action Rate"].keys()
    for joint_name in joint_names:
        row = [joint_name]
        for experiment in experiment_names:
            rate = data[experiment].get("Avg Action Rate", {}).get(joint_name, "N/A")
            row.append(rate)
        avg_action_rate_data.append(row)

    avg_action_rate_table = tabulate(avg_action_rate_data, headers=avg_action_rate_headers, tablefmt="grid")
    return avg_action_rate_table


def prepare_contant_force_table(experiment_names, data):
    # Prepare Avg Force and Max Force table data
    force_data = []
    force_headers = ["Metric"] + experiment_names

    for metric in ["Avg Force", "Max Force"]:
        row = [metric]
        for experiment in experiment_names:
            value = data[experiment].get(metric, "N/A")
            row.append(value)
        force_data.append(row)

    force_table = tabulate(force_data, headers=force_headers, tablefmt="grid")
    return force_table


def prepare_position_violations_table(experiment_names, data):
    # Collect all joint names
    all_joints = set()
    for experiment in experiment_names:
        if "Position Violations" in data[experiment]:
            all_joints.update(data[experiment]["Position Violations"].keys())

    # Prepare the table data
    position_violations_data = []
    position_violations_headers = ["Joint Name"] + experiment_names

    for joint_name in sorted(all_joints):
        row = [joint_name]
        for experiment in experiment_names:
            count = data[experiment].get("Position Violations", {}).get(joint_name, "N/A")
            row.append(count)
        position_violations_data.append(row)

    position_violations_table = tabulate(position_violations_data, headers=position_violations_headers, tablefmt="grid")
    return position_violations_table


if __name__ == "__main__":
    current_dir = Path(__file__).parent.parent / "./logs/h12_locomotion"
    experiment_names = [
        "croux_sim",
        "croux_sim_2",
        "unitree_sim",
        "golden_sim",
        "croux_real",
        "croux_real_2",
        "unitree_real",
        "golden_real",
    ]

    data = {experiment: {} for experiment in experiment_names}
    # Extract Data
    for experiment in experiment_names:
        metrics_file = current_dir / experiment / "metrics.json"
        if metrics_file.exists():
            generate_safety_file(metrics_file)
            extracted_data = extract_data(str(metrics_file))

            data[experiment]["Avg Action Rate"] = {
                joint_name: f"{rate:.3f}" for joint_name, rate in extracted_data["avg_action_rate"].items()
            }
            data[experiment]["Avg Force"] = f"{extracted_data['avg_force']:.0f}"
            data[experiment]["Max Force"] = f"{extracted_data['max_force']:.0f}"

            safety_file_path = current_dir / experiment / "safety_check.json"
            violations = check_position_limits(safety_file_path)
            data[experiment]["Position Violations"] = violations
        else:
            print(f"File not found: {metrics_file}")

    avg_action_rate_table = prepare_action_rate_table(experiment_names, data)
    force_table = prepare_contant_force_table(experiment_names, data)
    position_violations_table = prepare_position_violations_table(experiment_names, data)

    # Print Avg Action Rate table
    print("Avg Action Rate:")
    print(avg_action_rate_table)
    print("\n")

    # Print Avg Force and Max Force table
    print("Avg Force and Max Force:")
    print(force_table)
    print("\n")

    # Print Position Violations table
    print("Position Violations:")
    print(position_violations_table)
    print("\n")

    # Write to log.txt in the parent directory
    log_file_path = Path(__file__).parent / "log.txt"
    with open(log_file_path, "w") as log_file:
        log_file.write("Avg Action Rate Table:\n")
        log_file.write(avg_action_rate_table)
        log_file.write("\n\n")
        log_file.write("Avg Force and Max Force Table:\n")
        log_file.write(force_table)
        log_file.write("\n\n")
        log_file.write("Position Violations Table:\n")
        log_file.write(position_violations_table)

    print(f"Tables have been written to {log_file_path}")
