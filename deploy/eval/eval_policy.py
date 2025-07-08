import json
from pathlib import Path


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
        data = json.load(file)

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


if __name__ == "__main__":
    current_dir = Path(__file__).parent
    json_files = [
        "stable_policy.json",
        "golden_policy.json",
        "unitree_policy.json",
    ]
    for json_file in json_files:
        json_file_path = current_dir / json_file
        if json_file_path.exists():
            extracted_data = extract_data(str(json_file_path))

            print(f"{json_file.replace('.json', '')}:")
            for joint_name, rate in extracted_data["avg_action_rate"].items():
                print(f"  - {joint_name}: {rate:.3f}")

            # print(
            #     f"{json_file.replace('.json', '')}: avg:{extracted_data['average_force']:.0f} max:{extracted_data['max_force']:.0f}"
            # )
            # print(
            #     f"{json_file.replace('.json', '')}: avg:{extracted_data['avg_action_rate']} max:{extracted_data['max_action_rate']}"
            # )
        else:
            print(f"File not found: {json_file_path}")
