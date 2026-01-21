import os

import matplotlib.pyplot as plt


def parse_file(filename):
    """
    Parses the log file to extract evaluation counts and fitness values.
    Returns lists of x (evaluations) and y (fitness).
    """
    all_x = []
    all_y = []

    if not os.path.exists(filename):
        return [], []

    with open(filename, "r") as f:
        lines = f.readlines()

    evaluation_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        value = None

        # Parse lines like "SWARM: 123.45" or "SIMPLEX: 123.45"
        if line.startswith("SWARM:") or line.startswith("SIMPLEX:"):
            try:
                value = float(line.split(":")[1].strip())
            except ValueError:
                continue
        else:
            # Try to parse bare number
            try:
                value = float(line)
            except ValueError:
                continue

        if value is not None:
            evaluation_count += 1
            all_x.append(evaluation_count)
            all_y.append(value)

    return all_x, all_y


def main():
    # Configuration
    file_configs = [
        {"filename": "rastring/hybrid.txt", "label": "Hybrid", "color": "green"},
        {"filename": "rastring/pso.txt", "label": "PSO", "color": "blue"},
        {"filename": "rastring/nm.txt", "label": "NM SIMPLEX", "color": "red"},
    ]

    # 1. Parse all files first and store data
    loaded_data = []

    for config in file_configs:
        fname = config["filename"]
        x, y = parse_file(fname)

        if x and y:
            config["data_x"] = x
            config["data_y"] = y
            loaded_data.append(config)
            print(f"Loaded {config['label']}: {len(x)} points")
        else:
            print(f"Warning: {fname} missing or empty.")

    if not loaded_data:
        print("No valid data found.")
        return

    # 2. Find the minimum length among all loaded datasets
    min_length = min(len(d["data_x"]) for d in loaded_data)
    print(f"\nTruncating all datasets to the minimum length found: {min_length}")

    # 3. Plot truncated data
    plt.figure(figsize=(12, 7))

    for config in loaded_data:
        # Slice the data to min_length
        x_trunc = config["data_x"][:min_length]
        y_trunc = config["data_y"][:min_length]

        plt.plot(
            x_trunc,
            y_trunc,
            label=f"{config['label']} (n={min_length})",
            color=config["color"],
            linewidth=1.5,
            alpha=0.8,
        )

    plt.title("Convergence Comparison (Truncated to shortest run)")
    plt.xlabel("Evaluations")
    plt.ylabel("Best Cost / Fitness (Lower is better)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    output_filename = "rastring/comparison_plot.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Saved combined graph to {output_filename}")
    plt.close()


if __name__ == "__main__":
    main()
