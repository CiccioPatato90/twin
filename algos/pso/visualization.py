import glob
import matplotlib.pyplot as plt
import os

def parse_file(filename):
    swarm_x = []
    swarm_y = []
    simplex_x = []
    simplex_y = []
    
    all_x = []
    all_y = []

    with open(filename, 'r') as f:
        lines = f.readlines()
    
    evaluation_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        value = None
        source = None
        
        if line.startswith("SWARM:"):
            try:
                value = float(line.split(":")[1].strip())
                source = "SWARM"
            except ValueError:
                continue
        elif line.startswith("SIMPLEX:"):
            try:
                value = float(line.split(":")[1].strip())
                source = "SIMPLEX"
            except ValueError:
                continue
        else:
            # Try to parse bare number if any, though likely noise based on file sample
            try:
                 # Check if it's just a number like "0"
                 val = float(line)
                 # We don't know the source, maybe just skip or treat as generic
                 # Skipping to be safe and adhere to "color based on source"
                 continue 
            except ValueError:
                continue

        evaluation_count += 1
        
        all_x.append(evaluation_count)
        all_y.append(value)

        if source == "SWARM":
            swarm_x.append(evaluation_count)
            swarm_y.append(value)
        elif source == "SIMPLEX":
            simplex_x.append(evaluation_count)
            simplex_y.append(value)
            
    return all_x, all_y, swarm_x, swarm_y, simplex_x, simplex_y

def main():
    files = glob.glob("*.txt")
    # Filter for relevant files just in case
    files = [f for f in files if f in ['evaluations.txt', 'pso.txt', 'nm.txt']]
    
    if not files:
        print("No .txt files found (evaluations.txt, pso.txt, nm.txt).")
        return

    files.sort()

    for filename in files:
        # Create a new figure for each file with a normal screen aspect ratio
        plt.figure(figsize=(10, 6))
        
        all_x, all_y, swarm_x, swarm_y, simplex_x, simplex_y = parse_file(filename)
        
        # Plot the trajectory line first (thin, neutral color)
        plt.plot(all_x, all_y, color='lightgray', linestyle='-', linewidth=1, label='Trajectory')
        
        # Plot Swarm points
        if swarm_x:
            plt.scatter(swarm_x, swarm_y, color='blue', s=15, label='Swarm (PSO)')
        
        # Plot Simplex points
        if simplex_x:
            plt.scatter(simplex_x, simplex_y, color='red', s=15, label='Simplex (NM)')
            
        count_text = f"Counts - SWARM: {len(swarm_x)}, SIMPLEX: {len(simplex_x)}"
        plt.title(f"Optimization Progress - {filename}\n{count_text}")
        plt.xlabel("Evaluations")
        plt.ylabel("Cost / Fitness (Lower is better)")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Construct output filename: e.g., plot_pso.png
        base_name = os.path.splitext(filename)[0]
        output_filename = f"plot_{base_name}.png"
        
        plt.tight_layout()
        plt.savefig(output_filename)
        print(f"Saved {output_filename}")
        plt.close() # Close the figure to free memory

if __name__ == "__main__":
    main()