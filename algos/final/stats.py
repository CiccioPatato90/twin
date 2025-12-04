import numpy as np
import matplotlib.pyplot as plt
import re

def parse_results_file(filename):
    """
    Parse results.txt file and extract the three numpy arrays.
    Uses numpy's array string representation parser.
    """
    with open(filename, "r") as f:
        content = f.read()
    
    def extract_array(var_name, content):
        """Extract and parse numpy array from text file"""
        # Find the start of this variable's data
        start_pattern = rf'{var_name}:\s*\['
        start_match = re.search(start_pattern, content)
        
        if not start_match:
            return None
        
        # Find where this array ends (look for next variable or end of content)
        start_pos = start_match.end() - 1  # Include the opening bracket
        remaining = content[start_pos:]
        
        # Find the matching closing bracket by counting brackets
        bracket_count = 0
        end_pos = 0
        for i, char in enumerate(remaining):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    end_pos = i + 1
                    break
        
        if end_pos == 0:
            return None
        
        array_str = remaining[:end_pos]
        
        # Convert numpy array string format to Python list format
        # Step 1: Normalize whitespace
        array_str = ' '.join(array_str.split())
        
        # Step 2: Add commas between numbers (space-separated numbers)
        def add_commas_between_numbers(text):
            # Match: number space number -> number, number
            pattern = r'(\d+\.?\d*(?:[eE][+-]?\d+)?)\s+(\d+\.?\d*(?:[eE][+-]?\d+)?)'
            result = re.sub(pattern, r'\1, \2', text)
            return result
        
        # Apply multiple times to handle all number pairs
        for _ in range(20):  # Max iterations
            new_str = add_commas_between_numbers(array_str)
            if new_str == array_str:
                break
            array_str = new_str
        
        # Step 3: Add commas between list elements (] [ -> ], [)
        array_str = re.sub(r'\]\s+\[', '], [', array_str)
        
        # Step 4: Clean up bracket spacing
        array_str = re.sub(r'\[\s+', '[', array_str)
        array_str = re.sub(r'\s+\]', ']', array_str)
        array_str = re.sub(r'\s+', ' ', array_str)  # Final cleanup
        
        try:
            # Evaluate as Python list and convert to numpy array
            array_data = eval(array_str)
            result = np.array(array_data, dtype=float)
            return result
        except Exception as e:
            print(f"Error parsing {var_name}: {e}")
            print(f"First 500 chars of problematic string:\n{array_str[:500]}")
            return None
    
    print("Parsing results from file...")
    results_conn = extract_array("results_conn", content)
    results_coverage = extract_array("results_coverage", content)
    results_lq = extract_array("results_lq", content)
    
    return results_conn, results_coverage, results_lq

def create_condensed_plot(results_conn, results_coverage, results_lq, test_values):
    """
    Create a single condensed, color-coded plot with all three metrics.
    """
    metrics = {
        'Connectivity': results_conn,
        'Coverage': results_coverage,
        'Link Quality': results_lq
    }
    
    coefficient_names = ['Alpha', 'Beta', 'Gamma', 'Delta']
    coefficient_axes = [0, 1, 2, 3]  # Which axis to vary (i, j, k, l)
    
    # Create a single figure with 3 rows (metrics) x 4 columns (coefficients)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Metrics vs Coefficient Values (Condensed View)', fontsize=16, fontweight='bold', y=0.995)
    
    # Color scheme for each metric
    colors = {
        'Connectivity': '#1f77b4',      # Blue
        'Coverage': '#2ca02c',          # Green
        'Link Quality': '#ff7f0e'       # Orange
    }
    
    for metric_idx, (metric_name, metric_data) in enumerate(metrics.items()):
        for coeff_idx, (coeff_name, axis_idx) in enumerate(zip(coefficient_names, coefficient_axes)):
            ax = axes[metric_idx, coeff_idx]
            
            # Calculate mean for each value of this coefficient
            means = []
            for val_idx in range(len(test_values)):
                if axis_idx == 0:  # Vary alpha (i)
                    mean_val = np.mean(metric_data[val_idx, :, :, :])
                elif axis_idx == 1:  # Vary beta (j)
                    mean_val = np.mean(metric_data[:, val_idx, :, :])
                elif axis_idx == 2:  # Vary gamma (k)
                    mean_val = np.mean(metric_data[:, :, val_idx, :])
                else:  # Vary delta (l)
                    mean_val = np.mean(metric_data[:, :, :, val_idx])
                means.append(mean_val)
            
            # Plot with color coding
            ax.plot(test_values, means, marker='o', linewidth=2.5, markersize=6,
                   color=colors[metric_name], alpha=0.8, label=metric_name)
            ax.fill_between(test_values, means, alpha=0.2, color=colors[metric_name])
            
            # Labels and formatting
            if metric_idx == 0:  # Top row
                ax.set_title(f'{coeff_name}', fontsize=11, fontweight='bold', pad=5)
            if metric_idx == 2:  # Bottom row
                ax.set_xlabel(f'{coeff_name} Value', fontsize=10)
            if coeff_idx == 0:  # Left column
                ax.set_ylabel(f'{metric_name}\n(Mean)', fontsize=10, fontweight='bold')
            
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xticks(test_values[::max(1, len(test_values)//5)])  # Show fewer x-ticks if many values
            
            # Add subtle background color for each metric row
            ax.set_facecolor((0.98, 0.98, 0.98))
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    return fig

if __name__ == "__main__":
    # Parse results
    results_conn, results_coverage, results_lq = parse_results_file("results.txt")
    
    if results_conn is None or results_coverage is None or results_lq is None:
        print("Error: Could not parse all arrays from results.txt")
        exit(1)
    
    print(f"Successfully loaded arrays:")
    print(f"  Connectivity shape: {results_conn.shape}")
    print(f"  Coverage shape: {results_coverage.shape}")
    print(f"  Link Quality shape: {results_lq.shape}")
    
    # Determine test_values from array shape
    num_values = results_conn.shape[0]
    # Assuming the range is 0 to 100 based on the code
    test_values = np.linspace(0, 100, num_values).astype(int)
    print(f"  Test values: {test_values}")
    
    # Create condensed plot
    print("\nGenerating condensed plot...")
    fig = create_condensed_plot(results_conn, results_coverage, results_lq, test_values)
    
    # Save the plot
    output_file = 'metrics_condensed_plot.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved condensed plot to: {output_file}")
    
    plt.show()

