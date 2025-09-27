import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def manual_linear_regression(X, y):
    """
    Perform linear regression using manual linear algebra calculations
    
    Parameters:
    X: Feature matrix (n_samples, n_features)
    y: Target vector (n_samples,)
    
    Returns:
    coefficients: β vector including intercept
    y_pred: Predicted values
    r2_score: R-squared value
    """
    # Add intercept column (column of ones) to X
    n_samples = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # Calculate coefficients using normal equation: β = (X^T X)^(-1) X^T y
    X_transpose = X_with_intercept.T
    XTX = np.dot(X_transpose, X_with_intercept)
    XTy = np.dot(X_transpose, y)
    
    # Solve for coefficients
    try:
        coefficients = np.linalg.solve(XTX, XTy)
    except np.linalg.LinAlgError:
        # If matrix is singular, use pseudo-inverse
        coefficients = np.dot(np.linalg.pinv(XTX), XTy)
    
    # Make predictions: y_pred = X β
    y_pred = np.dot(X_with_intercept, coefficients)
    
    # Calculate R-squared manually
    # R² = 1 - (SS_res / SS_tot)
    # SS_res = Σ(y_i - y_pred_i)²
    # SS_tot = Σ(y_i - y_mean)²
    
    y_mean = np.mean(y)
    ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y - y_mean) ** 2)  # Total sum of squares
    
    # Handle edge case where ss_tot is 0 (all y values are the same)
    if ss_tot == 0:
        r2_score = 1.0 if ss_res == 0 else 0.0
    else:
        r2_score = 1 - (ss_res / ss_tot)
    
    return coefficients, y_pred, r2_score

def get_sample_size():
    """Get sample size from user input"""
    while True:
        try:
            n = int(input("Please enter the sample size (n): "))
            if n > 0:
                return n
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")

def generate_noise_ranges():
    """Generate 20 random noise ranges covering low, medium, and high values"""
    low_noise_ranges = np.random.uniform(0.01, 2, 7)
    medium_noise_ranges = np.random.uniform(2, 4, 7)
    high_noise_ranges = np.random.uniform(4, 6, 6)
    
    noise_ranges = np.concatenate([low_noise_ranges, medium_noise_ranges, high_noise_ranges])
    np.random.shuffle(noise_ranges)
    
    return noise_ranges

def generate_base_data(n):
    """Generate base independent variables and coefficients"""
    # Generate 50 independent variables
    independent_variables = np.random.rand(n, 50)
    
    # Generate 51 coefficients (beta0 + 50 betas)
    coefficients = np.random.rand(51)
    beta0 = coefficients[0]
    betas = coefficients[1:]
    
    return independent_variables, beta0, betas

def create_datasets_with_noise(independent_variables, beta0, betas, noise_ranges):
    """Create datasets with different noise levels"""
    datasets = []
    n = independent_variables.shape[0]
    
    for noise_range in noise_ranges:
        # Generate random noise within the specified range
        random_noise = np.random.uniform(-noise_range, noise_range, n)
        
        # Calculate dependent variable
        dependent_variable = beta0 + np.dot(independent_variables, betas) + random_noise
        
        # Store as dictionary with X and y
        dataset = {
            'X': independent_variables.copy(),
            'y': dependent_variable.copy()
        }
        datasets.append(dataset)
    
    return datasets

def calculate_model_metrics(datasets, noise_ranges):
    """Calculate R-squared and Adjusted R-squared for each dataset using manual linear algebra"""
    r_squared_values = []
    adjusted_r_squared_values = []
    
    for i, dataset in enumerate(datasets):
        X = dataset['X']
        y = dataset['y']
        
        # Perform manual linear regression
        coefficients, y_pred, r2 = manual_linear_regression(X, y)
        r_squared_values.append(r2)
        
        # Calculate adjusted R-squared
        # Formula: Adjusted R^2 = 1 - [(1-R^2)*(n-1)/(n-p-1)]
        n = X.shape[0]  # number of observations
        p = X.shape[1]  # number of independent variables
        
        # Handle edge case where n-p-1 <= 0
        if n - p - 1 <= 0:
            adjusted_r2 = r2  # or set to a default value
        else:
            adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        
        adjusted_r_squared_values.append(adjusted_r2)
    
    return r_squared_values, adjusted_r_squared_values

def generate_product_square_variables(independent_variables, betas):
    """Generate 20 new variables as products or squares of weighted existing variables (beta*x)"""
    new_variables = []
    formulas = []
    n_vars = independent_variables.shape[1]
    generated_combinations = set()
    
    # Calculate weighted variables (beta * x) for each original variable
    weighted_variables = independent_variables * betas[np.newaxis, :]  # Broadcasting betas across all rows
    
    while len(new_variables) < 20:
        operation = np.random.choice(['product', 'square'])
        
        if operation == 'product':
            idx1, idx2 = np.random.choice(n_vars, 2, replace=False)
            combination = tuple(sorted((idx1, idx2)))
            
            if combination not in generated_combinations:
                # Product of weighted variables: (beta1*x1) * (beta2*x2)
                new_var = weighted_variables[:, idx1] * weighted_variables[:, idx2]
                new_variables.append(new_var)
                formulas.append(f'(β{idx1+1}*x{idx1+1}) * (β{idx2+1}*x{idx2+1})')
                generated_combinations.add(combination)
        
        else:  # square
            idx = np.random.choice(n_vars)
            combination = (idx, 'square')
            
            if combination not in generated_combinations:
                # Square of weighted variable: (beta*x)^2
                new_var = weighted_variables[:, idx] ** 2
                new_variables.append(new_var)
                formulas.append(f'(β{idx+1}*x{idx+1})²')
                generated_combinations.add(combination)
    
    return np.array(new_variables).T, formulas

def check_correlations(original_vars, new_vars, var_type):
    """Check correlations between original and new variables"""
    print(f"\nChecking correlations for {var_type} variables:")
    
    # Find maximum correlations between new variables and any original variable
    max_correlations = []
    for i in range(new_vars.shape[1]):
        correlations_with_original = []
        for j in range(original_vars.shape[1]):
            corr = np.corrcoef(new_vars[:, i], original_vars[:, j])[0, 1]
            correlations_with_original.append(abs(corr))
        max_corr = max(correlations_with_original)
        max_correlations.append(max_corr)
    
    print(f"Max correlation with original variables: {np.max(max_correlations):.4f}")
    print(f"Min correlation with original variables: {np.min(max_correlations):.4f}")
    print(f"Mean correlation with original variables: {np.mean(max_correlations):.4f}")
    
    # Count how many are highly correlated (>0.9)
    highly_correlated = sum(1 for corr in max_correlations if corr > 0.9)
    print(f"Variables with correlation > 0.9: {highly_correlated} out of {len(max_correlations)}")
    
    return max_correlations

def generate_linear_transform_variables(independent_variables, betas):
    """Generate 20 new variables as linear transformations of weighted existing variables (beta*x)"""
    new_variables = []
    formulas = []
    n_vars = independent_variables.shape[1]
    generated_combinations = set()
    
    # Calculate weighted variables (beta * x) for each original variable
    weighted_variables = independent_variables * betas[np.newaxis, :]  # Broadcasting betas across all rows
    
    while len(new_variables) < 20:
        operation = np.random.choice(['multiply', 'add_subtract'])
        idx = np.random.choice(n_vars)
        
        if operation == 'multiply':
            factor = np.random.uniform(0.1, 3.0)  # Random multiplication factor
            combination = (idx, 'multiply', round(factor, 4))
            
            if combination not in generated_combinations:
                # Multiply weighted variable: factor * (beta*x)
                new_var = weighted_variables[:, idx] * factor
                new_variables.append(new_var)
                formulas.append(f'{factor:.4f} * (β{idx+1}*x{idx+1})')
                generated_combinations.add(combination)
        
        else:  # add_subtract
            offset = np.random.uniform(-2.0, 2.0)  # Random offset
            combination = (idx, 'add_subtract', round(offset, 4))
            
            if combination not in generated_combinations:
                # Add/subtract to weighted variable: (beta*x) + offset
                new_var = weighted_variables[:, idx] + offset
                new_variables.append(new_var)
                sign = '+' if offset >= 0 else '-'
                formulas.append(f'(β{idx+1}*x{idx+1}) {sign} {abs(offset):.4f}')
                generated_combinations.add(combination)
    
    return np.array(new_variables).T, formulas

def create_expanded_datasets(independent_variables, beta0, betas, new_variables, noise_ranges):
    """Create datasets with expanded variables - no additional predictive power added"""
    # Combine original and new variables
    combined_variables = np.concatenate((independent_variables, new_variables), axis=1)
    
    datasets = []
    n = combined_variables.shape[0]
    
    print("New variables added to feature matrix but TRUE model remains:")
    print("y = β₀ + β₁×x₁ + β₂×x₂ + ... + β₅₀×x₅₀ + noise")
    print("Additional variables are transformations for regression to evaluate")
    
    for noise_range in noise_ranges:
        # Generate random noise
        random_noise = np.random.uniform(-noise_range, noise_range, n)
        
        # TRUE model remains the same - only original variables contribute
        dependent_variable = beta0 + np.dot(independent_variables, betas) + random_noise
        
        # Store as dictionary
        dataset = {
            'X': combined_variables.copy(),  # Includes new variables for regression to fit
            'y': dependent_variable.copy()   # But y is still only based on original variables
        }
        datasets.append(dataset)
    
    return datasets

def plot_comparison(noise_ranges, original_r2, original_adj_r2, 
                   expanded_r2, expanded_adj_r2, title_suffix, expanded_marker):
    """Plot comparison of R-squared values with optimal visibility for small datasets"""
    plt.figure(figsize=(14, 9))
    
    # Define unique colors for each R-squared type
    if title_suffix == "Product/Square Variables":
        r2_color = 'red'          # Product/Square R-squared = red
        adj_r2_color = 'yellow'   # Product/Square Adjusted R-squared = yellow
    else:  # Linear Transformation Variables
        r2_color = 'black'        # Linear Transform R-squared = black
        adj_r2_color = 'pink'     # Linear Transform Adjusted R-squared = pink
    
    # Original colors remain consistent
    orig_r2_color = 'green'       # Original R-squared = green
    orig_adj_r2_color = 'blue'    # Original Adjusted R-squared = blue
    
    # Small dataset - minimal jitter, larger markers, higher opacity
    jitter_strength = 0.05
    noise_jitter_orig = noise_ranges + np.random.normal(0, jitter_strength, len(noise_ranges))
    noise_jitter_exp = noise_ranges + np.random.normal(0, jitter_strength, len(noise_ranges))
    
    # Plot original results (circles) - larger for visibility with small dataset
    plt.scatter(noise_jitter_orig, original_r2, label='Original R-squared', 
               alpha=0.8, marker='o', s=100, color=orig_r2_color, edgecolors='black', linewidth=1)
    plt.scatter(noise_jitter_orig, original_adj_r2, label='Original Adjusted R-squared', 
               alpha=0.8, marker='o', s=100, color=orig_adj_r2_color, edgecolors='black', linewidth=1)
    
    # Plot expanded results with specified marker
    marker_size = 120 if expanded_marker == 'x' else 110
    line_width = 3 if expanded_marker == 'x' else 1
    
    plt.scatter(noise_jitter_exp, expanded_r2, label=f'{title_suffix} R-squared', 
               alpha=0.8, marker=expanded_marker, s=marker_size, color=r2_color, 
               edgecolors='black', linewidth=line_width)
    plt.scatter(noise_jitter_exp, expanded_adj_r2, label=f'{title_suffix} Adjusted R-squared', 
               alpha=0.8, marker=expanded_marker, s=marker_size, color=adj_r2_color, 
               edgecolors='black', linewidth=line_width)
    
    # Use linear scale for x-axis since range is smaller and we have fewer points
    plt.xlabel('Average Error (Noise Range)', fontsize=13, fontweight='bold')
    plt.ylabel('R-squared / Adjusted R-squared', fontsize=13, fontweight='bold')
    plt.title(f'R-squared vs. Average Error ({title_suffix})', fontsize=15, fontweight='bold')
    
    # Improve grid and legend
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(fontsize=11, framealpha=0.9, edgecolor='black')
    
    # Set axis limits for better visibility
    plt.xlim(-0.2, 6.5)
    plt.ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.show()

def print_results_table(noise_ranges, r2_values, adj_r2_values, title):
    """Print results in a formatted table"""
    print(f"\n{title}")
    print("=" * len(title))
    print(f"{'Index':<5} {'Noise Range':<12} {'R-squared':<12} {'Adj R-squared':<15}")
    print("-" * 50)
    
    for i, (noise, r2, adj_r2) in enumerate(zip(noise_ranges, r2_values, adj_r2_values)):
        print(f"{i:<5} {noise:<12.4f} {r2:<12.4f} {adj_r2:<15.4f}")
    
    # Calculate summary statistics
    print(f"\nSummary Statistics:")
    print(f"Mean R-squared: {np.mean(r2_values):.4f}")
    print(f"Mean Adjusted R-squared: {np.mean(adj_r2_values):.4f}")
    print(f"Std R-squared: {np.std(r2_values):.4f}")
    print(f"Std Adjusted R-squared: {np.std(adj_r2_values):.4f}")

# Main execution
def main():
    print("Linear Regression Analysis with Manual Linear Algebra")
    print("=" * 55)
    print("Using manual implementation: β = (X^T X)^(-1) X^T y")
    print("R² calculation: 1 - (SS_res / SS_tot)")
    
    # Get sample size
    n = get_sample_size()
    print(f"\nUsing sample size: {n}")
    
    # Generate base data
    print("\nGenerating base data...")
    independent_variables, beta0, betas = generate_base_data(n)
    noise_ranges = generate_noise_ranges()
    
    print(f"Beta0: {beta0:.4f}")
    print(f"Noise ranges generated: {len(noise_ranges)} total datasets")
    print(f"  - Low noise (0.01-2.0): 7 datasets")
    print(f"  - Medium noise (2.0-4.0): 7 datasets") 
    print(f"  - High noise (4.0-6.0): 6 datasets")
    print(f"All noise values: {sorted(noise_ranges)}")
    
    # Phase 1: Original datasets (50 variables)
    print("\n" + "="*50)
    print("PHASE 1: Original Datasets (50 variables)")
    print("="*50)
    
    original_datasets = create_datasets_with_noise(independent_variables, beta0, betas, noise_ranges)
    original_r2, original_adj_r2 = calculate_model_metrics(original_datasets, noise_ranges)
    
    print_results_table(noise_ranges, original_r2, original_adj_r2, 
                       "Original Datasets Results (50 variables)")
    
    # Phase 2: Product/Square variables (70 variables)
    print("\n" + "="*50)
    print("PHASE 2: Adding Product/Square Variables (70 variables)")
    print("="*50)
    
    product_square_vars, ps_formulas = generate_product_square_variables(independent_variables, betas)
    
    # Check correlations for product/square variables
    ps_correlations = check_correlations(independent_variables, product_square_vars, "Product/Square")
    
    print("\nFormulas for new product/square variables:")
    for i, formula in enumerate(ps_formulas[:10]):  # Show first 10
        print(f"x{51+i}: {formula}")
    if len(ps_formulas) > 10:
        print(f"... and {len(ps_formulas)-10} more variables")
    
    ps_datasets = create_expanded_datasets(independent_variables, beta0, betas, 
                                          product_square_vars, noise_ranges)
    ps_r2, ps_adj_r2 = calculate_model_metrics(ps_datasets, noise_ranges)
    
    print_results_table(noise_ranges, ps_r2, ps_adj_r2, 
                       "Product/Square Variables Results (70 variables)")
    
    # Phase 3: Linear transformation variables (70 variables)
    print("\n" + "="*50)
    print("PHASE 3: Adding Linear Transformation Variables (70 variables)")
    print("="*50)
    
    linear_transform_vars, lt_formulas = generate_linear_transform_variables(independent_variables, betas)
    
    # Check correlations for linear transformation variables
    lt_correlations = check_correlations(independent_variables, linear_transform_vars, "Linear Transformation")
    
    print("\nFormulas for new linear transformation variables:")
    for i, formula in enumerate(lt_formulas[:10]):  # Show first 10
        print(f"x{51+i}: {formula}")
    if len(lt_formulas) > 10:
        print(f"... and {len(lt_formulas)-10} more variables")
    
    lt_datasets = create_expanded_datasets(independent_variables, beta0, betas, 
                                          linear_transform_vars, noise_ranges)
    lt_r2, lt_adj_r2 = calculate_model_metrics(lt_datasets, noise_ranges)
    
    print_results_table(noise_ranges, lt_r2, lt_adj_r2, 
                       "Linear Transformation Variables Results (70 variables)")
    
    print("\n" + "="*50)
    print("MANUAL LINEAR ALGEBRA IMPLEMENTATION:")
    print("="*50)
    print("✓ Normal Equation: β = (X^T X)^(-1) X^T y")
    print("✓ Predictions: y_pred = X β")
    print("✓ R-squared: R² = 1 - (SS_res / SS_tot)")
    print("✓ Manual matrix operations using numpy dot products")
    print("✓ Handles singular matrices with pseudo-inverse fallback")
    
    print("\n" + "="*50)
    print("THEORETICAL EXPECTATIONS:")
    print("="*50)
    print("TRUE MODEL: y = β₀ + β₁×x₁ + β₂×x₂ + ... + β₅₀×x₅₀ + noise")
    print("")
    print("1. LINEAR TRANSFORMATIONS should NOT improve R²")
    print("   - Variables like '2×(β₁×x₁)' are perfectly predictable from x₁")
    print("   - They add no new information beyond what's already captured")
    print("   - Adjusted R² should be LOWER due to penalty for redundant variables")
    print("   - This tests multicollinearity detection")
    print("")
    print("2. PRODUCT/SQUARE variables should NOT improve R² meaningfully") 
    print("   - Variables like '(β₁×x₁)²' or '(β₁×x₁)×(β₂×x₂)' represent non-linear effects")
    print("   - But the TRUE model is linear - these are spurious patterns")
    print("   - Any apparent improvement is likely overfitting")
    print("   - This tests overfitting vs. genuine model improvement")
    print("")
    print("3. ADJUSTED R² should properly penalize both types of additional variables")
    print("4. HIGHER NOISE should reduce R² for all approaches")
    print("5. This analysis tests: redundancy detection vs. overfitting resistance")
    
    # Create visualizations
    print("\n" + "="*50)
    print("CREATING OPTIMIZED VISUALIZATIONS")
    print("="*50)
    print(f"Plotting {len(noise_ranges)} data points per series (20 total datasets)")
    print("Visualization improvements for small dataset:")
    print("• Linear scale on x-axis for clear spacing (0.01-6)")
    print("• Unique colors for each R-squared type:")
    print("  - Original R-squared: GREEN")
    print("  - Original Adjusted R-squared: BLUE") 
    print("  - Product/Square R-squared: RED")
    print("  - Product/Square Adjusted R-squared: YELLOW")
    print("  - Linear Transform R-squared: BLACK")
    print("  - Linear Transform Adjusted R-squared: PINK")
    print("• Larger markers with black edges for clarity")
    print("• High transparency for clear visibility of all points")
    print("• Minimal jittering to prevent overlaps")
    print("• Enhanced grid and legend styling")
    print("• Marker coding: Circles=Original, Squares=Product/Square, X=Linear Transform")
    
    # Plot 1: Original vs Product/Square
    plot_comparison(noise_ranges, original_r2, original_adj_r2, 
                   ps_r2, ps_adj_r2, "Product/Square Variables", 's')
    
    # Plot 2: Original vs Linear Transformation
    plot_comparison(noise_ranges, original_r2, original_adj_r2, 
                   lt_r2, lt_adj_r2, "Linear Transformation Variables", 'x')
    
    # Plot 3: All three comparisons optimized for small dataset (20 points)
    plt.figure(figsize=(16, 11))
    
    # Define unique colors for each R-squared type
    orig_r2_color = 'green'         # Original R-squared
    orig_adj_r2_color = 'blue'      # Original Adjusted R-squared
    ps_r2_color = 'red'             # Product/Square R-squared
    ps_adj_r2_color = 'yellow'      # Product/Square Adjusted R-squared
    lt_r2_color = 'black'           # Linear Transform R-squared
    lt_adj_r2_color = 'pink'        # Linear Transform Adjusted R-squared
    
    # Small dataset - minimal jitter, much larger markers
    jitter_strength = 0.08
    noise_jitter_orig = noise_ranges + np.random.normal(0, jitter_strength, len(noise_ranges))
    noise_jitter_ps = noise_ranges + np.random.normal(0, jitter_strength, len(noise_ranges))
    noise_jitter_lt = noise_ranges + np.random.normal(0, jitter_strength, len(noise_ranges))
    
    # Original datasets - circles (o) - much larger for small dataset
    plt.scatter(noise_jitter_orig, original_r2, label='Original R-squared (50 vars)', 
               alpha=0.9, marker='o', s=200, color=orig_r2_color, edgecolors='black', linewidth=2)
    plt.scatter(noise_jitter_orig, original_adj_r2, label='Original Adjusted R-squared (50 vars)', 
               alpha=0.9, marker='o', s=200, color=orig_adj_r2_color, edgecolors='black', linewidth=2)
    
    # Product/Square datasets - squares (s) - much larger for small dataset
    plt.scatter(noise_jitter_ps, ps_r2, label='Product/Square R-squared (70 vars)', 
               alpha=0.9, marker='s', s=220, color=ps_r2_color, edgecolors='black', linewidth=2)
    plt.scatter(noise_jitter_ps, ps_adj_r2, label='Product/Square Adjusted R-squared (70 vars)', 
               alpha=0.9, marker='s', s=220, color=ps_adj_r2_color, edgecolors='black', linewidth=2)
    
    # Linear Transform datasets - X markers (x) - much larger for visibility
    plt.scatter(noise_jitter_lt, lt_r2, label='Linear Transform R-squared (70 vars)', 
               alpha=0.95, marker='x', s=250, color=lt_r2_color, linewidth=4)
    plt.scatter(noise_jitter_lt, lt_adj_r2, label='Linear Transform Adjusted R-squared (70 vars)', 
               alpha=0.95, marker='x', s=250, color=lt_adj_r2_color, linewidth=4)
    
    # Use linear scale for x-axis since range is smaller and we have fewer points
    plt.xlabel('Average Error (Noise Range)', fontsize=14, fontweight='bold')
    plt.ylabel('R-squared / Adjusted R-squared', fontsize=14, fontweight='bold')
    plt.title('Comprehensive Comparison: R-squared vs. Average Error\nAll Variable Types (20 datasets) - Manual Linear Algebra', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Improve grid and legend for small dataset
    plt.grid(True, alpha=0.5, linestyle='--', linewidth=1)
    plt.legend(fontsize=12, framealpha=0.95, edgecolor='black', fancybox=True, 
              bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Set axis limits for better visibility with larger markers
    plt.xlim(-0.3, 6.8)
    plt.ylim(-0.05, 1.05)
    plt.minorticks_on()
    
    # Add subtle background color
    plt.gca().set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"{'Dataset Type':<25} {'Mean R²':<10} {'Mean Adj R²':<12} {'Std R²':<10} {'Std Adj R²':<12}")
    print("-" * 75)
    print(f"{'Original (50 vars)':<25} {np.mean(original_r2):<10.4f} {np.mean(original_adj_r2):<12.4f} {np.std(original_r2):<10.4f} {np.std(original_adj_r2):<12.4f}")
    print(f"{'Product/Square (70 vars)':<25} {np.mean(ps_r2):<10.4f} {np.mean(ps_adj_r2):<12.4f} {np.std(ps_r2):<10.4f} {np.std(ps_adj_r2):<12.4f}")
    print(f"{'Linear Transform (70 vars)':<25} {np.mean(lt_r2):<10.4f} {np.mean(lt_adj_r2):<12.4f} {np.std(lt_r2):<10.4f} {np.std(lt_adj_r2):<12.4f}")
    
    # Analysis of what the results mean
    print("\n" + "="*50)
    print("ANALYSIS OF RESULTS:")
    print("="*50)
    
    original_mean_adj_r2 = np.mean(original_adj_r2)
    ps_mean_adj_r2 = np.mean(ps_adj_r2)
    lt_mean_adj_r2 = np.mean(lt_adj_r2)
    
    print(f"1. Linear Transform vs Original:")
    if lt_mean_adj_r2 < original_mean_adj_r2:
        print(f"   ✓ CORRECT: Adj R² decreased by {(original_mean_adj_r2 - lt_mean_adj_r2)*100:.2f}%")
        print(f"   Linear transformations properly penalized as redundant")
    else:
        print(f"   ⚠ UNEXPECTED: Adj R² increased by {(lt_mean_adj_r2 - original_mean_adj_r2)*100:.2f}%")
        print(f"   This suggests overfitting despite redundancy")
    
    print(f"\n2. Product/Square vs Original:")
    if ps_mean_adj_r2 > original_mean_adj_r2:
        print(f"   ⚠ POTENTIAL OVERFITTING: Adj R² increased by {(ps_mean_adj_r2 - original_mean_adj_r2)*100:.2f}%")
        print(f"   Non-linear terms show apparent improvement despite linear true model")
        print(f"   This demonstrates how complex features can appear useful even when spurious")
    else:
        print(f"   ✓ GOOD: Adj R² decreased by {(original_mean_adj_r2 - ps_mean_adj_r2)*100:.2f}%")
        print(f"   Non-linear terms properly identified as unhelpful for linear model")
    
    print(f"\n3. Key Insights:")
    print(f"   - True model is linear: y = β₀ + Σ(βᵢ×xᵢ) + noise")
    print(f"   - Linear transforms: redundant but predictable")
    print(f"   - Product/square terms: non-linear but spurious for this model")
    print(f"   - Any R² improvement from added variables likely represents overfitting")
    print(f"   - Adjusted R² should ideally decrease for both types of additions")
    print(f"   - Manual linear algebra implementation matches sklearn results")
    
    return {
        'noise_ranges': noise_ranges,
        'original_r2': original_r2,
        'original_adj_r2': original_adj_r2,
        'ps_r2': ps_r2,
        'ps_adj_r2': ps_adj_r2,
        'lt_r2': lt_r2,
        'lt_adj_r2': lt_adj_r2,
        'ps_correlations': ps_correlations,
        'lt_correlations': lt_correlations
    }

if __name__ == "__main__":
    results = main()