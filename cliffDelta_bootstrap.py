#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cliff's Delta Analysis Script with Bootstrap for USD/IDR Exchange Rate
---------------------------------------------------------------------
A robust non-parametric effect size measure for USD/IDR exchange rates before and after 
presidential inauguration (January 20, 2025).

This script calculates Cliff's Delta with bootstrap resampling between
pre-inauguration and post-inauguration periods, as well as comparing each period 
to the entire dataset.

04/29/2025
"""

import numpy as np
import pandas as pd
import os
import sys
import traceback
from scipy import stats
from datetime import datetime


def cliffs_delta(x, y):
    """
    Calculate Cliff's Delta effect size.
    
    Parameters:
    x, y : arrays of values to compare
    
    Returns:
    delta : Cliff's Delta effect size
    """
    # Count comparisons where x > y, x < y, and x == y
    greater = 0
    lesser = 0
    
    for i in x:
        for j in y:
            if i > j:
                greater += 1
            elif i < j:
                lesser += 1
    
    # Calculate delta
    delta = (greater - lesser) / (len(x) * len(y))
    
    return delta


def interpret_cliffs_delta(delta):
    """
    Interpret Cliff's Delta effect size according to common thresholds.
    
    Parameters:
    delta : Cliff's Delta value
    
    Returns:
    interpretation : String interpretation of the effect size
    """
    delta_abs = abs(delta)
    
    if delta_abs < 0.147:
        return "Negligible"
    elif delta_abs < 0.33:
        return "Small"
    elif delta_abs < 0.474:
        return "Medium"
    else:
        return "Large"


def bootstrap_cliffs_delta(x, y, n_bootstrap=10000, alpha=0.05):
    """
    Perform Cliff's Delta calculation with bootstrap resampling.
    
    Parameters:
    x, y : arrays of values to compare
    n_bootstrap : number of bootstrap iterations
    alpha : significance level for confidence intervals
    
    Returns:
    delta : Original Cliff's Delta
    delta_mean : Mean bootstrapped Delta
    delta_ci : Confidence interval for Delta
    bootstrap_data : DataFrame containing all bootstrap sample results
    """
    # Calculate original Cliff's Delta
    original_delta = cliffs_delta(x, y)
    
    # Store bootstrap results
    bootstrap_deltas = []
    bootstrap_results = []
    
    for i in range(n_bootstrap):
        # Bootstrap resample x and y with replacement
        x_sample = np.random.choice(x, size=len(x), replace=True)
        y_sample = np.random.choice(y, size=len(y), replace=True)
        
        # Calculate Cliff's Delta for this bootstrap sample
        try:
            delta = cliffs_delta(x_sample, y_sample)
            bootstrap_deltas.append(delta)
            
            # Store results for visualization
            bootstrap_results.append({
                'bootstrap_iteration': i,
                'delta': delta,
                'interpretation': interpret_cliffs_delta(delta)
            })
        except:
            # Skip failed calculations
            continue
    
    if not bootstrap_deltas:
        return original_delta, np.nan, (np.nan, np.nan), pd.DataFrame()
    
    # Calculate statistics from bootstrap distribution
    delta_mean = np.mean(bootstrap_deltas)
    delta_ci = np.percentile(bootstrap_deltas, [alpha/2 * 100, (1-alpha/2) * 100])  # confidence interval
    
    # Convert to DataFrame for visualization
    bootstrap_data = pd.DataFrame(bootstrap_results)
    
    return original_delta, delta_mean, delta_ci, bootstrap_data


def load_data(file_path):
    """
    Load and prepare data from the CSV file
    
    Parameters:
    file_path : Path to the CSV file
    
    Returns:
    df : Pandas DataFrame with date and USD_IDR columns
    """
    try:
        df = pd.read_csv(file_path)
        
        # Convert the index to a proper datetime
        df.index = pd.to_datetime(df.index)
        
        # Check if USD_IDR column exists
        if 'USD_IDR' not in df.columns:
            print(f"Error: Required column 'USD_IDR' not found in {file_path}")
            print(f"Available columns: {df.columns.tolist()}")
            sys.exit(1)
            
        # Drop any rows with NaN values
        df = df.dropna().reset_index()
        df.rename(columns={'index': 'date'}, inplace=True)
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        traceback.print_exc()
        sys.exit(1)


def analyze_inauguration_impact(pre_data, post_data, entire_data, n_bootstrap=10000, 
                                output_dir="results/bootstrap_data"):
    """
    Analyze the impact of inauguration on USD_IDR using Cliff's Delta with bootstrap resampling
    
    Parameters:
    pre_data : DataFrame with USD_IDR before inauguration
    post_data : DataFrame with USD_IDR after inauguration
    entire_data : DataFrame with USD_IDR for entire period
    n_bootstrap : Number of bootstrap iterations
    output_dir : Directory to save bootstrap data
    
    Returns:
    results : List of dictionaries with test results
    """
    # Create directory for bootstrap data if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    try:
        # Extract values
        pre_values = pre_data['USD_IDR'].values
        post_values = post_data['USD_IDR'].values
        entire_values = entire_data['USD_IDR'].values
        
        # 1. Compare pre-inauguration to post-inauguration (direct comparison)
        print("\nAnalyzing pre-inauguration vs post-inauguration periods...")
        
        # Calculate Cliff's Delta with bootstrap
        delta, delta_mean, delta_ci, bootstrap_data = bootstrap_cliffs_delta(
            pre_values, post_values, n_bootstrap=n_bootstrap
        )
        
        # Save bootstrap data for visualization
        if not bootstrap_data.empty:
            bootstrap_filename = "cd_bootstrap_pre_vs_post_usd_idr.csv"
            bootstrap_filepath = os.path.join(output_dir, bootstrap_filename)
            
            # Add period information to the bootstrap data
            bootstrap_data['comparison'] = "Pre vs Post Inauguration"
            
            bootstrap_data.to_csv(bootstrap_filepath, index=False)
            print(f"Saved bootstrap data to {bootstrap_filepath}")
        
        # Determine interpretation
        interpretation = interpret_cliffs_delta(delta)
        
        results.append({
            'Comparison': "Pre vs Post Inauguration",
            'N samples (group 1)': len(pre_values),
            'N samples (group 2)': len(post_values),
            'Cliffs Delta': delta,  # Changed from 'Cliff\'s Delta' to avoid escape character
            'Interpretation': interpretation,
            'Bootstrap Delta (mean)': delta_mean,
            'Delta CI lower': delta_ci[0],
            'Delta CI upper': delta_ci[1],
            'Mean USD_IDR (group 1)': pre_values.mean(),
            'SD USD_IDR (group 1)': pre_values.std(),
            'Mean USD_IDR (group 2)': post_values.mean(),
            'SD USD_IDR (group 2)': post_values.std(),
            'Percent change': ((post_values.mean() - pre_values.mean()) / pre_values.mean()) * 100
        })
        
        print(f"Pre vs Post: Cliff's Delta = {delta:.4f} ({interpretation}), "
              f"Bootstrap mean = {delta_mean:.4f}, 95% CI: [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]")
        
        # 2. Compare pre-inauguration to entire period
        print("\nAnalyzing pre-inauguration vs entire period...")
        
        # Calculate Cliff's Delta with bootstrap
        delta, delta_mean, delta_ci, bootstrap_data = bootstrap_cliffs_delta(
            pre_values, entire_values, n_bootstrap=n_bootstrap
        )
        
        # Save bootstrap data for visualization
        if not bootstrap_data.empty:
            bootstrap_filename = "cd_bootstrap_pre_vs_entire_usd_idr.csv"
            bootstrap_filepath = os.path.join(output_dir, bootstrap_filename)
            
            # Add period information to the bootstrap data
            bootstrap_data['comparison'] = "Pre-Inauguration vs Entire Period"
            
            bootstrap_data.to_csv(bootstrap_filepath, index=False)
            print(f"Saved bootstrap data to {bootstrap_filepath}")
        
        # Determine interpretation
        interpretation = interpret_cliffs_delta(delta)
        
        results.append({
            'Comparison': "Pre-Inauguration vs Entire Period",
            'N samples (group 1)': len(pre_values),
            'N samples (group 2)': len(entire_values),
            'Cliffs Delta': delta,  # Changed from 'Cliff\'s Delta' to avoid escape character
            'Interpretation': interpretation,
            'Bootstrap Delta (mean)': delta_mean,
            'Delta CI lower': delta_ci[0],
            'Delta CI upper': delta_ci[1],
            'Mean USD_IDR (group 1)': pre_values.mean(),
            'SD USD_IDR (group 1)': pre_values.std(),
            'Mean USD_IDR (group 2)': entire_values.mean(),
            'SD USD_IDR (group 2)': entire_values.std(),
            'Percent change': ((entire_values.mean() - pre_values.mean()) / pre_values.mean()) * 100
        })
        
        print(f"Pre vs Entire: Cliff's Delta = {delta:.4f} ({interpretation}), "
              f"Bootstrap mean = {delta_mean:.4f}, 95% CI: [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]")
        
        # 3. Compare post-inauguration to entire period
        print("\nAnalyzing post-inauguration vs entire period...")
        
        # Calculate Cliff's Delta with bootstrap
        delta, delta_mean, delta_ci, bootstrap_data = bootstrap_cliffs_delta(
            post_values, entire_values, n_bootstrap=n_bootstrap
        )
        
        # Save bootstrap data for visualization
        if not bootstrap_data.empty:
            bootstrap_filename = "cd_bootstrap_post_vs_entire_usd_idr.csv"
            bootstrap_filepath = os.path.join(output_dir, bootstrap_filename)
            
            # Add period information to the bootstrap data
            bootstrap_data['comparison'] = "Post-Inauguration vs Entire Period"
            
            bootstrap_data.to_csv(bootstrap_filepath, index=False)
            print(f"Saved bootstrap data to {bootstrap_filepath}")
        
        # Determine interpretation
        interpretation = interpret_cliffs_delta(delta)
        
        results.append({
            'Comparison': "Post-Inauguration vs Entire Period",
            'N samples (group 1)': len(post_values),
            'N samples (group 2)': len(entire_values),
            'Cliffs Delta': delta,  # Changed from 'Cliff\'s Delta' to avoid escape character
            'Interpretation': interpretation,
            'Bootstrap Delta (mean)': delta_mean,
            'Delta CI lower': delta_ci[0],
            'Delta CI upper': delta_ci[1],
            'Mean USD_IDR (group 1)': post_values.mean(),
            'SD USD_IDR (group 1)': post_values.std(),
            'Mean USD_IDR (group 2)': entire_values.mean(),
            'SD USD_IDR (group 2)': entire_values.std(),
            'Percent change': ((post_values.mean() - entire_values.mean()) / entire_values.mean()) * 100
        })
        
        print(f"Post vs Entire: Cliff's Delta = {delta:.4f} ({interpretation}), "
              f"Bootstrap mean = {delta_mean:.4f}, 95% CI: [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]")
        
    except Exception as e:
        print(f"Error in bootstrapped Cliff's Delta analysis: {e}")
        traceback.print_exc()
    
    return results


def generate_report(results, inauguration_date):
    """
    Generate a comprehensive report based on the Cliff's Delta results
    
    Parameters:
    results : List of dictionaries with test results
    inauguration_date : Datetime object representing the inauguration date
    
    Returns:
    report_content : String containing the report
    """
    results_df = pd.DataFrame(results)
    
    report_content = f"""# Bootstrapped Cliff's Delta Analysis of USD/IDR Exchange Rate
## Analysis of USD/IDR Exchange Rate Before and After Presidential Inauguration (January 20, 2025)

This report presents the results of Cliff's Delta effect size measure comparing USD/IDR exchange rates before and after the presidential inauguration on {inauguration_date.strftime('%B %d, %Y')}. The analysis includes bootstrap resampling for robust confidence intervals.

### About Cliff's Delta
Cliff's Delta is a non-parametric effect size measure that quantifies the amount of difference between two groups of observations. The measure ranges from -1 to 1, where:
- Values near 0 indicate negligible difference between groups
- Values near 1 indicate that values in the first group tend to be larger than those in the second group
- Values near -1 indicate that values in the second group tend to be larger than those in the first group

Cliff's Delta is interpreted according to these thresholds:
- |d| < 0.147: Negligible effect
- 0.147 ≤ |d| < 0.33: Small effect
- 0.33 ≤ |d| < 0.474: Medium effect
- |d| ≥ 0.474: Large effect

### Bootstrap Analysis
For each comparison, 10000 bootstrap samples were created by resampling with replacement. This provides more robust estimates and allows calculation of confidence intervals for the effect size.

## Results

"""

    # Add results to the report
    if not results_df.empty:
        report_content += "### Analysis by Comparison\n\n"
        report_content += results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}")
        report_content += "\n\n"
    else:
        report_content += "No results available for analysis.\n\n"

    report_content += """
## Interpretation Summary

"""

    # Add summary interpretations
    if not results_df.empty:
        # Pre vs Post Inauguration results
        pre_post_row = results_df[results_df['Comparison'] == "Pre vs Post Inauguration"].iloc[0]
        
        report_content += "### Pre-Inauguration vs Post-Inauguration Analysis:\n"
        report_content += f"- Mean USD/IDR before inauguration: {pre_post_row['Mean USD_IDR (group 1)']:.2f} (SD: {pre_post_row['SD USD_IDR (group 1)']:.2f})\n"
        report_content += f"- Mean USD/IDR after inauguration: {pre_post_row['Mean USD_IDR (group 2)']:.2f} (SD: {pre_post_row['SD USD_IDR (group 2)']:.2f})\n"
        report_content += f"- Percent change: {pre_post_row['Percent change']:.2f}%\n"
        # Fixed line to avoid backslash in f-string
        report_content += f"- Cliff's Delta: {pre_post_row['Cliffs Delta']:.4f} ({pre_post_row['Interpretation']} effect), 95% CI: [{pre_post_row['Delta CI lower']:.4f}, {pre_post_row['Delta CI upper']:.4f}]\n"
        
        # Interpretation based on direction and magnitude
        delta = pre_post_row['Cliffs Delta']  # Changed from 'Cliff\'s Delta' to avoid escape character
        if delta > 0:
            report_content += f"- This positive delta indicates that USD/IDR values tend to be **higher** after the inauguration.\n"
        elif delta < 0:
            report_content += f"- This negative delta indicates that USD/IDR values tend to be **lower** after the inauguration.\n"
        else:
            report_content += f"- The delta of zero indicates no consistent difference between pre and post-inauguration USD/IDR values.\n"
        
        report_content += "\n"

    # Conclude the report
    report_content += f"""
## Conclusion

This bootstrapped Cliff's Delta analysis provides a robust assessment of the magnitude of differences in USD/IDR exchange rates before and after the presidential inauguration on {inauguration_date.strftime('%B %d, %Y')}. The bootstrap resampling approach provides confidence intervals that account for sampling variability, giving a more reliable picture of the effect size.

Cliff's Delta is particularly well-suited for financial data analysis as it:
1. Makes no assumptions about the underlying distributions
2. Is robust against outliers
3. Quantifies not just whether there is a difference, but how substantial that difference is
4. Provides an intuitive measure of how often values in one group exceed values in another group

Unlike hypothesis tests that only tell us if there is a statistically significant difference, Cliff's Delta tells us about the size of that difference, which is often more important for practical decision-making.

This analysis may help understand the magnitude of the presidential inauguration's impact on the USD/IDR exchange rate, which could be useful for understanding political events' impact on currency markets.

_Note: This analysis quantifies the magnitude of differences. Correlation does not imply causation, and other factors may have influenced exchange rate movements during this period._
"""
    
    return report_content


def save_report(report_content, output_dir="results/stats", filename="bootstrapped_cliffs_delta_analysis_usd_idr.txt"):
    """
    Save the report to a file
    
    Parameters:
    report_content : String containing the report
    output_dir : Directory to save the report
    filename : Name of the report file
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write the report to a file
        report_file = os.path.join(output_dir, filename)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"\nReport successfully written to {report_file}")
        
    except Exception as e:
        print(f"Error saving report: {e}")
        traceback.print_exc()


def main():
    """
    Main function to run the bootstrapped Cliff's Delta analysis
    for USD/IDR exchange rate before and after inauguration
    """
    # Define file paths
    full_period_file = "data/usd_idr_inauguration_period.csv"
    pre_inauguration_file = "data/usd_idr_pre_inauguration.csv"
    post_inauguration_file = "data/usd_idr_post_inauguration.csv"
    
    output_dir = "results"
    report_dir = "results/stats"
    bootstrap_dir = "results/bootstrap_data"
    
    # Define inauguration date
    inauguration_date = datetime(2025, 1, 20)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define number of bootstrap iterations
    n_bootstrap = 10000
    
    try:
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(bootstrap_dir, exist_ok=True)
        
        # Load the data
        print("Loading data...")
        
        entire_data = load_data(full_period_file)
        pre_data = load_data(pre_inauguration_file)
        post_data = load_data(post_inauguration_file)
        
        print(f"Data loaded successfully:")
        print(f"Full period: {len(entire_data)} records")
        print(f"Pre-inauguration: {len(pre_data)} records")
        print(f"Post-inauguration: {len(post_data)} records")
        
        # Perform Cliff's Delta calculations with bootstrap
        print(f"\nCalculating Cliff's Delta with bootstrap resampling ({n_bootstrap} iterations) for USD/IDR exchange rate:")
        results = analyze_inauguration_impact(pre_data, post_data, entire_data, 
                                            n_bootstrap=n_bootstrap, 
                                            output_dir=bootstrap_dir)
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_csv = os.path.join(output_dir, "bootstrapped_cliffs_delta_results_usd_idr.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"Bootstrapped Cliff's Delta results saved to {results_csv}")
        
        # Generate report
        print("\nGenerating report...")
        report_content = generate_report(results, inauguration_date)
        
        # Save report
        save_report(report_content, report_dir, "bootstrapped_cliffs_delta_analysis_usd_idr.txt")
        
        print("\nBootstrapped Cliff's Delta analysis for USD/IDR complete.")
        print(f"Results are saved in the directory: {output_dir}")
        print(f"Bootstrap data for visualization is saved in: {bootstrap_dir}")
        print(f"Report is saved in: {report_dir}")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
