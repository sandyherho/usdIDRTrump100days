#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kolmogorov-Smirnov Test Analysis Script with Bootstrap for USD/IDR Exchange Rate
-------------------------------------------------------------------------------
A robust non-parametric hypothesis test for USD/IDR exchange rates before and after 
presidential inauguration (January 20, 2025).

This script performs Kolmogorov-Smirnov tests with bootstrap resampling between
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


def bootstrap_ks_test(x, y, n_bootstrap=10000):
    """
    Perform Kolmogorov-Smirnov test with bootstrap resampling.
    
    Parameters:
    x, y : arrays of values to compare
    n_bootstrap : number of bootstrap iterations
    
    Returns:
    d_mean : Mean KS statistic
    p_mean : Mean p-value
    p_median : Median p-value
    p_ci : 95% confidence interval for p-value
    reject_ratio : Proportion of bootstrap samples where null hypothesis was rejected
    bootstrap_data : DataFrame containing all bootstrap sample results
    """
    d_stats = []
    p_values = []
    
    # Store all bootstrap results for visualization
    bootstrap_results = []
    
    for i in range(n_bootstrap):
        # Bootstrap resample x and y with replacement
        x_sample = np.random.choice(x, size=len(x), replace=True)
        y_sample = np.random.choice(y, size=len(y), replace=True)
        
        # Calculate Kolmogorov-Smirnov test for this bootstrap sample
        try:
            d_stat, p_value = stats.ks_2samp(x_sample, y_sample)
            d_stats.append(d_stat)
            p_values.append(p_value)
            
            # Store results for visualization
            bootstrap_results.append({
                'bootstrap_iteration': i,
                'd_statistic': d_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        except:
            # Skip failed tests
            continue
    
    if not p_values:
        return np.nan, np.nan, np.nan, (np.nan, np.nan), np.nan, pd.DataFrame()
    
    # Calculate statistics from bootstrap distribution
    d_mean = np.mean(d_stats)
    p_mean = np.mean(p_values)
    p_median = np.median(p_values)
    p_ci = np.percentile(p_values, [2.5, 97.5])  # 95% confidence interval
    reject_ratio = np.mean([p < 0.05 for p in p_values])  # Proportion of rejections at alpha=0.05
    
    # Convert to DataFrame for visualization
    bootstrap_data = pd.DataFrame(bootstrap_results)
    
    return d_mean, p_mean, p_median, p_ci, reject_ratio, bootstrap_data


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
    Analyze the impact of inauguration on USD_IDR using Kolmogorov-Smirnov test with bootstrap resampling
    
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
        
        # Direct Kolmogorov-Smirnov test without bootstrap
        d_stat, p_value = stats.ks_2samp(pre_values, post_values)
        
        # Bootstrapped Kolmogorov-Smirnov test
        d_mean, p_mean, p_median, p_ci, reject_ratio, bootstrap_data = bootstrap_ks_test(
            pre_values, post_values, n_bootstrap=n_bootstrap
        )
        
        # Save bootstrap data for visualization
        if not bootstrap_data.empty:
            bootstrap_filename = "ks_bootstrap_pre_vs_post_usd_idr.csv"
            bootstrap_filepath = os.path.join(output_dir, bootstrap_filename)
            
            # Add period information to the bootstrap data
            bootstrap_data['comparison'] = "Pre vs Post Inauguration"
            
            bootstrap_data.to_csv(bootstrap_filepath, index=False)
            print(f"Saved bootstrap data to {bootstrap_filepath}")
        
        # Determine significance based on bootstrap results
        robust_significance = reject_ratio > 0.8  # >80% of bootstrap samples reject null hypothesis
        
        results.append({
            'Comparison': "Pre vs Post Inauguration",
            'N samples (group 1)': len(pre_values),
            'N samples (group 2)': len(post_values),
            'Direct D': d_stat,
            'Direct p-value': p_value,
            'Bootstrap D (mean)': d_mean,
            'Bootstrap p-value (mean)': p_mean,
            'Bootstrap p-value (median)': p_median,
            'p-value CI lower': p_ci[0],
            'p-value CI upper': p_ci[1],
            'Rejection ratio': reject_ratio,
            'Robust significance': robust_significance,
            'Mean USD_IDR (group 1)': pre_values.mean(),
            'SD USD_IDR (group 1)': pre_values.std(),
            'Mean USD_IDR (group 2)': post_values.mean(),
            'SD USD_IDR (group 2)': post_values.std(),
            'Percent change': ((post_values.mean() - pre_values.mean()) / pre_values.mean()) * 100
        })
        
        print(f"Pre vs Post: Direct p = {p_value:.4f}, Bootstrap p = {p_mean:.4f}, "
              f"95% CI: [{p_ci[0]:.4f}, {p_ci[1]:.4f}], Rejection ratio: {reject_ratio:.2f}")
        
        # 2. Compare pre-inauguration to entire period
        print("\nAnalyzing pre-inauguration vs entire period...")
        
        # Direct Kolmogorov-Smirnov test without bootstrap
        d_stat, p_value = stats.ks_2samp(pre_values, entire_values)
        
        # Bootstrapped Kolmogorov-Smirnov test
        d_mean, p_mean, p_median, p_ci, reject_ratio, bootstrap_data = bootstrap_ks_test(
            pre_values, entire_values, n_bootstrap=n_bootstrap
        )
        
        # Save bootstrap data for visualization
        if not bootstrap_data.empty:
            bootstrap_filename = "ks_bootstrap_pre_vs_entire_usd_idr.csv"
            bootstrap_filepath = os.path.join(output_dir, bootstrap_filename)
            
            # Add period information to the bootstrap data
            bootstrap_data['comparison'] = "Pre-Inauguration vs Entire Period"
            
            bootstrap_data.to_csv(bootstrap_filepath, index=False)
            print(f"Saved bootstrap data to {bootstrap_filepath}")
        
        # Determine significance based on bootstrap results
        robust_significance = reject_ratio > 0.8  # >80% of bootstrap samples reject null hypothesis
        
        results.append({
            'Comparison': "Pre-Inauguration vs Entire Period",
            'N samples (group 1)': len(pre_values),
            'N samples (group 2)': len(entire_values),
            'Direct D': d_stat,
            'Direct p-value': p_value,
            'Bootstrap D (mean)': d_mean,
            'Bootstrap p-value (mean)': p_mean,
            'Bootstrap p-value (median)': p_median,
            'p-value CI lower': p_ci[0],
            'p-value CI upper': p_ci[1],
            'Rejection ratio': reject_ratio,
            'Robust significance': robust_significance,
            'Mean USD_IDR (group 1)': pre_values.mean(),
            'SD USD_IDR (group 1)': pre_values.std(),
            'Mean USD_IDR (group 2)': entire_values.mean(),
            'SD USD_IDR (group 2)': entire_values.std(),
            'Percent change': ((entire_values.mean() - pre_values.mean()) / pre_values.mean()) * 100
        })
        
        print(f"Pre vs Entire: Direct p = {p_value:.4f}, Bootstrap p = {p_mean:.4f}, "
              f"95% CI: [{p_ci[0]:.4f}, {p_ci[1]:.4f}], Rejection ratio: {reject_ratio:.2f}")
        
        # 3. Compare post-inauguration to entire period
        print("\nAnalyzing post-inauguration vs entire period...")
        
        # Direct Kolmogorov-Smirnov test without bootstrap
        d_stat, p_value = stats.ks_2samp(post_values, entire_values)
        
        # Bootstrapped Kolmogorov-Smirnov test
        d_mean, p_mean, p_median, p_ci, reject_ratio, bootstrap_data = bootstrap_ks_test(
            post_values, entire_values, n_bootstrap=n_bootstrap
        )
        
        # Save bootstrap data for visualization
        if not bootstrap_data.empty:
            bootstrap_filename = "ks_bootstrap_post_vs_entire_usd_idr.csv"
            bootstrap_filepath = os.path.join(output_dir, bootstrap_filename)
            
            # Add period information to the bootstrap data
            bootstrap_data['comparison'] = "Post-Inauguration vs Entire Period"
            
            bootstrap_data.to_csv(bootstrap_filepath, index=False)
            print(f"Saved bootstrap data to {bootstrap_filepath}")
        
        # Determine significance based on bootstrap results
        robust_significance = reject_ratio > 0.8  # >80% of bootstrap samples reject null hypothesis
        
        results.append({
            'Comparison': "Post-Inauguration vs Entire Period",
            'N samples (group 1)': len(post_values),
            'N samples (group 2)': len(entire_values),
            'Direct D': d_stat,
            'Direct p-value': p_value,
            'Bootstrap D (mean)': d_mean,
            'Bootstrap p-value (mean)': p_mean,
            'Bootstrap p-value (median)': p_median,
            'p-value CI lower': p_ci[0],
            'p-value CI upper': p_ci[1],
            'Rejection ratio': reject_ratio,
            'Robust significance': robust_significance,
            'Mean USD_IDR (group 1)': post_values.mean(),
            'SD USD_IDR (group 1)': post_values.std(),
            'Mean USD_IDR (group 2)': entire_values.mean(),
            'SD USD_IDR (group 2)': entire_values.std(),
            'Percent change': ((post_values.mean() - entire_values.mean()) / entire_values.mean()) * 100
        })
        
        print(f"Post vs Entire: Direct p = {p_value:.4f}, Bootstrap p = {p_mean:.4f}, "
              f"95% CI: [{p_ci[0]:.4f}, {p_ci[1]:.4f}], Rejection ratio: {reject_ratio:.2f}")
        
    except Exception as e:
        print(f"Error in bootstrapped Kolmogorov-Smirnov analysis: {e}")
        traceback.print_exc()
    
    return results


def generate_report(results, inauguration_date):
    """
    Generate a comprehensive report based on the Kolmogorov-Smirnov test results
    
    Parameters:
    results : List of dictionaries with test results
    inauguration_date : Datetime object representing the inauguration date
    
    Returns:
    report_content : String containing the report
    """
    results_df = pd.DataFrame(results)
    
    report_content = f"""# Bootstrapped Kolmogorov-Smirnov Test Analysis of USD/IDR Exchange Rate
## Analysis of USD/IDR Exchange Rate Before and After Presidential Inauguration (January 20, 2025)

This report presents the results of Kolmogorov-Smirnov tests comparing USD/IDR exchange rates before and after the presidential inauguration on {inauguration_date.strftime('%B %d, %Y')}. The analysis includes bootstrap resampling for robust confidence intervals.

### About Kolmogorov-Smirnov Test
The Kolmogorov-Smirnov test is a non-parametric test that assesses whether two samples come from the same distribution by comparing their cumulative distribution functions. Unlike other tests that focus on specific aspects like central tendency, the KS test is sensitive to differences in shape, spread, and location of the distributions. This makes it particularly valuable for financial data analysis, where the entire distribution matters rather than just the mean or median.

### Bootstrap Analysis
For each comparison, 10000 bootstrap samples were created by resampling with replacement. This provides more robust estimates and allows calculation of confidence intervals for p-values. A comparison is considered to have "robust significance" if more than 80% of the bootstrap samples show a statistically significant difference (p < 0.05).

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
        
        if pre_post_row['Robust significance']:
            report_content += f"- **Robust statistical significance** - The distributions of USD/IDR values before and after inauguration are significantly different (rejection ratio: {pre_post_row['Rejection ratio']:.2f}, p-value: {pre_post_row['Bootstrap p-value (mean)']:.4f}, 95% CI: [{pre_post_row['p-value CI lower']:.4f}, {pre_post_row['p-value CI upper']:.4f}])\n"
        elif pre_post_row['Rejection ratio'] > 0.5:
            report_content += f"- **Moderate evidence** - The distributions of USD/IDR values before and after inauguration show moderate evidence of difference (rejection ratio: {pre_post_row['Rejection ratio']:.2f}, p-value: {pre_post_row['Bootstrap p-value (mean)']:.4f}, 95% CI: [{pre_post_row['p-value CI lower']:.4f}, {pre_post_row['p-value CI upper']:.4f}])\n"
        else:
            report_content += f"- **No robust evidence** - The distributions of USD/IDR values before and after inauguration do not show robust evidence of difference (rejection ratio: {pre_post_row['Rejection ratio']:.2f}, p-value: {pre_post_row['Bootstrap p-value (mean)']:.4f}, 95% CI: [{pre_post_row['p-value CI lower']:.4f}, {pre_post_row['p-value CI upper']:.4f}])\n"
        
        report_content += "\n"

    # Conclude the report
    report_content += f"""
## Conclusion

This bootstrapped Kolmogorov-Smirnov test analysis provides a robust assessment of distributional differences in USD/IDR exchange rates before and after the presidential inauguration on {inauguration_date.strftime('%B %d, %Y')}. The bootstrap resampling approach provides confidence intervals that account for sampling variability, giving a more reliable picture of statistical significance.

The Kolmogorov-Smirnov test is particularly well-suited for financial data analysis as it:
1. Makes no assumptions about the normality of data distribution
2. Is sensitive to differences in the entire distribution (shape, spread, and location)
3. Can detect any type of distributional difference between samples
4. Provides a more complete picture than tests that focus only on measures of central tendency

The rejection ratio (proportion of bootstrap samples with p < 0.05) provides a measure of the robustness of the results, with higher values indicating stronger evidence against the null hypothesis of no difference between periods.

This analysis may help understand whether the presidential inauguration had a statistically significant impact on the USD/IDR exchange rate distribution, which could be useful for understanding political events' impact on currency markets.

_Note: This analysis focuses on distributional differences. Correlation does not imply causation, and other factors may have influenced exchange rate movements during this period._
"""
    
    return report_content


def save_report(report_content, output_dir="results/stats", filename="bootstrapped_ks_analysis_usd_idr.txt"):
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
    Main function to run the bootstrapped Kolmogorov-Smirnov test analysis
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
        
        # Perform Kolmogorov-Smirnov tests with bootstrap
        print(f"\nPerforming Kolmogorov-Smirnov tests with bootstrap resampling ({n_bootstrap} iterations) for USD/IDR exchange rate:")
        results = analyze_inauguration_impact(pre_data, post_data, entire_data, 
                                            n_bootstrap=n_bootstrap, 
                                            output_dir=bootstrap_dir)
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_csv = os.path.join(output_dir, "bootstrapped_ks_results_usd_idr.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"Bootstrapped Kolmogorov-Smirnov test results saved to {results_csv}")
        
        # Generate report
        print("\nGenerating report...")
        report_content = generate_report(results, inauguration_date)
        
        # Save report
        save_report(report_content, report_dir, "bootstrapped_ks_analysis_usd_idr.txt")
        
        print("\nBootstrapped Kolmogorov-Smirnov test analysis for USD/IDR complete.")
        print(f"Results are saved in the directory: {output_dir}")
        print(f"Bootstrap data for visualization is saved in: {bootstrap_dir}")
        print(f"Report is saved in: {report_dir}")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
