#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brown-Forsythe Test Analysis Script with Bootstrap for USD/IDR Exchange Rate
---------------------------------------------------------------------------
A robust test for homogeneity of variances in USD/IDR exchange rates before and after 
presidential inauguration (January 20, 2025).

This script performs Brown-Forsythe tests with bootstrap resampling between
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


def brown_forsythe_test(x, y):
    """
    Perform Brown-Forsythe test for equal variances.
    
    Parameters:
    x, y : arrays of values to compare
    
    Returns:
    stat : Test statistic
    p_value : P-value for the test
    """
    # Calculate deviations from the median for each group
    x_median = np.median(x)
    y_median = np.median(y)
    
    x_dev = np.abs(x - x_median)
    y_dev = np.abs(y - y_median)
    
    # Combine deviations and create group labels
    combined_dev = np.concatenate([x_dev, y_dev])
    group_labels = np.concatenate([np.zeros(len(x)), np.ones(len(y))])
    
    # Perform one-way ANOVA on the deviations
    stat, p_value = stats.f_oneway(x_dev, y_dev)
    
    return stat, p_value


def bootstrap_brown_forsythe(x, y, n_bootstrap=10000):
    """
    Perform Brown-Forsythe test with bootstrap resampling.
    
    Parameters:
    x, y : arrays of values to compare
    n_bootstrap : number of bootstrap iterations
    
    Returns:
    stat_mean : Mean test statistic
    p_mean : Mean p-value
    p_median : Median p-value
    p_ci : 95% confidence interval for p-value
    reject_ratio : Proportion of bootstrap samples where null hypothesis was rejected
    bootstrap_data : DataFrame containing all bootstrap sample results
    """
    stats_values = []
    p_values = []
    
    # Store all bootstrap results for visualization
    bootstrap_results = []
    
    for i in range(n_bootstrap):
        # Bootstrap resample x and y with replacement
        x_sample = np.random.choice(x, size=len(x), replace=True)
        y_sample = np.random.choice(y, size=len(y), replace=True)
        
        # Calculate Brown-Forsythe test for this bootstrap sample
        try:
            stat, p_value = brown_forsythe_test(x_sample, y_sample)
            stats_values.append(stat)
            p_values.append(p_value)
            
            # Store results for visualization
            bootstrap_results.append({
                'bootstrap_iteration': i,
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        except:
            # Skip failed tests
            continue
    
    if not p_values:
        return np.nan, np.nan, np.nan, (np.nan, np.nan), np.nan, pd.DataFrame()
    
    # Calculate statistics from bootstrap distribution
    stat_mean = np.mean(stats_values)
    p_mean = np.mean(p_values)
    p_median = np.median(p_values)
    p_ci = np.percentile(p_values, [2.5, 97.5])  # 95% confidence interval
    reject_ratio = np.mean([p < 0.05 for p in p_values])  # Proportion of rejections at alpha=0.05
    
    # Convert to DataFrame for visualization
    bootstrap_data = pd.DataFrame(bootstrap_results)
    
    return stat_mean, p_mean, p_median, p_ci, reject_ratio, bootstrap_data


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
    Analyze the impact of inauguration on USD_IDR using Brown-Forsythe test with bootstrap resampling
    
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
        
        # Direct Brown-Forsythe test without bootstrap
        stat, p_value = brown_forsythe_test(pre_values, post_values)
        
        # Bootstrapped Brown-Forsythe test
        stat_mean, p_mean, p_median, p_ci, reject_ratio, bootstrap_data = bootstrap_brown_forsythe(
            pre_values, post_values, n_bootstrap=n_bootstrap
        )
        
        # Save bootstrap data for visualization
        if not bootstrap_data.empty:
            bootstrap_filename = "bf_bootstrap_pre_vs_post_usd_idr.csv"
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
            'Direct Statistic': stat,
            'Direct p-value': p_value,
            'Bootstrap Statistic (mean)': stat_mean,
            'Bootstrap p-value (mean)': p_mean,
            'Bootstrap p-value (median)': p_median,
            'p-value CI lower': p_ci[0],
            'p-value CI upper': p_ci[1],
            'Rejection ratio': reject_ratio,
            'Robust significance': robust_significance,
            'Variance (group 1)': np.var(pre_values, ddof=1),
            'SD (group 1)': np.std(pre_values, ddof=1),
            'Variance (group 2)': np.var(post_values, ddof=1),
            'SD (group 2)': np.std(post_values, ddof=1),
            'Variance ratio': np.var(post_values, ddof=1) / np.var(pre_values, ddof=1)
        })
        
        print(f"Pre vs Post: Direct p = {p_value:.4f}, Bootstrap p = {p_mean:.4f}, "
              f"95% CI: [{p_ci[0]:.4f}, {p_ci[1]:.4f}], Rejection ratio: {reject_ratio:.2f}")
        
        # 2. Compare pre-inauguration to entire period
        print("\nAnalyzing pre-inauguration vs entire period...")
        
        # Direct Brown-Forsythe test without bootstrap
        stat, p_value = brown_forsythe_test(pre_values, entire_values)
        
        # Bootstrapped Brown-Forsythe test
        stat_mean, p_mean, p_median, p_ci, reject_ratio, bootstrap_data = bootstrap_brown_forsythe(
            pre_values, entire_values, n_bootstrap=n_bootstrap
        )
        
        # Save bootstrap data for visualization
        if not bootstrap_data.empty:
            bootstrap_filename = "bf_bootstrap_pre_vs_entire_usd_idr.csv"
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
            'Direct Statistic': stat,
            'Direct p-value': p_value,
            'Bootstrap Statistic (mean)': stat_mean,
            'Bootstrap p-value (mean)': p_mean,
            'Bootstrap p-value (median)': p_median,
            'p-value CI lower': p_ci[0],
            'p-value CI upper': p_ci[1],
            'Rejection ratio': reject_ratio,
            'Robust significance': robust_significance,
            'Variance (group 1)': np.var(pre_values, ddof=1),
            'SD (group 1)': np.std(pre_values, ddof=1),
            'Variance (group 2)': np.var(entire_values, ddof=1),
            'SD (group 2)': np.std(entire_values, ddof=1),
            'Variance ratio': np.var(entire_values, ddof=1) / np.var(pre_values, ddof=1)
        })
        
        print(f"Pre vs Entire: Direct p = {p_value:.4f}, Bootstrap p = {p_mean:.4f}, "
              f"95% CI: [{p_ci[0]:.4f}, {p_ci[1]:.4f}], Rejection ratio: {reject_ratio:.2f}")
        
        # 3. Compare post-inauguration to entire period
        print("\nAnalyzing post-inauguration vs entire period...")
        
        # Direct Brown-Forsythe test without bootstrap
        stat, p_value = brown_forsythe_test(post_values, entire_values)
        
        # Bootstrapped Brown-Forsythe test
        stat_mean, p_mean, p_median, p_ci, reject_ratio, bootstrap_data = bootstrap_brown_forsythe(
            post_values, entire_values, n_bootstrap=n_bootstrap
        )
        
        # Save bootstrap data for visualization
        if not bootstrap_data.empty:
            bootstrap_filename = "bf_bootstrap_post_vs_entire_usd_idr.csv"
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
            'Direct Statistic': stat,
            'Direct p-value': p_value,
            'Bootstrap Statistic (mean)': stat_mean,
            'Bootstrap p-value (mean)': p_mean,
            'Bootstrap p-value (median)': p_median,
            'p-value CI lower': p_ci[0],
            'p-value CI upper': p_ci[1],
            'Rejection ratio': reject_ratio,
            'Robust significance': robust_significance,
            'Variance (group 1)': np.var(post_values, ddof=1),
            'SD (group 1)': np.std(post_values, ddof=1),
            'Variance (group 2)': np.var(entire_values, ddof=1),
            'SD (group 2)': np.std(entire_values, ddof=1),
            'Variance ratio': np.var(entire_values, ddof=1) / np.var(post_values, ddof=1)
        })
        
        print(f"Post vs Entire: Direct p = {p_value:.4f}, Bootstrap p = {p_mean:.4f}, "
              f"95% CI: [{p_ci[0]:.4f}, {p_ci[1]:.4f}], Rejection ratio: {reject_ratio:.2f}")
        
    except Exception as e:
        print(f"Error in bootstrapped Brown-Forsythe analysis: {e}")
        traceback.print_exc()
    
    return results


def generate_report(results, inauguration_date):
    """
    Generate a comprehensive report based on the Brown-Forsythe test results
    
    Parameters:
    results : List of dictionaries with test results
    inauguration_date : Datetime object representing the inauguration date
    
    Returns:
    report_content : String containing the report
    """
    results_df = pd.DataFrame(results)
    
    report_content = f"""# Bootstrapped Brown-Forsythe Test Analysis of USD/IDR Exchange Rate
## Analysis of USD/IDR Exchange Rate Variance Before and After Presidential Inauguration (January 20, 2025)

This report presents the results of Brown-Forsythe tests comparing the variance of USD/IDR exchange rates before and after the presidential inauguration on {inauguration_date.strftime('%B %d, %Y')}. The analysis includes bootstrap resampling for robust confidence intervals.

### About Brown-Forsythe Test
The Brown-Forsythe test is a robust test for homogeneity of variances across groups. Unlike the traditional Levene's test which uses deviations from group means, the Brown-Forsythe test uses deviations from group medians, making it more robust against non-normal distributions and outliers. The test evaluates the null hypothesis that the variances of the groups are equal.

### Bootstrap Analysis
For each comparison, 10000 bootstrap samples were created by resampling with replacement. This provides more robust estimates and allows calculation of confidence intervals for p-values. A comparison is considered to have "robust significance" if more than 80% of the bootstrap samples reject the null hypothesis (p < 0.05).

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
        report_content += f"- Variance of USD/IDR before inauguration: {pre_post_row['Variance (group 1)']:.2f} (SD: {pre_post_row['SD (group 1)']:.2f})\n"
        report_content += f"- Variance of USD/IDR after inauguration: {pre_post_row['Variance (group 2)']:.2f} (SD: {pre_post_row['SD (group 2)']:.2f})\n"
        report_content += f"- Variance ratio (post/pre): {pre_post_row['Variance ratio']:.4f}\n"
        
        if pre_post_row['Robust significance']:
            report_content += f"- **Robust statistical significance** - The variance differs significantly between pre and post-inauguration periods (rejection ratio: {pre_post_row['Rejection ratio']:.2f}, p-value: {pre_post_row['Bootstrap p-value (mean)']:.4f}, 95% CI: [{pre_post_row['p-value CI lower']:.4f}, {pre_post_row['p-value CI upper']:.4f}])\n"
        elif pre_post_row['Rejection ratio'] > 0.5:
            report_content += f"- **Moderate evidence** - There is moderate evidence that the variance differs between pre and post-inauguration periods (rejection ratio: {pre_post_row['Rejection ratio']:.2f}, p-value: {pre_post_row['Bootstrap p-value (mean)']:.4f}, 95% CI: [{pre_post_row['p-value CI lower']:.4f}, {pre_post_row['p-value CI upper']:.4f}])\n"
        else:
            report_content += f"- **No robust evidence** - There is no robust evidence that the variance differs between pre and post-inauguration periods (rejection ratio: {pre_post_row['Rejection ratio']:.2f}, p-value: {pre_post_row['Bootstrap p-value (mean)']:.4f}, 95% CI: [{pre_post_row['p-value CI lower']:.4f}, {pre_post_row['p-value CI upper']:.4f}])\n"
        
        report_content += "\n"

    # Conclude the report
    report_content += f"""
## Conclusion

This bootstrapped Brown-Forsythe test analysis provides a robust assessment of differences in variance of USD/IDR exchange rates before and after the presidential inauguration on {inauguration_date.strftime('%B %d, %Y')}. The bootstrap resampling approach provides confidence intervals that account for sampling variability, giving a more reliable picture of statistical significance.

The Brown-Forsythe test is particularly well-suited for financial data analysis as it:
1. Makes no assumptions about the normality of data distribution
2. Is robust against outliers by using deviations from the median rather than the mean
3. Specifically tests for differences in variance, which is a key measure of volatility in financial markets
4. Works well even with unequal sample sizes

Changes in variance before and after a political event like a presidential inauguration can indicate changes in market uncertainty and risk. Higher variance generally suggests greater uncertainty and volatility, while lower variance suggests more stable market conditions.

This analysis may help understand whether the presidential inauguration had a significant impact on the volatility of the USD/IDR exchange rate, which could be useful for understanding political events' impact on currency market stability.

_Note: This analysis focuses on differences in variance. Correlation does not imply causation, and other factors may have influenced exchange rate volatility during this period._
"""
    
    return report_content


def save_report(report_content, output_dir="results/stats", filename="bootstrapped_brown_forsythe_analysis_usd_idr.txt"):
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
    Main function to run the bootstrapped Brown-Forsythe test analysis
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
        
        # Perform Brown-Forsythe tests with bootstrap
        print(f"\nPerforming Brown-Forsythe tests with bootstrap resampling ({n_bootstrap} iterations) for USD/IDR exchange rate:")
        results = analyze_inauguration_impact(pre_data, post_data, entire_data, 
                                            n_bootstrap=n_bootstrap, 
                                            output_dir=bootstrap_dir)
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_csv = os.path.join(output_dir, "bootstrapped_brown_forsythe_results_usd_idr.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"Bootstrapped Brown-Forsythe test results saved to {results_csv}")
        
        # Generate report
        print("\nGenerating report...")
        report_content = generate_report(results, inauguration_date)
        
        # Save report
        save_report(report_content, report_dir, "bootstrapped_brown_forsythe_analysis_usd_idr.txt")
        
        print("\nBootstrapped Brown-Forsythe test analysis for USD/IDR complete.")
        print(f"Results are saved in the directory: {output_dir}")
        print(f"Bootstrap data for visualization is saved in: {bootstrap_dir}")
        print(f"Report is saved in: {report_dir}")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
