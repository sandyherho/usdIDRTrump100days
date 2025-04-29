#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Robust Normality Tests and Visualization for USD/IDR Exchange Rate
-----------------------------------------------------------------
This script performs robust normality tests and creates visualizations for
USD/IDR exchange rates before and after presidential inauguration (January 20, 2025).

04/29/2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import traceback
from scipy import stats
from datetime import datetime


def shapiro_wilk_test(data):
    """
    Perform Shapiro-Wilk test for normality.
    
    Parameters:
    data : array of values to test
    
    Returns:
    stat : Test statistic
    p_value : P-value for the test
    """
    return stats.shapiro(data)


def anderson_darling_test(data):
    """
    Perform Anderson-Darling test for normality.
    
    Parameters:
    data : array of values to test
    
    Returns:
    stat : Test statistic
    critical_values : Critical values for different significance levels
    significance_levels : Significance levels for the critical values
    """
    return stats.anderson(data, dist='norm')


def jarque_bera_test(data):
    """
    Perform Jarque-Bera test for normality.
    
    Parameters:
    data : array of values to test
    
    Returns:
    stat : Test statistic
    p_value : P-value for the test
    """
    return stats.jarque_bera(data)


def d_agostino_pearson_test(data):
    """
    Perform D'Agostino-Pearson test for normality.
    
    Parameters:
    data : array of values to test
    
    Returns:
    stat : Test statistic
    p_value : P-value for the test
    """
    return stats.normaltest(data)


def lilliefors_test(data):
    """
    Perform Lilliefors test for normality.
    
    Parameters:
    data : array of values to test
    
    Returns:
    stat : Test statistic
    p_value : P-value for the test
    """
    return stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))


def robust_descriptive_stats(data):
    """
    Calculate robust descriptive statistics for the data.
    
    Parameters:
    data : array of values
    
    Returns:
    stats_dict : Dictionary containing various statistics
    """
    # Handle empty data
    if len(data) == 0:
        return {
            'count': 0,
            'mean': np.nan,
            'median': np.nan,
            'trimmed_mean_10': np.nan,
            'trimmed_mean_20': np.nan,
            'std': np.nan,
            'mad': np.nan,
            'iqr': np.nan,
            'min': np.nan,
            'max': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan
        }
    
    # Calculate trimmed means (robust to outliers)
    trimmed_mean_10 = stats.trim_mean(data, 0.1)  # 10% trimmed
    trimmed_mean_20 = stats.trim_mean(data, 0.2)  # 20% trimmed
    
    # Calculate Median Absolute Deviation (MAD)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    # Calculate Interquartile Range (IQR)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    
    # Calculate skewness and kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    
    return {
        'count': len(data),
        'mean': np.mean(data),
        'median': median,
        'trimmed_mean_10': trimmed_mean_10,
        'trimmed_mean_20': trimmed_mean_20,
        'std': np.std(data, ddof=1),
        'mad': mad,
        'iqr': iqr,
        'min': np.min(data),
        'max': np.max(data),
        'skewness': skewness,
        'kurtosis': kurtosis
    }


def create_kde_plot(pre_data, post_data, full_data, output_dir="results/figures", 
                   filename="kde_plot_usd_idr.png"):
    """
    Create Kernel Density Estimation (KDE) plot for the three datasets.
    
    Parameters:
    pre_data : DataFrame with USD_IDR before inauguration
    post_data : DataFrame with USD_IDR after inauguration
    full_data : DataFrame with USD_IDR for entire period
    output_dir : Directory to save the figure
    filename : Filename for the plot
    """
    try:
        # Extract values
        pre_values = pre_data['USD_IDR'].values
        post_values = post_data['USD_IDR'].values
        full_values = full_data['USD_IDR'].values
        
        # Set figure size and style
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Plot KDE for each period
        sns.kdeplot(pre_values, label='Pre-Inauguration', alpha=0.7, linewidth=2)
        sns.kdeplot(post_values, label='Post-Inauguration', alpha=0.7, linewidth=2)
        sns.kdeplot(full_values, label='Entire Period', alpha=0.7, linewidth=2, linestyle='--')
        
        # Add labels and legend
        plt.xlabel('USD/IDR Exchange Rate [IDR]', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.legend(fontsize=13)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, filename), dpi=400, bbox_inches='tight')
        plt.close()
        
        print(f"KDE plot saved to {os.path.join(output_dir, filename)}")
    
    except Exception as e:
        print(f"Error creating KDE plot: {e}")
        traceback.print_exc()


def create_boxplot(pre_data, post_data, full_data, output_dir="results/figures", 
                  filename="boxplot_usd_idr.png"):
    """
    Create boxplot for the three datasets.
    
    Parameters:
    pre_data : DataFrame with USD_IDR before inauguration
    post_data : DataFrame with USD_IDR after inauguration
    full_data : DataFrame with USD_IDR for entire period
    output_dir : Directory to save the figure
    filename : Filename for the plot
    """
    try:
        # Extract values
        pre_values = pre_data['USD_IDR'].values
        post_values = post_data['USD_IDR'].values
        full_values = full_data['USD_IDR'].values
        
        # Prepare data for boxplot
        data_to_plot = [pre_values, post_values, full_values]
        labels = ['Pre-Inauguration', 'Post-Inauguration', 'Entire Period']
        
        # Set figure size and style
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Create boxplot
        box = plt.boxplot(data_to_plot, patch_artist=True, labels=labels, widths=0.6)
        
        # Customize boxplot colors
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add grid lines
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add labels
        plt.xlabel('Period', fontsize=16)
        plt.ylabel('USD/IDR Exchange Rate [IDR]', fontsize=16)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save figure
        plt.savefig(os.path.join(output_dir, filename), dpi=400, bbox_inches='tight')
        plt.close()
        
        print(f"Boxplot saved to {os.path.join(output_dir, filename)}")
    
    except Exception as e:
        print(f"Error creating boxplot: {e}")
        traceback.print_exc()


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


def run_normality_tests(pre_data, post_data, entire_data, output_dir="results/stats"):
    """
    Run comprehensive normality tests on the datasets.
    
    Parameters:
    pre_data : DataFrame with USD_IDR before inauguration
    post_data : DataFrame with USD_IDR after inauguration
    entire_data : DataFrame with USD_IDR for entire period
    output_dir : Directory to save the results
    
    Returns:
    results : Dictionary with test results for each period
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract values
    pre_values = pre_data['USD_IDR'].values
    post_values = post_data['USD_IDR'].values
    entire_values = entire_data['USD_IDR'].values
    
    periods = {
        'Pre-Inauguration': pre_values,
        'Post-Inauguration': post_values,
        'Entire Period': entire_values
    }
    
    # Store all results
    results = {}
    
    # Run tests for each period
    for period_name, data in periods.items():
        print(f"\nRunning normality tests for {period_name} data...")
        
        # Calculate robust descriptive statistics
        desc_stats = robust_descriptive_stats(data)
        
        # Run Shapiro-Wilk test
        try:
            sw_stat, sw_p = shapiro_wilk_test(data)
            sw_normal = sw_p > 0.05
        except Exception as e:
            print(f"  Error in Shapiro-Wilk test: {e}")
            sw_stat, sw_p, sw_normal = np.nan, np.nan, None
        
        # Run Anderson-Darling test
        try:
            ad_result = anderson_darling_test(data)
            ad_stat = ad_result.statistic
            ad_normal = all(ad_stat < ad_result.critical_values)
        except Exception as e:
            print(f"  Error in Anderson-Darling test: {e}")
            ad_stat, ad_normal = np.nan, None
        
        # Run Jarque-Bera test
        try:
            jb_stat, jb_p = jarque_bera_test(data)
            jb_normal = jb_p > 0.05
        except Exception as e:
            print(f"  Error in Jarque-Bera test: {e}")
            jb_stat, jb_p, jb_normal = np.nan, np.nan, None
        
        # Run D'Agostino-Pearson test
        try:
            dp_stat, dp_p = d_agostino_pearson_test(data)
            dp_normal = dp_p > 0.05
        except Exception as e:
            print(f"  Error in D'Agostino-Pearson test: {e}")
            dp_stat, dp_p, dp_normal = np.nan, np.nan, None
        
        # Run Lilliefors test
        try:
            lf_stat, lf_p = lilliefors_test(data)
            lf_normal = lf_p > 0.05
        except Exception as e:
            print(f"  Error in Lilliefors test: {e}")
            lf_stat, lf_p, lf_normal = np.nan, np.nan, None
        
        # Collect test results
        test_results = {
            'Shapiro-Wilk': {'statistic': sw_stat, 'p-value': sw_p, 'normal': sw_normal},
            'Anderson-Darling': {'statistic': ad_stat, 'normal': ad_normal},
            'Jarque-Bera': {'statistic': jb_stat, 'p-value': jb_p, 'normal': jb_normal},
            'D\'Agostino-Pearson': {'statistic': dp_stat, 'p-value': dp_p, 'normal': dp_normal},
            'Lilliefors': {'statistic': lf_stat, 'p-value': lf_p, 'normal': lf_normal}
        }
        
        # Combine all results for this period
        results[period_name] = {
            'descriptive_stats': desc_stats,
            'normality_tests': test_results
        }
        
        # Print summary
        print(f"  Summary for {period_name}:")
        print(f"  - Sample size: {desc_stats['count']}")
        print(f"  - Mean: {desc_stats['mean']:.2f}, Median: {desc_stats['median']:.2f}")
        print(f"  - Standard Deviation: {desc_stats['std']:.2f}, MAD: {desc_stats['mad']:.2f}")
        print(f"  - Skewness: {desc_stats['skewness']:.4f}, Kurtosis: {desc_stats['kurtosis']:.4f}")
        print(f"  - Normality test results:")
        for test_name, test_res in test_results.items():
            if 'p-value' in test_res:
                print(f"    - {test_name}: {'Normal' if test_res['normal'] else 'Non-normal'} (p={test_res['p-value']:.4f})")
            else:
                print(f"    - {test_name}: {'Normal' if test_res['normal'] else 'Non-normal'}")
    
    # Save results to CSV
    try:
        # Prepare data for test results DataFrame
        test_data = []
        for period_name, results_dict in results.items():
            for test_name, test_res in results_dict['normality_tests'].items():
                row = {'Period': period_name, 'Test': test_name}
                for key, value in test_res.items():
                    row[key] = value
                test_data.append(row)
        
        # Create and save test results DataFrame
        test_df = pd.DataFrame(test_data)
        test_csv = os.path.join(output_dir, "normality_test_results_usd_idr.csv")
        test_df.to_csv(test_csv, index=False)
        print(f"\nNormality test results saved to {test_csv}")
        
        # Prepare data for descriptive statistics DataFrame
        stats_data = []
        for period_name, results_dict in results.items():
            row = {'Period': period_name}
            row.update(results_dict['descriptive_stats'])
            stats_data.append(row)
        
        # Create and save descriptive statistics DataFrame
        stats_df = pd.DataFrame(stats_data)
        stats_csv = os.path.join(output_dir, "descriptive_stats_usd_idr.csv")
        stats_df.to_csv(stats_csv, index=False)
        print(f"Descriptive statistics saved to {stats_csv}")
        
    except Exception as e:
        print(f"Error saving results to CSV: {e}")
        traceback.print_exc()
    
    return results


def generate_report(results, inauguration_date, output_dir="results/stats"):
    """
    Generate a comprehensive report based on the normality test results
    
    Parameters:
    results : Dictionary with test results for each period
    inauguration_date : Datetime object representing the inauguration date
    output_dir : Directory to save the report
    """
    report_content = f"""# Robust Normality Tests for USD/IDR Exchange Rate
## Analysis of USD/IDR Exchange Rate Distribution Before and After Presidential Inauguration (January 20, 2025)

This report presents the results of robust normality tests on USD/IDR exchange rates before and after the presidential inauguration on {inauguration_date.strftime('%B %d, %Y')}.

### About Normality Tests
Normality is a key assumption in many statistical tests. Testing for normality helps determine whether parametric tests (which assume normal distribution) or non-parametric tests (which don't require normality) are more appropriate for the data. This analysis employs multiple robust normality tests:

1. **Shapiro-Wilk Test**: Effective for sample sizes up to 2000. The null hypothesis is that the data is normally distributed.

2. **Anderson-Darling Test**: Places more weight on the tails of the distribution, making it particularly sensitive to deviations from normality in the tails.

3. **Jarque-Bera Test**: Based on skewness and kurtosis, this test is particularly effective for detecting deviations from normality due to asymmetry or heavy tails.

4. **D'Agostino-Pearson Test**: Combines skewness and kurtosis to form an omnibus test of normality.

5. **Lilliefors Test**: A modification of the Kolmogorov-Smirnov test that is appropriate when the parameters of the normal distribution are estimated from the data.

## Results

"""

    # Add descriptive statistics to the report
    report_content += "### Descriptive Statistics\n\n"
    report_content += "| Period | Count | Mean | Median | 10% Trimmed Mean | 20% Trimmed Mean | Std Dev | MAD | IQR | Min | Max | Skewness | Kurtosis |\n"
    report_content += "|--------|-------|------|--------|-----------------|------------------|---------|-----|-----|-----|-----|----------|----------|\n"
    
    for period_name, results_dict in results.items():
        stats = results_dict['descriptive_stats']
        report_content += f"| {period_name} | {stats['count']} | {stats['mean']:.2f} | {stats['median']:.2f} | "
        report_content += f"{stats['trimmed_mean_10']:.2f} | {stats['trimmed_mean_20']:.2f} | {stats['std']:.2f} | "
        report_content += f"{stats['mad']:.2f} | {stats['iqr']:.2f} | {stats['min']:.2f} | {stats['max']:.2f} | "
        report_content += f"{stats['skewness']:.4f} | {stats['kurtosis']:.4f} |\n"
    
    report_content += "\n"
    
    # Add normality test results to the report
    report_content += "### Normality Test Results\n\n"
    for period_name, results_dict in results.items():
        report_content += f"#### {period_name}\n\n"
        report_content += "| Test | Statistic | p-value | Normal Distribution? |\n"
        report_content += "|------|-----------|---------|----------------------|\n"
        
        for test_name, test_res in results_dict['normality_tests'].items():
            statistic = f"{test_res['statistic']:.4f}" if 'statistic' in test_res and not np.isnan(test_res['statistic']) else "N/A"
            p_value = f"{test_res['p-value']:.4f}" if 'p-value' in test_res and not np.isnan(test_res['p-value']) else "N/A"
            normal = "Yes" if test_res['normal'] else "No" if test_res['normal'] is not None else "N/A"
            
            report_content += f"| {test_name} | {statistic} | {p_value} | {normal} |\n"
        
        report_content += "\n"
    
    # Add interpretation summary to the report
    report_content += """
## Interpretation Summary

The normality tests provide a basis for determining whether the USD/IDR exchange rate data follows a normal distribution in each period. This information is crucial for selecting appropriate statistical tests for further analysis:

- **If data is normally distributed**: Parametric tests like t-tests and ANOVA may be appropriate.
- **If data is not normally distributed**: Non-parametric tests like Mann-Whitney U, Kolmogorov-Smirnov, or Wilcoxon signed-rank tests are more appropriate.

Financial data, including exchange rates, often deviates from normality due to:
- Excess kurtosis (fat tails)
- Non-zero skewness (asymmetry)
- Volatility clustering

The robust descriptive statistics provide additional insights:
- **Comparison of mean and median**: Large differences indicate asymmetry in the distribution
- **Trimmed means**: Less sensitive to outliers than the regular mean
- **MAD and IQR**: Robust measures of dispersion, less sensitive to outliers than standard deviation
- **Skewness**: Measures the asymmetry of the distribution
- **Kurtosis**: Measures the heaviness of the tails of the distribution

## Conclusion

This comprehensive normality assessment helps guide the selection of appropriate statistical methods for analyzing USD/IDR exchange rates before and after the presidential inauguration. Based on these results, the appropriate tests for analyzing differences between the periods can be selected.

Given that financial time series data rarely follows a perfect normal distribution, the robust non-parametric tests performed in parallel analyses (Mann-Whitney U, Kolmogorov-Smirnov, and Cliff's Delta) are particularly valuable for reliable comparisons between periods.

_Note: Visualizations of these distributions are available in the accompanying KDE plot and boxplot figures._
"""
    
    try:
        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write the report to a file
        report_file = os.path.join(output_dir, "normality_test_report_usd_idr.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"\nNormality test report successfully written to {report_file}")
        
    except Exception as e:
        print(f"Error saving report: {e}")
        traceback.print_exc()


def main():
    """
    Main function to run robust normality tests and create visualizations
    for USD/IDR exchange rate before and after inauguration
    """
    # Define file paths
    full_period_file = "data/usd_idr_inauguration_period.csv"
    pre_inauguration_file = "data/usd_idr_pre_inauguration.csv"
    post_inauguration_file = "data/usd_idr_post_inauguration.csv"
    
    output_dir = "results"
    stats_dir = "results/stats"
    figures_dir = "results/figures"
    
    # Define inauguration date
    inauguration_date = datetime(2025, 1, 20)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        
        # Load the data
        print("Loading data...")
        
        entire_data = load_data(full_period_file)
        pre_data = load_data(pre_inauguration_file)
        post_data = load_data(post_inauguration_file)
        
        print(f"Data loaded successfully:")
        print(f"Full period: {len(entire_data)} records")
        print(f"Pre-inauguration: {len(pre_data)} records")
        print(f"Post-inauguration: {len(post_data)} records")
        
        # Run normality tests
        print("\nRunning robust normality tests...")
        results = run_normality_tests(pre_data, post_data, entire_data, output_dir=stats_dir)
        
        # Generate report
        print("\nGenerating normality test report...")
        generate_report(results, inauguration_date, output_dir=stats_dir)
        
        # Create visualizations
        print("\nCreating visualizations...")
        create_kde_plot(pre_data, post_data, entire_data, output_dir=figures_dir)
        create_boxplot(pre_data, post_data, entire_data, output_dir=figures_dir)
        
        print("\nRobust normality tests and visualizations for USD/IDR complete.")
        print(f"Results are saved in the directory: {output_dir}")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
