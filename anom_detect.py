#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import os
from datetime import datetime
import matplotlib as mpl

# Configuration
DATA_PATH = './data/usd_idr_post_inauguration.csv'
OUTPUT_DIR = './results'

# Enhanced color scheme
COLORS = {
    'background': '#f9f9f9',
    'grid': '#e0e0e0',
    'line': '#2c3e50',
    'anomaly': '#e74c3c',
    'score': '#8e44ad',
    'threshold': '#c0392b',
    'highlight': '#f39c12'
}

def create_directories():
    """Ensure output directories exist"""
    os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/stats', exist_ok=True)

def load_and_preprocess_data():
    """Load and prepare time series data"""
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    
    # Create enhanced time-series features
    df['Returns'] = df['USD_IDR'].pct_change().fillna(0)
    df['Log_Returns'] = np.log(df['USD_IDR'] / df['USD_IDR'].shift(1)).fillna(0)
    
    # Multiple rolling statistics
    df['Rolling_Mean_7'] = df['USD_IDR'].rolling(window=7, min_periods=1).mean()
    df['Rolling_Std_7'] = df['USD_IDR'].rolling(window=7, min_periods=1).std()
    df['Rolling_Mean_14'] = df['USD_IDR'].rolling(window=14, min_periods=1).mean()
    df['Rolling_Std_14'] = df['USD_IDR'].rolling(window=14, min_periods=1).std()
    
    # Temporal features
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfMonth'] = df.index.day
    
    return df

def ensemble_anomaly_detection(df):
    """Use multiple methods and ensemble the results for robust detection"""
    # Features for anomaly detection
    features = ['USD_IDR', 'Returns', 'Log_Returns', 'Rolling_Mean_7', 
                'Rolling_Std_7', 'Rolling_Mean_14', 'Month', 'DayOfWeek']
    
    # Prepare dataset
    X = df[features].fillna(df[features].mean())
    
    # Method 1: Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        bootstrap=True,
        random_state=42,
        max_features=len(features)
    )
    df['IF_Anomaly'] = iso_forest.fit_predict(X)
    df['IF_Score'] = -iso_forest.score_samples(X)
    
    # Method 2: One-Class SVM with nonlinear kernel
    ocsvm = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=0.05
    )
    df['OCSVM_Anomaly'] = ocsvm.fit_predict(X)
    df['OCSVM_Score'] = -ocsvm.decision_function(X)
    
    # Method 3: Statistical approach (robust outlier detection based on IQR)
    Q1 = df['Returns'].quantile(0.25)
    Q3 = df['Returns'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df['Stats_Anomaly'] = ((df['Returns'] < lower_bound) | (df['Returns'] > upper_bound)).astype(int) * (-2) + 1
    df['Stats_Score'] = abs((df['Returns'] - df['Returns'].median()) / IQR)
    
    # Ensemble voting (consensus approach)
    anomaly_cols = ['IF_Anomaly', 'OCSVM_Anomaly', 'Stats_Anomaly']
    # Count methods that agree there's an anomaly
    df['Anomaly_Votes'] = (df[anomaly_cols] == -1).sum(axis=1)
    # Combine scores with weighted average
    df['Ensemble_Score'] = (df['IF_Score'] * 0.4 + 
                           df['OCSVM_Score'] * 0.4 + 
                           df['Stats_Score'] * 0.2)
    
    # Final decision - anomaly if at least 2 methods agree
    df['Anomaly'] = (df['Anomaly_Votes'] >= 2).astype(int) * (-2) + 1
    
    # Normalize ensemble score to 0-1 range for interpretability
    min_score = df['Ensemble_Score'].min()
    max_score = df['Ensemble_Score'].max()
    if max_score > min_score:  # Prevent division by zero
        df['Anomaly_Score'] = (df['Ensemble_Score'] - min_score) / (max_score - min_score)
    else:
        df['Anomaly_Score'] = df['Ensemble_Score'] * 0  # All zeros if no range
    
    return df

def generate_visualization(df):
    """Create cleaner 2x1 plots with focus on anomalies and scores with improved aesthetics"""
    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with custom background
    fig = plt.figure(figsize=(15, 10), facecolor=COLORS['background'])
    
    # Create two subplots
    ax1 = plt.subplot(2, 1, 1)  # Exchange rate
    ax2 = plt.subplot(2, 1, 2)  # Anomaly scores
    
    # Set background color
    for ax in [ax1, ax2]:
        ax.set_facecolor(COLORS['background'])
        ax.grid(True, color=COLORS['grid'], linestyle='-', linewidth=0.5, alpha=0.7)
    
    # Plot 1: USD/IDR rate with only anomalies highlighted
    df['USD_IDR'].plot(ax=ax1, color=COLORS['line'], linewidth=2.5, label='USD/IDR Rate')
    
    # Highlight anomalies without date annotations
    anomalies = df[df['Anomaly'] == -1]
    ax1.scatter(anomalies.index, anomalies['USD_IDR'], 
               color=COLORS['anomaly'], s=100, zorder=5, label='Anomalies', 
               marker='o', edgecolor='white', linewidth=1.5)
    
    # Cleaner formatting for plot 1 - REMOVED date labels from top plot
    ax1.set_ylabel('Exchange Rate [IDR per USD]', fontsize=16, fontweight='bold')
    # Set the legend with 'best' option
    ax1.legend(loc='best', frameon=True, framealpha=0.9, fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    
    # Hide x-axis tick labels for the top plot
    ax1.set_xticklabels([])
    ax1.set_xlabel(' ')
    
    # Plot 2: Only ensemble score and threshold with improved visuals
    df['Ensemble_Score'].plot(ax=ax2, color=COLORS['score'], linewidth=2.5, 
                             label='Anomaly Score')
    
    # Highlight anomaly threshold with improved visuals
    threshold = df['Ensemble_Score'].quantile(0.95)
    ax2.axhline(threshold, color=COLORS['threshold'], linestyle='--', 
               linewidth=2, alpha=0.8, label='Anomaly Threshold')
    
    # Mark dates with anomalies with more subtle vertical lines
    for date in anomalies.index:
        ax2.axvline(date, color=COLORS['anomaly'], alpha=0.15, linewidth=1.5)
    
    # Formatting for plot 2
    ax2.set_xlabel('Date', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Anomaly Score', fontsize=16, fontweight='bold')
    ax2.legend(loc='best', frameon=True, framealpha=0.9, fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    
    # Date formatting for only the bottom axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=40, ha='right')
    
    # Add padding between plots
    plt.subplots_adjust(hspace=0.15)
    
    # Add a border to the entire figure
    for spine in ax1.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1)
    
    for spine in ax2.spines.values():
        spine.set_edgecolor('#cccccc')
        spine.set_linewidth(1)
    
    # Use standard naming convention for the output file
    plt.tight_layout(pad=3.0)
    plt.savefig(f'{OUTPUT_DIR}/figures/usd_idr_post_inauguration_anomalies.png', dpi=400, bbox_inches='tight')
    plt.close()

def generate_report(df):
    """Create detailed statistical report"""
    anomalies = df[df['Anomaly'] == -1]
    
    # Calculate additional statistics
    method_agreement = {
        '1 method': len(df[df['Anomaly_Votes'] == 1]),
        '2 methods': len(df[df['Anomaly_Votes'] == 2]),
        '3 methods': len(df[df['Anomaly_Votes'] == 3]),
    }
    
    # Method comparison
    method_stats = {}
    for method in ['IF_Anomaly', 'OCSVM_Anomaly', 'Stats_Anomaly']:
        method_name = method.replace('_Anomaly', '')
        anomaly_count = len(df[df[method] == -1])
        method_stats[method_name] = {
            'anomalies': anomaly_count,
            'percentage': anomaly_count / len(df) * 100
        }
    
    # Time-based clustering of anomalies (find consecutive anomalies)
    df['anomaly_group'] = (df['Anomaly'].diff() != 0).cumsum()
    anomaly_clusters = df[df['Anomaly'] == -1].groupby('anomaly_group')
    cluster_stats = {
        'total_clusters': len(anomaly_clusters),
        'largest_cluster': max([len(group) for _, group in anomaly_clusters]) if len(anomaly_clusters) > 0 else 0,
        'avg_cluster_size': np.mean([len(group) for _, group in anomaly_clusters]) if len(anomaly_clusters) > 0 else 0
    }
    
    report = f"""USD/IDR Post-Inauguration Exchange Rate Anomaly Analysis
===========================================================
Analysis Date: {datetime.now().strftime('%Y-%m-%d')}
Analysis Period: {df.index.min().date()} to {df.index.max().date()}

Dataset Statistics:
------------------
- Total Observations: {len(df):,}
- Mean Exchange Rate: {df['USD_IDR'].mean():.2f}
- Maximum Rate: {df['USD_IDR'].max():.2f} (on {df['USD_IDR'].idxmax().date()})
- Minimum Rate: {df['USD_IDR'].min():.2f} (on {df['USD_IDR'].idxmin().date()})
- Overall Volatility (Std Dev): {df['USD_IDR'].std():.2f}
- Daily Average Return: {df['Returns'].mean()*100:.4f}%

Anomaly Detection Results:
-------------------------
- Total Anomalies Detected (Ensemble): {len(anomalies)} ({len(anomalies)/len(df):.2%})
- Anomaly Detection Methods Agreement:
  - {method_agreement['1 method']} points detected by only 1 method
  - {method_agreement['2 methods']} points detected by 2 methods
  - {method_agreement['3 methods']} points detected by all 3 methods

Individual Method Performance:
----------------------------
"""

    # Add individual method stats
    for method, stats in method_stats.items():
        report += f"- {method}: {stats['anomalies']} anomalies ({stats['percentage']:.2f}%)\n"
    
    report += f"""
Anomaly Clustering Analysis:
--------------------------
- Number of distinct anomaly clusters: {cluster_stats['total_clusters']}
- Largest consecutive anomaly sequence: {cluster_stats['largest_cluster']} days
- Average anomaly cluster size: {cluster_stats['avg_cluster_size']:.2f} days

Top 5 Most Significant Anomalies:
-------------------------------
"""
    
    # Add top anomalies by score
    top_anomalies = anomalies.nlargest(5, 'Anomaly_Score')
    for idx, (date, row) in enumerate(top_anomalies.iterrows(), 1):
        report += f"{idx}. Date: {date.date()}, Rate: {row['USD_IDR']:.2f}, "
        report += f"Score: {row['Anomaly_Score']:.4f}, "
        report += f"Methods Agreement: {row['Anomaly_Votes']}/3\n"
    
    report += """
Recommendations:
--------------
1. Investigate key policy announcements or economic events coinciding with detected anomalies
2. Perform cross-correlation analysis with major economic indicators around anomaly clusters
3. Consider market sentiment analysis during periods of method consensus on anomalies
4. Analyze liquidity conditions during detected anomaly periods
5. Compare detected anomalies with other Southeast Asian currencies to identify regional vs. IDR-specific patterns
6. Examine the relationship between anomaly clusters and changes in monetary policy
"""
    
    print(report)
    with open(f'{OUTPUT_DIR}/stats/post_inauguration_anomaly_report.txt', 'w') as f:
        f.write(report)

def save_results(df):
    """Persist results to CSV with additional metrics"""
    # Save full results
    results_columns = ['USD_IDR', 'Returns', 'Rolling_Mean_7', 'Rolling_Mean_14',
                      'Anomaly', 'Anomaly_Score', 'Anomaly_Votes',
                      'IF_Score', 'OCSVM_Score', 'Stats_Score']
    
    df[results_columns].to_csv(f'{OUTPUT_DIR}/post_inauguration_anomaly_results.csv')
    
    # Save detected anomalies with details
    anomaly_columns = results_columns + ['IF_Anomaly', 'OCSVM_Anomaly', 'Stats_Anomaly']
    anomalies = df[df['Anomaly'] == -1][anomaly_columns]
    anomalies.to_csv(f'{OUTPUT_DIR}/post_inauguration_detected_anomalies.csv')
    
    # Create a method comparison summary
    method_comparison = pd.DataFrame({
        'Isolation_Forest': (df['IF_Anomaly'] == -1).astype(int),
        'OCSVM': (df['OCSVM_Anomaly'] == -1).astype(int),
        'Statistical': (df['Stats_Anomaly'] == -1).astype(int),
        'Ensemble': (df['Anomaly'] == -1).astype(int)
    })
    method_comparison.index = df.index
    method_comparison.to_csv(f'{OUTPUT_DIR}/method_comparison_summary.csv')

def main():
    """Main workflow with enhanced robustness"""
    try:
        print("Starting robust anomaly detection analysis...")
        create_directories()
        
        # Load and prepare data
        print("Loading and preprocessing data...")
        df = load_and_preprocess_data()
        
        # Apply ensemble anomaly detection
        print("Applying ensemble nonparametric anomaly detection...")
        df = ensemble_anomaly_detection(df)
        
        # Generate visualizations and reports
        print("Generating visualizations...")
        generate_visualization(df)
        
        print("Generating statistical report...")
        generate_report(df)
        
        print("Saving detailed results...")
        save_results(df)
        
        print("Analysis complete! Results saved to:", OUTPUT_DIR)
    except Exception as e:
        print(f"Error in analysis: {e}")
        raise

if __name__ == "__main__":
    main()
