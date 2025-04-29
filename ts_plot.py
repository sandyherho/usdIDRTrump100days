#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
USD/IDR Exchange Rate Time Series Plot
-------------------------------------
Visualization of USD/IDR exchange rate data.

SHSH <sandy.herho@email.ucr.edu>
04/29/2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import os
from datetime import datetime

# Set file paths
data_file = "data/usd_idr_inauguration_period.csv"
output_dir = "results/figures"
output_file = "usd_idr_exchange_rate.png"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the data
df = pd.read_csv(data_file)

# Ensure date column is properly formatted
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
elif 'date' in df.columns:
    df['Date'] = pd.to_datetime(df['date'])
    
# Sort by date
df = df.sort_values('Date')

# Define inauguration date
inauguration_date = datetime(2025, 1, 20)

# Create the plot
plt.figure(figsize=(12, 7))

# Plot data
plt.plot(df['Date'], df['USD_IDR'], color='#1F77B4', linewidth=2, label='USD/IDR Exchange Rate')

# Add vertical line for inauguration date
plt.axvline(x=inauguration_date, color='red', linestyle='--', linewidth=1.5, 
            label=f'Inauguration: {inauguration_date.strftime("%Y-%m-%d")}')

# Format axes
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Date', fontsize=16)
plt.ylabel('USD/IDR Exchange Rate [IDR]', fontsize=16)

# Format x-axis dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.xticks(rotation=45)

# Format y-axis with thousand separators
plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))

# Add legend
plt.legend(loc='best')

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(output_dir, output_file), dpi=400, bbox_inches='tight')
plt.close()

print(f"Time series plot saved to {os.path.join(output_dir, output_file)}")
