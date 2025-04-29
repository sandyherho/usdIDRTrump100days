#!/usr/bin/env python
"""
Data retrieval module for USD/IDR exchange rate analysis.
This module collects USD to IDR exchange rate data for the 100 days before and after inauguration.
"""

import os
import logging
import shutil
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_directories():
    """Remove existing data directories and recreate them."""
    dirs = ["data"]
    
    # Remove existing directories
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
            logger.info(f"Removed existing directory: {d}")
    
    # Create fresh directories
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    logger.info("Created fresh output directories.")

def collect_usd_idr_data(start_date, end_date):
    """
    Collect USD/IDR exchange rate data for the specified date range.
    
    Parameters:
    -----------
    start_date : datetime
        Start date for data collection
    end_date : datetime
        End date for data collection
        
    Returns:
    --------
    pandas.DataFrame
        USD_IDR data
    """
    logger.info(f"Collecting USD/IDR data from {start_date} to {end_date}")
    
    try:
        # Download USD/IDR data from Yahoo Finance
        logger.info("Downloading USD/IDR exchange rate data...")
        usd_idr_data = yf.download("IDR=X", start=start_date, end=end_date)
        
        # Check if data is empty
        if usd_idr_data.empty:
            logger.error("No USD/IDR data retrieved from Yahoo Finance API.")
            return pd.DataFrame()
        
        # Extract closing prices for USD/IDR
        if 'Close' in usd_idr_data.columns:
            usd_idr_close = usd_idr_data['Close']
        elif 'Adj Close' in usd_idr_data.columns:
            usd_idr_close = usd_idr_data['Adj Close']
        else:
            logger.error("No 'Close' or 'Adj Close' columns found in USD/IDR data.")
            return pd.DataFrame()
        
        # Create a new DataFrame with just USD/IDR data
        usd_idr_df = pd.DataFrame(usd_idr_close)
        usd_idr_df.columns = ['USD_IDR']
        
        logger.info(f"Successfully created USD/IDR dataset with {len(usd_idr_df)} data points")
        return usd_idr_df
        
    except Exception as e:
        logger.error(f"Error collecting USD/IDR data: {str(e)}")
        return pd.DataFrame()

def save_data(data, filename):
    """Save data to CSV file."""
    filepath = os.path.join("data", filename)
    data.to_csv(filepath)
    logger.info(f"Saved data to {filepath}")
    return filepath

def prepare_inauguration_event():
    """
    Prepare inauguration event date for the analysis.
    
    Returns:
    --------
    datetime
        Inauguration date
    """
    # Trump second term inauguration date: January 20, 2025
    inauguration_date = datetime(2025, 1, 20)
    logger.info(f"Using inauguration date: {inauguration_date}")
    
    return inauguration_date

def split_data_by_date(data, split_date):
    """
    Split data into before and after a specific date.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to split
    split_date : datetime
        Date to split on
        
    Returns:
    --------
    tuple
        (pre_data, post_data)
    """
    pre_data = data[data.index < split_date].copy()
    post_data = data[data.index >= split_date].copy()
    
    logger.info(f"Split data into {len(pre_data)} pre-event points and {len(post_data)} post-event points")
    
    return pre_data, post_data

def main():
    """Main function to execute data retrieval."""
    # Clean and recreate directories
    clean_directories()
    
    # Define analysis parameters
    inauguration_date = prepare_inauguration_event()
    
    # Analysis period: 100 days before and after inauguration
    start_date = inauguration_date - timedelta(days=100)
    end_date = inauguration_date + timedelta(days=100)
    
    # Collect USD/IDR data for the specified period
    logger.info("Collecting USD/IDR data for inauguration analysis period")
    usd_idr_data = collect_usd_idr_data(start_date, end_date)
    
    # Save the data
    if not usd_idr_data.empty:
        # Save the full period data
        save_data(usd_idr_data, "usd_idr_inauguration_period.csv")
        
        # Split data into pre and post inauguration
        pre_inauguration, post_inauguration = split_data_by_date(usd_idr_data, inauguration_date)
        save_data(pre_inauguration, "usd_idr_pre_inauguration.csv")
        save_data(post_inauguration, "usd_idr_post_inauguration.csv")
        logger.info("Successfully saved USD/IDR data for inauguration analysis")
    else:
        logger.error("Failed to collect USD/IDR data. Exiting.")
    
    logger.info("Data retrieval complete.")

if __name__ == "__main__":
    main()
