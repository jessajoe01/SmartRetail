import pandas as pd
import numpy as np


def detect_date_format(date_series, sample_size=100):
    """Analyzes date column to detect format"""
    sample = date_series.dropna().head(sample_size)
    
    if len(sample) == 0:
        return "No valid dates found"
    
    formats_found = {
        'YYYY-MM-DD': 0, 'DD-MM-YYYY': 0, 'MM-DD-YYYY': 0,
        'DD/MM/YYYY': 0, 'MM/DD/YYYY': 0, 'Other': 0
    }
    
    for date_str in sample.astype(str):
        if '-' in date_str:
            parts = date_str.split('-')
            if len(parts) == 3:
                if len(parts[0]) == 4:
                    formats_found['YYYY-MM-DD'] += 1
                elif int(parts[0]) <= 12:
                    formats_found['MM-DD-YYYY'] += 1
                else:
                    formats_found['DD-MM-YYYY'] += 1
        elif '/' in date_str:
            parts = date_str.split('/')
            if len(parts) == 3:
                if int(parts[0]) <= 12:
                    formats_found['MM/DD/YYYY'] += 1
                else:
                    formats_found['DD/MM/YYYY'] += 1
        else:
            formats_found['Other'] += 1
    
    most_common = max(formats_found, key=formats_found.get)
    
    return {
        'most_common_format': most_common,
        'format_counts': formats_found,
        'sample_dates': sample.head(5).tolist()
    }


def parse_dates_robustly(date_series):
    """Tries multiple date formats to minimize data loss"""
    
    date_formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
        '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%d-%b-%Y', '%d %b %Y',
        '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S',
    ]
    
    parsed_dates = pd.Series([pd.NaT] * len(date_series), index=date_series.index)
    remaining_mask = pd.isna(parsed_dates)
    
    # Try pandas automatic parsing first
    try:
        parsed_dates = pd.to_datetime(date_series, format='mixed', dayfirst=True, errors='coerce')
        remaining_mask = pd.isna(parsed_dates)
        
        if remaining_mask.sum() == 0:
            return parsed_dates
    except:
        pass
    
    # Try each format for remaining unparsed dates
    for fmt in date_formats:
        if remaining_mask.sum() == 0:
            break
        
        try:
            parsed_subset = pd.to_datetime(
                date_series[remaining_mask], 
                format=fmt, 
                errors='coerce'
            )
            
            successfully_parsed = ~pd.isna(parsed_subset)
            parsed_dates[remaining_mask] = parsed_subset
            remaining_mask = pd.isna(parsed_dates)
            
        except:
            continue
    
    return parsed_dates


def clean_dataset(df):
    """Cleans dataset by removing duplicates and filling missing values"""
    df = df.dropna(how='all').copy()
    df = df.drop_duplicates().copy()
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
    
    return df


def map_columns(df):
    """Smart column detection - maps user columns to standard names"""
    
    column_patterns = {
        'Order Date': ['order_date', 'date', 'order date', 'orderdate', 'transaction_date', 
                      'sale_date', 'sales_date', 'purchase_date', 'invoice_date'],
        'Sales': ['sales', 'revenue', 'amount', 'total', 'sale_amount', 'total_sales',
                 'sale', 'sales_amount', 'net_sales', 'gross_sales'],
        'Quantity': ['quantity', 'qty', 'units', 'units_sold', 'quantity_sold'],
        'Product': ['product', 'product_name', 'product name', 'item', 'item_name', 
                   'product_id', 'sku', 'product_code'],
        'Category': ['category', 'product_category', 'cat', 'product category', 'type'],
        'Sub Category': ['sub_category', 'subcategory', 'sub category', 'subcat'],
        'Profit': ['profit', 'net_profit', 'profit_amount', 'net profit', 'profits'],
        'Discount': ['discount', 'discount_percent', 'discount_rate', 'disc', 'discount%',
                    'discount_percentage'],
        'Region': ['region', 'area', 'zone', 'territory'],
        'State': ['state', 'province', 'location'],
        'Customer ID': ['customer_id', 'customer id', 'customerid', 'cust_id']
    }
    
    column_mapping = {}
    
    for standard_name, variations in column_patterns.items():
        for col in df.columns:
            normalized_col = col.lower().strip().replace('_', ' ').replace('-', ' ')
            
            if normalized_col in variations or col == standard_name:
                column_mapping[col] = standard_name
                break
    
    df = df.rename(columns=column_mapping)
    
    return df


def validate_columns(df):
    """Validates that required columns exist"""
    required = ['Order Date', 'Sales']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        raise ValueError(
            f"❌ Dataset must contain columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Please ensure your dataset has at least a date column and a sales/revenue column."
        )


def create_product_level_aggregation(df):
    """
    Creates product-wise monthly aggregation
    THIS IS THE KEY CHANGE - Now tracks each product separately!
    """
    
    # Parse dates
    print("📅 Parsing dates...")
    df['Order Date'] = parse_dates_robustly(df['Order Date'])
    
    invalid_dates = df['Order Date'].isna().sum()
    total_rows = len(df)
    
    if invalid_dates > 0:
        print(f"⚠️ Warning: {invalid_dates}/{total_rows} dates couldn't be parsed ({invalid_dates/total_rows*100:.1f}%)")
    
    df = df.dropna(subset=['Order Date'])
    
    if len(df) == 0:
        raise ValueError(
            "❌ All dates failed to parse! Please check your date column format.\n"
            "Supported formats: YYYY-MM-DD, DD-MM-YYYY, MM-DD-YYYY, DD/MM/YYYY, etc."
        )
    
    print(f"✅ Successfully parsed {len(df)}/{total_rows} rows ({len(df)/total_rows*100:.1f}%)")
    
    # Create month start date
    df['MonthStart'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()
    
    # Check if we have product information
    has_product = 'Product' in df.columns
    has_category = 'Category' in df.columns
    
    # Build groupby keys
    group_keys = ['MonthStart']
    
    if has_product:
        group_keys.append('Product')
    
    if has_category:
        group_keys.append('Category')
    
    # Build aggregation dictionary
    agg_dict = {'Sales': 'sum'}
    
    if 'Quantity' in df.columns:
        agg_dict['Quantity'] = 'sum'
    
    if 'Profit' in df.columns:
        agg_dict['Profit'] = 'sum'
    
    if 'Discount' in df.columns:
        agg_dict['Discount'] = 'mean'
    
    # Aggregate
    if len(group_keys) > 1:
        # Product-level aggregation
        product_agg = df.groupby(group_keys).agg(agg_dict).reset_index()
        product_agg = product_agg.rename(columns={'MonthStart': 'ds'})
        
        # Also create total aggregation for overall forecasting
        total_agg = df.groupby('MonthStart').agg(agg_dict).reset_index()
        total_agg = total_agg.rename(columns={'MonthStart': 'ds'})
        
        return {
            'product_level': product_agg,
            'total_level': total_agg,
            'has_products': True
        }
    else:
        # Only total aggregation
        monthly_agg = df.groupby('MonthStart').agg(agg_dict).reset_index()
        monthly_agg = monthly_agg.rename(columns={'MonthStart': 'ds'})
        
        return {
            'product_level': None,
            'total_level': monthly_agg,
            'has_products': False
        }


def add_monthly_features(monthly_agg, group_col=None):
    """
    Adds time-based, lag, and rolling features
    Can work on product-level or total-level data
    """
    
    # Sort by date (and product if applicable)
    sort_cols = ['ds']
    if group_col and group_col in monthly_agg.columns:
        sort_cols.insert(0, group_col)
    
    monthly_agg = monthly_agg.sort_values(sort_cols).reset_index(drop=True)
    
    # Time features
    monthly_agg['year'] = monthly_agg['ds'].dt.year
    monthly_agg['month'] = monthly_agg['ds'].dt.month
    monthly_agg['quarter'] = monthly_agg['ds'].dt.quarter
    
    # Lag and rolling features (grouped by product if applicable)
    if group_col and group_col in monthly_agg.columns:
        # Product-wise features
        for product in monthly_agg[group_col].unique():
            mask = monthly_agg[group_col] == product
            product_sales = monthly_agg.loc[mask, 'Sales']
            
            monthly_agg.loc[mask, 'lag_1'] = product_sales.shift(1)
            monthly_agg.loc[mask, 'lag_2'] = product_sales.shift(2)
            monthly_agg.loc[mask, 'lag_3'] = product_sales.shift(3)
            monthly_agg.loc[mask, 'lag_6'] = product_sales.shift(6)
            monthly_agg.loc[mask, 'lag_12'] = product_sales.shift(12)
            
            monthly_agg.loc[mask, 'ma_3'] = product_sales.rolling(3).mean()
            monthly_agg.loc[mask, 'ma_6'] = product_sales.rolling(6).mean()
    else:
        # Total-level features
        monthly_agg['lag_1'] = monthly_agg['Sales'].shift(1)
        monthly_agg['lag_2'] = monthly_agg['Sales'].shift(2)
        monthly_agg['lag_3'] = monthly_agg['Sales'].shift(3)
        monthly_agg['lag_6'] = monthly_agg['Sales'].shift(6)
        monthly_agg['lag_12'] = monthly_agg['Sales'].shift(12)
        
        monthly_agg['ma_3'] = monthly_agg['Sales'].rolling(3).mean()
        monthly_agg['ma_6'] = monthly_agg['Sales'].rolling(6).mean()
    
    # Fill NaNs with forward fill
    monthly_agg = monthly_agg.ffill()
    
    # Remove rows with NaNs
    monthly_agg = monthly_agg.dropna().reset_index(drop=True)
    
    return monthly_agg


def create_timeline_aggregation(df, timeline='monthly'):
    """
    UNIFIED AGGREGATION FUNCTION - Works for all timelines
    
    Parameters:
    df: DataFrame - must have 'Order Date' and 'Sales' columns
    timeline: str - 'daily', 'weekly', 'monthly', 'yearly'
    
    Returns:
    dict with product_level and total_level aggregations
    """
    
    # Parse dates first
    print(f"📅 Parsing dates for {timeline} aggregation...")
    df['Order Date'] = parse_dates_robustly(df['Order Date'])
    
    invalid_dates = df['Order Date'].isna().sum()
    total_rows = len(df)
    
    if invalid_dates > 0:
        print(f"⚠️ Warning: {invalid_dates}/{total_rows} dates couldn't be parsed ({invalid_dates/total_rows*100:.1f}%)")
    
    df = df.dropna(subset=['Order Date'])
    
    if len(df) == 0:
        raise ValueError(
            "❌ All dates failed to parse! Please check your date column format.\n"
            "Supported formats: YYYY-MM-DD, DD-MM-YYYY, MM-DD-YYYY, DD/MM/YYYY, etc."
        )
    
    print(f"✅ Successfully parsed {len(df)}/{total_rows} rows ({len(df)/total_rows*100:.1f}%)")
    
    # Create period column based on timeline
    if timeline == 'daily':
        df['Period'] = df['Order Date'].dt.floor('D')
    elif timeline == 'weekly':
        df['Period'] = df['Order Date'].dt.to_period('W').dt.to_timestamp()
    elif timeline == 'monthly':
        df['Period'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()
    elif timeline == 'yearly':
        df['Period'] = df['Order Date'].dt.to_period('Y').dt.to_timestamp()
    else:
        raise ValueError(f"Invalid timeline: {timeline}. Must be 'daily', 'weekly', 'monthly', or 'yearly'")
    
    # Check for product information
    has_product = 'Product' in df.columns
    has_category = 'Category' in df.columns
    
    # Build groupby keys
    group_keys = ['Period']
    
    if has_product:
        group_keys.append('Product')
    
    if has_category:
        group_keys.append('Category')
    
    # Build aggregation dictionary
    agg_dict = {'Sales': 'sum'}
    
    if 'Quantity' in df.columns:
        agg_dict['Quantity'] = 'sum'
    
    if 'Profit' in df.columns:
        agg_dict['Profit'] = 'sum'
    
    if 'Discount' in df.columns:
        agg_dict['Discount'] = 'mean'
    
    # Aggregate data
    if len(group_keys) > 1:
        # Product-level aggregation
        product_agg = df.groupby(group_keys).agg(agg_dict).reset_index()
        product_agg = product_agg.rename(columns={'Period': 'ds'})
        
        # Total-level aggregation
        total_agg = df.groupby('Period').agg(agg_dict).reset_index()
        total_agg = total_agg.rename(columns={'Period': 'ds'})
        
        return {
            'product_level': product_agg,
            'total_level': total_agg,
            'has_products': True,
            'timeline': timeline
        }
    else:
        # Only total aggregation (no product info)
        total_agg = df.groupby('Period').agg(agg_dict).reset_index()
        total_agg = total_agg.rename(columns={'Period': 'ds'})
        
        return {
            'product_level': None,
            'total_level': total_agg,
            'has_products': False,
            'timeline': timeline
        }


def add_timeline_features(agg_data, timeline='monthly', group_col=None):
    """
    Adds features appropriate for the selected timeline
    
    Parameters:
    agg_data: DataFrame - aggregated data
    timeline: str - 'daily', 'weekly', 'monthly', 'yearly'
    group_col: str - optional grouping column (e.g., 'Product')
    """
    
    # Sort data
    sort_cols = ['ds']
    if group_col and group_col in agg_data.columns:
        sort_cols.insert(0, group_col)
    
    agg_data = agg_data.sort_values(sort_cols).reset_index(drop=True)
    
    # Time features (common to all timelines)
    agg_data['year'] = agg_data['ds'].dt.year
    agg_data['month'] = agg_data['ds'].dt.month
    agg_data['quarter'] = agg_data['ds'].dt.quarter
    
    # Timeline-specific features
    if timeline == 'daily':
        agg_data['day_of_week'] = agg_data['ds'].dt.dayofweek
        agg_data['day_of_month'] = agg_data['ds'].dt.day
        agg_data['is_weekend'] = agg_data['day_of_week'].isin([5, 6]).astype(int)
    
    elif timeline == 'weekly':
        agg_data['week_of_year'] = agg_data['ds'].dt.isocalendar().week
    
    # Lag configuration by timeline
    lag_config = {
        'daily': [1, 7, 14, 30],      # yesterday, week ago, 2 weeks, month
        'weekly': [1, 4, 8, 12],       # last week, month, 2 months, quarter
        'monthly': [1, 3, 6, 12],      # last month, quarter, half-year, year
        'yearly': [1, 2, 3]            # 1, 2, 3 years ago
    }
    
    rolling_config = {
        'daily': [7, 30],     # week, month
        'weekly': [4, 12],    # month, quarter
        'monthly': [3, 6],    # quarter, half-year
        'yearly': [2, 3]      # 2-3 years
    }
    
    lags = lag_config.get(timeline, [1, 3, 6, 12])
    rolling_windows = rolling_config.get(timeline, [3, 6])
    
    # Add lag and rolling features
    if group_col and group_col in agg_data.columns:
        # Product-wise features
        for product in agg_data[group_col].unique():
            mask = agg_data[group_col] == product
            product_sales = agg_data.loc[mask, 'Sales']
            
            # Lags
            for lag in lags:
                agg_data.loc[mask, f'lag_{lag}'] = product_sales.shift(lag)
            
            # Rolling averages
            for window in rolling_windows:
                agg_data.loc[mask, f'ma_{window}'] = product_sales.rolling(window).mean()
    else:
        # Total-level features
        # Lags
        for lag in lags:
            agg_data[f'lag_{lag}'] = agg_data['Sales'].shift(lag)
        
        # Rolling averages
        for window in rolling_windows:
            agg_data[f'ma_{window}'] = agg_data['Sales'].rolling(window).mean()
    
    # Fill NaNs
    agg_data = agg_data.ffill()
    agg_data = agg_data.dropna().reset_index(drop=True)
    
    return agg_data


def get_top_products(df, n=10):
    """Returns top N products by sales"""
    if 'Product' not in df.columns:
        return []
    
    top_products = df.groupby('Product')['Sales'].sum().nlargest(n)
    return top_products.index.tolist()


def get_product_statistics(df):
    """Generates product-level statistics"""
    
    if 'Product' not in df.columns:
        return None
    
    stats = df.groupby('Product').agg({
        'Sales': ['sum', 'mean', 'count'],
        'Profit': 'sum' if 'Profit' in df.columns else 'count',
        'Quantity': 'sum' if 'Quantity' in df.columns else 'count'
    }).round(2)
    
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.sort_values('Sales_sum', ascending=False)
    
    return stats.reset_index()