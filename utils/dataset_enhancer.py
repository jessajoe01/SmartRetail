"""
Automatically enhance your dataset with missing columns
Run this once to create enhanced dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def enhance_retail_dataset(input_file='sales_dataset.csv', output_file='enhanced_sales_data.csv'):
    """
    Enhance the retail dataset with all missing columns for complete analytics
    """
    print("📊 Enhancing dataset for Smart Retail Analytics...")
    
    # Load your original data
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # 1. Ensure Order Date is datetime
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
    
    # 2. Add Product Details
    print("Adding product details...")
    df['Product_ID'] = 'PROD_' + df['Order ID'].str.extract(r'(\d+)').fillna('000')
    df['Product_Name'] = df['Sub Category'] + ' - ' + df['Category']
    
    # Generate realistic quantities (1-20 units)
    np.random.seed(42)
    df['Quantity'] = np.random.randint(1, 20, size=len(df))
    
    # Calculate unit price (Sales = Unit Price × Quantity × (1 - Discount))
    df['Unit_Price'] = df['Sales'] / (df['Quantity'] * (1 - df['Discount']))
    
    # Cost price (60-70% of unit price)
    df['Cost_Price'] = df['Unit_Price'] * np.random.uniform(0.6, 0.7, size=len(df))
    
    # Add brands based on category
    brand_mapping = {
        'Oil & Masala': ['MDH', 'Everest', 'Catch', 'Badshah'],
        'Beverages': ['Coca-Cola', 'Pepsi', 'Red Bull', 'Tropicana'],
        'Food Grains': ['Fortune', 'Ashirvaad', 'Daawat', 'Tata'],
        'Fruits & Veggies': ['Farm Fresh', 'Organic Valley', 'Local Farm'],
        'Snacks': ['Lays', 'Kurkure', 'Bingo', 'Haldirams'],
        'Bakery': ['Britannia', 'Modern', 'Harvest Gold', 'Priya Gold'],
        'Eggs, Meat & Fish': ['Suguna', 'Venky\'s', 'Licious', 'Fresh Catch']
    }
    
    def get_brand(category):
        if category in brand_mapping:
            return np.random.choice(brand_mapping[category])
        return 'Generic'
    
    df['Brand'] = df['Category'].apply(get_brand)
    
    # 3. Add Inventory Information
    print("Adding inventory details...")
    
    # Current stock levels (5-200 units)
    df['Current_Stock'] = np.random.randint(5, 200, size=len(df))
    
    # Calculate reorder point (20-30% of average monthly sales per product)
    # First, calculate average daily sales per product
    product_avg_sales = df.groupby('Product_ID')['Quantity'].mean()
    df['Reorder_Point'] = df['Product_ID'].map(product_avg_sales) * 30 * 0.25  # 25% of monthly avg
    
    # Lead time (3-14 days)
    df['Lead_Time_Days'] = np.random.choice([3, 5, 7, 10, 14], size=len(df), p=[0.2, 0.3, 0.25, 0.15, 0.1])
    
    # Supplier info
    df['Supplier_ID'] = 'SUP_' + pd.Series(np.random.randint(100, 999, size=len(df))).astype(str)
    
    # Warehouse based on region
    warehouse_map = {'North': 'WH_NORTH', 'South': 'WH_SOUTH', 'East': 'WH_EAST', 
                    'West': 'WH_WEST', 'Central': 'WH_CENTRAL'}
    df['Warehouse'] = df['Region'].map(warehouse_map)
    
    # 4. Add Customer Analytics
    print("Adding customer analytics...")
    
    # Customer ID
    df['Customer_ID'] = 'CUST_' + df['Order ID'].str.extract(r'(\d+)').fillna('000')
    
    # Customer segments
    segments = ['VIP', 'Loyal', 'Regular', 'New', 'Occasional']
    segment_probs = [0.05, 0.15, 0.5, 0.2, 0.1]
    df['Customer_Segment'] = np.random.choice(segments, size=len(df), p=segment_probs)
    
    # Loyalty points (0-5000)
    df['Loyalty_Points'] = np.random.randint(0, 5000, size=len(df))
    
    # Customer since date (random date before order)
    min_date = df['Order Date'].min()
    df['Customer_Since'] = min_date - pd.to_timedelta(
        np.random.randint(0, 730, size=len(df)), unit='D'
    )
    
    # Demographics
    df['Age_Group'] = np.random.choice(
        ['18-25', '26-35', '36-50', '51+'], 
        size=len(df), 
        p=[0.2, 0.4, 0.3, 0.1]
    )
    
    df['Gender'] = np.random.choice(
        ['Male', 'Female', 'Other'], 
        size=len(df), 
        p=[0.55, 0.42, 0.03]
    )
    
    # 5. Add GST Information
    print("Adding GST details...")
    
    gst_mapping = {
        'Oil & Masala': '5%',
        'Beverages': '12%',
        'Food Grains': '0%',
        'Fruits & Veggies': '0%',
        'Snacks': '12%',
        'Bakery': '5%',
        'Eggs, Meat & Fish': '0%',
        'Dairy': '5%',
        'Personal Care': '18%',
        'Home Care': '18%'
    }
    
    df['GST_Slab'] = df['Category'].map(gst_mapping).fillna('18%')
    df['GST_Rate'] = df['GST_Slab'].str.replace('%', '').astype(float) / 100
    df['Taxable_Value'] = df['Sales'] / (1 + df['GST_Rate'])
    df['GST_Amount'] = df['Sales'] - df['Taxable_Value']
    
    # HSN Codes
    hsn_mapping = {
        'Oil & Masala': '0904',
        'Beverages': '2202',
        'Food Grains': '1001',
        'Fruits & Veggies': '0701',
        'Snacks': '1905',
        'Bakery': '1905',
        'Eggs, Meat & Fish': '0207',
        'Dairy': '0401',
        'Personal Care': '3301',
        'Home Care': '3401'
    }
    df['HSN_Code'] = df['Category'].map(hsn_mapping).fillna('9999')
    
    # 6. Add Time Details
    print("Adding time analytics...")
    
    # Add random time to order date
    df['Order_Time'] = df['Order Date'] + pd.to_timedelta(
        np.random.randint(9, 21, size=len(df)), unit='h'
    ) + pd.to_timedelta(
        np.random.randint(0, 60, size=len(df)), unit='m'
    )
    
    df['Hour_of_Day'] = df['Order_Time'].dt.hour
    df['Day_of_Week'] = df['Order_Time'].dt.day_name()
    df['Is_Weekend'] = df['Day_of_Week'].isin(['Saturday', 'Sunday']).astype(int)
    df['Quarter'] = df['Order Date'].dt.quarter
    df['Month'] = df['Order Date'].dt.month
    
    # Financial Year (April to March)
    df['Financial_Year'] = df.apply(
        lambda row: f"FY{row['Order Date'].year}-{str(row['Order Date'].year+1)[2:]}" 
        if row['Order Date'].month >= 4 
        else f"FY{row['Order Date'].year-1}-{str(row['Order Date'].year)[2:]}", 
        axis=1
    )
    
    # Festival months in India
    festival_months = [1, 3, 4, 8, 9, 10, 11, 12]  # Jan, Mar, Apr, Aug-Dec
    df['Is_Festive'] = df['Month'].isin(festival_months).astype(int)
    
    # 7. Add Performance Metrics
    print("Adding performance metrics...")
    
    # Product rating (3.5-5.0)
    df['Product_Rating'] = np.round(np.random.uniform(3.5, 5.0, size=len(df)), 1)
    
    # Return rate (1-10%)
    df['Return_Rate'] = np.round(np.random.uniform(0.01, 0.10, size=len(df)), 3)
    
    # Product Profitability Index
    df['PPI'] = (df['Sales'] * df['Product_Rating']) / (df['Return_Rate'] + 0.001)
    
    # Margin percentage
    df['Margin_Percentage'] = ((df['Unit_Price'] - df['Cost_Price']) / df['Unit_Price']) * 100
    
    # 8. Add Promotion Details
    print("Adding promotion details...")
    
    df['Has_Promotion'] = (df['Discount'] > 0).astype(int)
    
    promotion_types = ['Festival Sale', 'Seasonal Offer', 'Clearance', 'Bundle Deal', 'Flash Sale']
    df['Promotion_Type'] = np.where(
        df['Has_Promotion'] == 1,
        np.random.choice(promotion_types, size=len(df)),
        'No Promotion'
    )
    
    # 9. Add Shipping Details
    print("Adding shipping details...")
    
    shipping_methods = ['Standard', 'Express', 'Same Day', 'Store Pickup']
    shipping_probs = [0.6, 0.2, 0.1, 0.1]
    df['Shipping_Method'] = np.random.choice(shipping_methods, size=len(df), p=shipping_probs)
    
    # Shipping cost based on order value
    df['Shipping_Cost'] = np.where(
        df['Sales'] > 2000,
        np.random.uniform(100, 200, size=len(df)),
        np.random.uniform(30, 100, size=len(df))
    )
    
    # Delivery status
    statuses = ['Delivered', 'In Transit', 'Processing', 'Cancelled']
    status_probs = [0.85, 0.08, 0.05, 0.02]
    df['Delivery_Status'] = np.random.choice(statuses, size=len(df), p=status_probs)
    
    # 10. Add Payment Details
    print("Adding payment details...")
    
    payment_methods = ['Credit Card', 'Debit Card', 'UPI', 'Cash', 'Net Banking']
    payment_probs = [0.25, 0.20, 0.35, 0.15, 0.05]
    df['Payment_Method'] = np.random.choice(payment_methods, size=len(df), p=payment_probs)
    
    # 11. Add Calculated Fields for Insights
    print("Adding calculated fields...")
    
    # Days of stock remaining
    df['Daily_Sales_Velocity'] = df.groupby('Product_ID')['Quantity'].transform('mean') / 30
    df['Days_of_Stock'] = df['Current_Stock'] / (df['Daily_Sales_Velocity'] + 0.001)
    
    # Stockout risk
    df['Stockout_Risk'] = (df['Days_of_Stock'] < df['Lead_Time_Days']).astype(int)
    
    # Overstock flag
    df['Overstock_Flag'] = (df['Current_Stock'] > df['Reorder_Point'] * 3).astype(int)
    
    # Reorder urgency
    df['Reorder_Urgency'] = pd.cut(
        df['Days_of_Stock'],
        bins=[0, 3, 7, 14, float('inf')],
        labels=['Critical', 'High', 'Medium', 'Low']
    )
    
    print("✅ Dataset enhancement complete!")
    print(f"Enhanced dataset shape: {df.shape}")
    print(f"Total columns added: {len(df.columns) - 11}")  # Original had 11 columns
    
    # Save enhanced dataset
    df.to_csv(output_file, index=False)
    print(f"💾 Enhanced dataset saved as: {output_file}")
    
    # Show sample of new columns
    print("\n📋 Sample of new columns:")
    sample_cols = ['Quantity', 'Unit_Price', 'Current_Stock', 'Customer_Segment', 
                  'GST_Slab', 'Product_Rating', 'PPI', 'Stockout_Risk']
    
    for col in sample_cols:
        if col in df.columns:
            print(f"{col}: {df[col].iloc[0]}")
    
    return df

# Run the enhancement
if __name__ == "__main__":
    enhanced_df = enhance_retail_dataset('sales data.csv', 'enhanced_sales_data.csv')
    
    # Print summary
    print("\n" + "="*60)
    print("ENHANCED DATASET SUMMARY")
    print("="*60)
    print(f"Total rows: {len(enhanced_df)}")
    print(f"Total columns: {len(enhanced_df.columns)}")
    print("\nNew column categories added:")
    print("1. Product Details: Product_ID, Product_Name, Quantity, Unit_Price, Cost_Price, Brand")
    print("2. Inventory: Current_Stock, Reorder_Point, Lead_Time_Days, Supplier_ID, Warehouse")
    print("3. Customer: Customer_ID, Customer_Segment, Loyalty_Points, Age_Group, Gender")
    print("4. GST: GST_Slab, GST_Rate, GST_Amount, HSN_Code")
    print("5. Time: Order_Time, Hour_of_Day, Day_of_Week, Is_Weekend, Financial_Year")
    print("6. Performance: Product_Rating, Return_Rate, PPI, Margin_Percentage")
    print("7. Promotion: Has_Promotion, Promotion_Type")
    print("8. Logistics: Shipping_Method, Shipping_Cost, Delivery_Status")
    print("9. Analytics: Days_of_Stock, Stockout_Risk, Overstock_Flag, Reorder_Urgency")