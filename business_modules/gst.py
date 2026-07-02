"""
GST CALCULATION MODULE
---------------------
This module calculates:
1. Output GST (on sales)
2. Input Tax Credit (ITC)
3. Net GST Payable
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def calculate_output_gst(predicted_sales, gst_rate):
    """
    Calculates GST collected on sales (Output Tax)
    """
    output_gst = predicted_sales * (gst_rate / 100)
    return round(output_gst, 2)


def calculate_input_tax_credit(purchase_amount, purchase_gst_rate):
    """
    Calculates Input Tax Credit (GST paid on purchases)
    """
    itc = purchase_amount * (purchase_gst_rate / 100)
    return round(itc, 2)


def calculate_net_gst(output_gst, input_tax_credit):
    """
    Calculates Net GST Payable
    Formula: Net GST = Output GST - Input Tax Credit
    """
    net_gst = output_gst - input_tax_credit
    return round(max(net_gst, 0), 2)


def gst_summary(predicted_sales,
                sales_gst_rate,
                purchase_amount=0,
                purchase_gst_rate=0):
    """
    Generates a complete GST summary
    """
    output_gst = calculate_output_gst(predicted_sales, sales_gst_rate)
    input_gst = calculate_input_tax_credit(purchase_amount, purchase_gst_rate)
    net_gst = calculate_net_gst(output_gst, input_gst)
    
    excess_itc = max(input_gst - output_gst, 0)

    return {
        "Predicted Sales": round(predicted_sales, 2),
        "Sales GST Rate (%)": sales_gst_rate,
        "Output GST": output_gst,
        "Purchase Amount": round(purchase_amount, 2),
        "Purchase GST Rate (%)": purchase_gst_rate,
        "Input Tax Credit": input_gst,
        "Net GST Payable": net_gst,
        "Excess ITC": round(excess_itc, 2) if excess_itc > 0 else 0
    }


def product_gst_summary(product_predictions, sales_gst_rate, 
                       purchase_amounts=None, purchase_gst_rate=0):
    """
    Calculates GST for each product
    
    Parameters:
    product_predictions: dict - predictions for each product
    sales_gst_rate: float - GST rate on sales
    purchase_amounts: dict - purchase amounts per product (optional)
    purchase_gst_rate: float - GST rate on purchases
    
    Returns:
    dict - GST summary per product
    """
    
    gst_report = {}
    
    for product, pred_data in product_predictions.items():
        predicted_sales = pred_data['latest_prediction']
        
        # Output GST
        output_gst = predicted_sales * (sales_gst_rate / 100)
        
        # Input Tax Credit
        purchase_amount = purchase_amounts.get(product, 0) if purchase_amounts else 0
        input_gst = purchase_amount * (purchase_gst_rate / 100)
        
        # Net GST
        net_gst = max(output_gst - input_gst, 0)
        excess_itc = max(input_gst - output_gst, 0)
        
        gst_report[product] = {
            'predicted_sales': round(predicted_sales, 2),
            'sales_gst_rate': sales_gst_rate,
            'output_gst': round(output_gst, 2),
            'purchase_amount': round(purchase_amount, 2),
            'purchase_gst_rate': purchase_gst_rate,
            'input_tax_credit': round(input_gst, 2),
            'net_gst_payable': round(net_gst, 2),
            'excess_itc': round(excess_itc, 2)
        }
    
    return gst_report


def get_gst_aggregate_summary(product_gst_report):
    """
    Aggregates GST data across all products
    """
    
    if not product_gst_report:
        return {
            'total_predicted_sales': 0,
            'total_output_gst': 0,
            'total_input_tax_credit': 0,
            'total_net_gst_payable': 0,
            'total_excess_itc': 0,
            'number_of_products': 0,
            'effective_gst_rate': 0
        }
    
    total_sales = sum(data['predicted_sales'] for data in product_gst_report.values())
    total_output = sum(data['output_gst'] for data in product_gst_report.values())
    total_input = sum(data['input_tax_credit'] for data in product_gst_report.values())
    total_net = sum(data['net_gst_payable'] for data in product_gst_report.values())
    total_excess = sum(data.get('excess_itc', 0) for data in product_gst_report.values())
    
    return {
        'total_predicted_sales': round(total_sales, 2),
        'total_output_gst': round(total_output, 2),
        'total_input_tax_credit': round(total_input, 2),
        'total_net_gst_payable': round(total_net, 2),
        'total_excess_itc': round(total_excess, 2),
        'number_of_products': len(product_gst_report),
        'effective_gst_rate': round((total_output / total_sales * 100), 2) if total_sales > 0 else 0
    }


def get_top_gst_contributors(product_gst_report, top_n=10):
    """
    Returns products contributing the most to GST liability
    """
    
    contributors = []
    
    for product, data in product_gst_report.items():
        contributors.append({
            'product': product,
            'net_gst_payable': data['net_gst_payable'],
            'output_gst': data['output_gst'],
            'predicted_sales': data['predicted_sales']
        })
    
    contributors.sort(key=lambda x: x['net_gst_payable'], reverse=True)
    
    return contributors[:top_n]


def project_gst_timeline(monthly_forecast, sales_gst_rate, purchase_ratio=0.6, purchase_gst_rate=18):
    """
    Projects GST liability across multiple months/periods
    
    Parameters:
    monthly_forecast: DataFrame - forecast with 'ds' and sales column
    sales_gst_rate: float - GST rate on sales
    purchase_ratio: float - purchases as ratio of sales
    purchase_gst_rate: float - GST rate on purchases
    
    Returns:
    DataFrame - timeline of GST projections
    """
    
    if isinstance(monthly_forecast, dict):
        monthly_forecast = pd.DataFrame(monthly_forecast)
    
    # Get sales column
    sales_col = 'Prophet_Forecast' if 'Prophet_Forecast' in monthly_forecast.columns else 'Predicted_Sales'
    
    gst_timeline = monthly_forecast[['ds', sales_col]].copy()
    gst_timeline = gst_timeline.rename(columns={sales_col: 'Predicted_Sales'})
    
    # Calculate GST
    gst_timeline['Output_GST'] = gst_timeline['Predicted_Sales'] * (sales_gst_rate / 100)
    gst_timeline['Estimated_Purchases'] = gst_timeline['Predicted_Sales'] * purchase_ratio
    gst_timeline['Input_Tax_Credit'] = gst_timeline['Estimated_Purchases'] * (purchase_gst_rate / 100)
    gst_timeline['Net_GST_Payable'] = gst_timeline['Output_GST'] - gst_timeline['Input_Tax_Credit']
    gst_timeline['Net_GST_Payable'] = gst_timeline['Net_GST_Payable'].apply(lambda x: max(x, 0))
    gst_timeline['Cumulative_GST'] = gst_timeline['Net_GST_Payable'].cumsum()
    
    # Round all values
    for col in ['Predicted_Sales', 'Output_GST', 'Estimated_Purchases', 'Input_Tax_Credit', 'Net_GST_Payable', 'Cumulative_GST']:
        gst_timeline[col] = gst_timeline[col].round(2)
    
    return gst_timeline


def get_gst_filing_alerts(gst_timeline, current_date=None):
    """
    Generates GST filing reminders based on Indian GST filing schedule
    """
    
    if current_date is None:
        current_date = datetime.now()
    
    alerts = []
    
    for _, row in gst_timeline.iterrows():
        period_date = pd.to_datetime(row['ds'])
        
        # Filing due date is 20th of next month
        filing_due = (period_date + pd.DateOffset(months=1)).replace(day=20)
        
        days_until_due = (filing_due - current_date).days
        
        if 0 <= days_until_due <= 30:
            alerts.append({
                'period': period_date.strftime('%B %Y'),
                'filing_due_date': filing_due.strftime('%d-%m-%Y'),
                'days_remaining': days_until_due,
                'gst_amount': row['Net_GST_Payable'],
                'urgency': 'HIGH' if days_until_due <= 7 else 'MEDIUM' if days_until_due <= 15 else 'LOW'
            })
    
    return sorted(alerts, key=lambda x: x['days_remaining'])