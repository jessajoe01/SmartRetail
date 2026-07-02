"""
INVENTORY FORECASTING & ALERT MODULE
-----------------------------------
Generates:
1. Reorder alerts
2. Stock-out warnings
3. Over-stock alerts
"""

import pandas as pd


def calculate_daily_demand(predicted_sales, period_days):
    """Converts predicted sales into average daily demand"""
    if period_days <= 0:
        raise ValueError("period_days must be > 0")
    return predicted_sales / period_days


def estimate_stock_days(current_stock, daily_demand):
    """Estimates how many days current stock will last"""
    if daily_demand <= 0:
        return float("inf")
    return current_stock / daily_demand


def inventory_status(current_stock,
                     predicted_sales,
                     period_days,
                     reorder_level,
                     overstock_level):
    """
    Determines inventory alerts for a single item
    """
    daily_demand = calculate_daily_demand(predicted_sales, period_days)
    stock_days = estimate_stock_days(current_stock, daily_demand)

    if current_stock <= reorder_level:
        status = "LOW STOCK"
        action = "Reorder immediately"
    elif current_stock >= overstock_level:
        status = "OVER STOCK"
        action = "Reduce purchase / offer discount"
    else:
        status = "OPTIMAL STOCK"
        action = "No action required"

    return {
        "Current Stock": round(current_stock, 2),
        "Predicted Sales": round(predicted_sales, 2),
        "Daily Demand": round(daily_demand, 2),
        "Stock Coverage (Days)": round(stock_days, 1),
        "Inventory Status": status,
        "Suggested Action": action
    }


def product_inventory_status(product_predictions, stock_data, period_days=30):
    """
    Generates inventory status for each product
    """
    inventory_report = {}
    
    for product, pred_data in product_predictions.items():
        current_stock = stock_data.get(product, 0)
        predicted_sales = pred_data['latest_prediction']
        
        daily_demand = predicted_sales / period_days if period_days > 0 else 0
        stock_days = current_stock / daily_demand if daily_demand > 0 else float('inf')
        
        # Determine status
        if stock_days < 3:
            status = "CRITICAL"
            action = "🚨 URGENT: Reorder immediately - less than 3 days stock!"
            priority = "URGENT"
        elif stock_days < 7:
            status = "CRITICAL"
            action = "⚠️ Reorder immediately - less than 1 week stock"
            priority = "HIGH"
        elif stock_days < 14:
            status = "LOW"
            action = "📦 Reorder soon - less than 2 weeks stock"
            priority = "MEDIUM"
        elif stock_days > 90:
            status = "OVERSTOCK"
            action = "💰 Severe overstock - aggressive discount recommended"
            priority = "HIGH"
        elif stock_days > 60:
            status = "OVERSTOCK"
            action = "📉 Reduce orders / offer discount"
            priority = "MEDIUM"
        else:
            status = "OPTIMAL"
            action = "✅ No action required"
            priority = "LOW"
        
        inventory_report[product] = {
            'current_stock': round(current_stock, 2),
            'predicted_sales': round(predicted_sales, 2),
            'daily_demand': round(daily_demand, 2),
            'stock_coverage_days': round(stock_days, 1),
            'status': status,
            'action': action,
            'priority': priority
        }
    
    return inventory_report


def calculate_reorder_quantities(product_predictions, stock_data, 
                                 safety_stock_days=7, lead_time_days=3):
    """
    Calculates optimal reorder quantities for each product
    """
    reorder_plan = {}
    
    for product, pred_data in product_predictions.items():
        current_stock = stock_data.get(product, 0)
        predicted_monthly_sales = pred_data['latest_prediction']
        daily_demand = predicted_monthly_sales / 30
        
        safety_stock = daily_demand * safety_stock_days
        lead_time_stock = daily_demand * lead_time_days
        required_stock = safety_stock + lead_time_stock
        
        if current_stock < required_stock:
            reorder_qty = required_stock - current_stock
            reorder_qty_with_buffer = reorder_qty * 1.1
            
            reorder_plan[product] = {
                'current_stock': round(current_stock, 2),
                'daily_demand': round(daily_demand, 2),
                'safety_stock': round(safety_stock, 2),
                'lead_time_stock': round(lead_time_stock, 2),
                'required_stock': round(required_stock, 2),
                'reorder_quantity': round(reorder_qty, 2),
                'reorder_quantity_with_buffer': round(reorder_qty_with_buffer, 2),
                'total_cost_estimate': round(reorder_qty * 50, 2),
                'priority': 'URGENT' if current_stock < safety_stock / 2 else 'HIGH' if current_stock < safety_stock else 'MEDIUM',
                'stock_shortage': round(required_stock - current_stock, 2)
            }
    
    return reorder_plan


def get_inventory_summary(inventory_report):
    """
    Generates a summary of inventory status across all products
    """
    if not inventory_report:
        return {
            'total_products': 0,
            'critical_items': 0,
            'low_items': 0,
            'optimal_items': 0,
            'overstock_items': 0,
            'urgent_actions_needed': 0,
            'health_score': 0
        }
    
    status_counts = {
        'CRITICAL': 0,
        'LOW': 0,
        'OPTIMAL': 0,
        'OVERSTOCK': 0
    }
    
    urgent_count = 0
    
    for product, data in inventory_report.items():
        status = data['status']
        priority = data.get('priority', 'LOW')
        
        status_counts[status] = status_counts.get(status, 0) + 1
        
        if priority == 'URGENT':
            urgent_count += 1
    
    return {
        'total_products': len(inventory_report),
        'critical_items': status_counts['CRITICAL'],
        'low_items': status_counts['LOW'],
        'optimal_items': status_counts['OPTIMAL'],
        'overstock_items': status_counts['OVERSTOCK'],
        'urgent_actions_needed': urgent_count,
        'health_score': round((status_counts['OPTIMAL'] / len(inventory_report)) * 100, 1) if inventory_report else 0
    }


def get_critical_products(inventory_report, top_n=10):
    """
    Returns the most critical products that need immediate attention
    """
    critical_products = []
    
    for product, data in inventory_report.items():
        if data['status'] in ['CRITICAL', 'LOW']:
            critical_products.append({
                'product': product,
                'stock_days': data['stock_coverage_days'],
                'status': data['status'],
                'priority': data.get('priority', 'MEDIUM'),
                'action': data['action']
            })
    
    critical_products.sort(key=lambda x: x['stock_days'])
    
    return critical_products[:top_n]


def get_overstock_products(inventory_report, top_n=10):
    """
    Returns products with excess inventory
    """
    overstock_products = []
    
    for product, data in inventory_report.items():
        if data['status'] == 'OVERSTOCK':
            overstock_products.append({
                'product': product,
                'stock_days': data['stock_coverage_days'],
                'current_stock': data['current_stock'],
                'daily_demand': data['daily_demand'],
                'action': data['action']
            })
    
    overstock_products.sort(key=lambda x: x['stock_days'], reverse=True)
    
    return overstock_products[:top_n]