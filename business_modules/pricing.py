"""
DYNAMIC PRICING & DISCOUNT RECOMMENDATION MODULE
-----------------------------------------------
Suggests:
1. Optimal price change
2. Discount recommendations
3. Revenue improvement strategy
"""

import pandas as pd
import numpy as np


def calculate_price_elasticity(price, sales):
    """
    Calculates price elasticity of demand.
    Elasticity = % change in demand / % change in price
    """
    if len(price) < 2 or len(sales) < 2:
        return None

    pct_change_sales = np.diff(sales) / sales[:-1]
    pct_change_price = np.diff(price) / price[:-1]
    
    pct_change_price = np.where(pct_change_price == 0, 0.0001, pct_change_price)
    elasticity = np.mean(pct_change_sales / pct_change_price)

    return elasticity


def pricing_strategy(predicted_sales,
                     current_price,
                     current_discount,
                     elasticity,
                     cost_price=None):
    """
    Generates pricing & discount recommendations
    """
    recommendation = {}

    if elasticity is not None and elasticity < -1:
        recommendation["Demand Sensitivity"] = "High"
        recommendation["Suggested Price Action"] = "Reduce price slightly"
        recommendation["Suggested Discount (%)"] = min(current_discount + 5, 30)

    elif elasticity is not None and -1 <= elasticity < 0:
        recommendation["Demand Sensitivity"] = "Low"
        recommendation["Suggested Price Action"] = "Increase price slightly"
        recommendation["Suggested Discount (%)"] = max(current_discount - 5, 0)

    else:
        recommendation["Demand Sensitivity"] = "Unknown"
        recommendation["Suggested Price Action"] = "Maintain price"
        recommendation["Suggested Discount (%)"] = current_discount

    adjusted_price = current_price * (1 - recommendation["Suggested Discount (%)"] / 100)
    estimated_revenue = predicted_sales * adjusted_price

    recommendation.update({
        "Current Price": current_price,
        "Current Discount (%)": current_discount,
        "Adjusted Price": round(adjusted_price, 2),
        "Estimated Revenue": round(estimated_revenue, 2),
        "Predicted Sales": round(predicted_sales, 2)
    })

    if cost_price:
        recommendation["Estimated Profit"] = round(
            (adjusted_price - cost_price) * predicted_sales, 2
        )

    return recommendation


def product_pricing_strategy(product_predictions, current_prices, 
                             cost_prices, discount_data=None):
    """
    Generates pricing recommendations for each product
    
    Parameters:
    product_predictions: dict - predictions for each product
    current_prices: dict - current selling prices {product: price}
    cost_prices: dict - cost prices {product: cost}
    discount_data: dict - optional discount data
    
    Returns:
    dict - pricing strategy for each product
    """
    
    pricing_report = {}
    
    for product, pred_data in product_predictions.items():
        predicted_sales = pred_data['latest_prediction']
        current_price = current_prices.get(product, 100)
        cost_price = cost_prices.get(product, 70)
        
        # Calculate profit margin
        profit_margin = ((current_price - cost_price) / current_price) * 100
        
        # Get average sales
        avg_sales = pred_data.get('avg_sales', predicted_sales)
        
        # Determine pricing strategy
        sales_ratio = predicted_sales / avg_sales if avg_sales > 0 else 1
        
        if sales_ratio < 0.8:
            suggested_action = "REDUCE PRICE"
            suggested_discount = 15
            reason = "📉 Sales below average - increase competitiveness"
            priority = "HIGH"
        elif sales_ratio < 0.9:
            suggested_action = "REDUCE PRICE"
            suggested_discount = 10
            reason = "⚠️ Sales declining - consider promotion"
            priority = "MEDIUM"
        elif sales_ratio > 1.3:
            suggested_action = "INCREASE PRICE"
            suggested_discount = 5
            reason = "🚀 High demand - maximize profit margin"
            priority = "HIGH"
        elif sales_ratio > 1.1:
            suggested_action = "INCREASE PRICE"
            suggested_discount = 7
            reason = "📈 Growing demand - reduce discount"
            priority = "MEDIUM"
        else:
            suggested_action = "MAINTAIN PRICE"
            suggested_discount = 10
            reason = "✅ Stable demand - maintain strategy"
            priority = "LOW"
        
        adjusted_price = current_price * (1 - suggested_discount / 100)
        estimated_revenue = predicted_sales * adjusted_price
        estimated_profit = (adjusted_price - cost_price) * predicted_sales
        
        current_revenue = predicted_sales * current_price
        current_profit = (current_price - cost_price) * predicted_sales
        
        revenue_change = ((estimated_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
        profit_change = ((estimated_profit - current_profit) / current_profit * 100) if current_profit > 0 else 0
        
        pricing_report[product] = {
            'current_price': round(current_price, 2),
            'cost_price': round(cost_price, 2),
            'profit_margin': round(profit_margin, 2),
            'suggested_action': suggested_action,
            'suggested_discount': suggested_discount,
            'adjusted_price': round(adjusted_price, 2),
            'predicted_sales': round(predicted_sales, 2),
            'estimated_revenue': round(estimated_revenue, 2),
            'estimated_profit': round(estimated_profit, 2),
            'revenue_change_%': round(revenue_change, 1),
            'profit_change_%': round(profit_change, 1),
            'reason': reason,
            'priority': priority
        }
    
    return pricing_report


def get_pricing_summary(pricing_report):
    """Generates summary statistics across all products"""
    
    if not pricing_report:
        return {
            'total_products': 0,
            'increase_price': 0,
            'reduce_price': 0,
            'maintain_price': 0,
            'total_potential_revenue': 0,
            'total_potential_profit': 0
        }
    
    action_counts = {
        'INCREASE PRICE': 0,
        'REDUCE PRICE': 0,
        'MAINTAIN PRICE': 0
    }
    
    total_revenue = 0
    total_profit = 0
    
    for product, data in pricing_report.items():
        action = data['suggested_action']
        action_counts[action] = action_counts.get(action, 0) + 1
        total_revenue += data['estimated_revenue']
        total_profit += data['estimated_profit']
    
    return {
        'total_products': len(pricing_report),
        'increase_price': action_counts['INCREASE PRICE'],
        'reduce_price': action_counts['REDUCE PRICE'],
        'maintain_price': action_counts['MAINTAIN PRICE'],
        'total_potential_revenue': round(total_revenue, 2),
        'total_potential_profit': round(total_profit, 2),
        'avg_profit_margin': round(np.mean([d['profit_margin'] for d in pricing_report.values()]), 2)
    }


def get_top_pricing_opportunities(pricing_report, top_n=10, sort_by='profit_change_%'):
    """Returns products with highest pricing optimization potential"""
    
    opportunities = []
    
    for product, data in pricing_report.items():
        opportunities.append({
            'product': product,
            'suggested_action': data['suggested_action'],
            'profit_change_%': data['profit_change_%'],
            'revenue_change_%': data['revenue_change_%'],
            'potential_profit': data['estimated_profit'],
            'reason': data['reason'],
            'priority': data['priority']
        })
    
    opportunities.sort(key=lambda x: abs(x[sort_by]), reverse=True)
    
    return opportunities[:top_n]


def calculate_price_optimization_impact(pricing_report):
    """Calculates total financial impact of implementing recommendations"""
    
    total_revenue_impact = 0
    total_profit_impact = 0
    
    products_improved = 0
    products_declined = 0
    
    for product, data in pricing_report.items():
        revenue_change = data.get('revenue_change_%', 0)
        profit_change = data.get('profit_change_%', 0)
        
        if profit_change > 0:
            products_improved += 1
        elif profit_change < 0:
            products_declined += 1
        
        total_revenue_impact += data['estimated_revenue'] * (revenue_change / 100)
        total_profit_impact += data['estimated_profit'] * (profit_change / 100)
    
    return {
        'total_revenue_impact': round(total_revenue_impact, 2),
        'total_profit_impact': round(total_profit_impact, 2),
        'products_improved': products_improved,
        'products_declined': products_declined,
        'net_improvement_rate_%': round((products_improved / len(pricing_report) * 100), 1) if pricing_report else 0
    }


def calculate_optimal_price(predicted_sales, cost_price, price_range=(50, 500), steps=50):
    """Calculates optimal price that maximizes profit"""
    
    min_price, max_price = price_range
    prices = np.linspace(min_price, max_price, steps)
    
    base_price = (min_price + max_price) / 2
    
    max_profit = 0
    optimal_price = base_price
    optimal_revenue = 0
    optimal_quantity = 0
    
    for price in prices:
        price_ratio = price / base_price
        quantity_multiplier = max(0.3, 1 - (price_ratio - 1) * 0.5)
        
        quantity = predicted_sales * quantity_multiplier
        revenue = price * quantity
        profit = (price - cost_price) * quantity
        
        if profit > max_profit:
            max_profit = profit
            optimal_price = price
            optimal_revenue = revenue
            optimal_quantity = quantity
    
    return {
        'optimal_price': round(optimal_price, 2),
        'max_profit': round(max_profit, 2),
        'optimal_revenue': round(optimal_revenue, 2),
        'optimal_quantity': round(optimal_quantity, 2),
        'profit_margin_%': round(((optimal_price - cost_price) / optimal_price * 100), 2)
    }