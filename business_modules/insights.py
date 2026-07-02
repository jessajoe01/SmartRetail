"""
BUSINESS INSIGHTS MODULE
------------------------
Generates human-readable insights explaining:
• Sales increase / decrease
• Seasonality impact
• Discount effectiveness
• Inventory risk
"""

import pandas as pd
import numpy as np


def sales_change_insights(df):
    """
    Explains why sales changed month-over-month
    """

    insights = []

    df = df.sort_values('ds').copy()
    df['sales_change_pct'] = df['Sales'].pct_change() * 100

    for i in range(1, len(df)):
        change = df.iloc[i]['sales_change_pct']
        month = df.iloc[i]['ds'].strftime('%B %Y')

        if change > 15:
            insights.append(
                f"📈 Sales increased significantly in {month} (+{change:.1f}%). Possible reasons: seasonality, promotions, or high demand."
            )
        elif change < -15:
            insights.append(
                f"📉 Sales dropped sharply in {month} ({change:.1f}%). Possible reasons: off-season period, high prices, or reduced demand."
            )

    return insights


def seasonal_insights(df):
    """
    Identifies best & worst performing months
    """

    insights = []

    monthly_avg = df.groupby(df['ds'].dt.month)['Sales'].mean()

    peak_month = monthly_avg.idxmax()
    low_month = monthly_avg.idxmin()

    insights.append(
        f"🔥 Peak sales usually occur in month {peak_month}. Inventory should be increased during this period."
    )

    insights.append(
        f"❄️ Lowest sales are observed in month {low_month}. Consider discounts or promotions."
    )

    return insights


def discount_effectiveness(df):
    """
    Analyzes whether discounts improve sales
    """

    insights = []

    if 'Discount' not in df.columns:
        return ["⚠️ Discount data not available to analyze effectiveness."]

    corr = df['Sales'].corr(df['Discount'])

    if corr > 0.4:
        insights.append("💡 Discounts strongly increase sales. Promotional strategies are effective.")
    elif corr < -0.2:
        insights.append("⚠️ Higher discounts do not increase sales. Pricing strategy should be reviewed.")
    else:
        insights.append("ℹ️ Discounts have limited impact on sales.")

    return insights


def inventory_risk_insights(predicted_sales, current_stock):
    """
    Generates inventory alerts
    """

    if current_stock < predicted_sales:
        return [f"🚨 Stock-out risk! Predicted sales ({predicted_sales:.0f}) exceed current stock ({current_stock})."]

    elif current_stock > predicted_sales * 2:
        return [f"📦 Overstock alert. Current stock is much higher than predicted demand."]

    else:
        return ["✅ Inventory levels are balanced."]


def generate_all_insights(df, predicted_sales, current_stock):
    """
    Combines all insights into one list
    """

    insights = []
    insights += sales_change_insights(df)
    insights += seasonal_insights(df)
    insights += discount_effectiveness(df)
    insights += inventory_risk_insights(predicted_sales, current_stock)

    return insights