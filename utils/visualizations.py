"""
Visualization utilities for the dashboard
"""
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

class DashboardVisualizations:
    """Creates interactive visualizations for the dashboard"""
    
    @staticmethod
    def create_sales_forecast_chart(historical_data, forecast_data, title="Sales Forecast"):
        """Create sales forecast chart with historical and predicted values"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data['ds'],
            y=historical_data['y'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='#3B82F6', width=2),
            marker=dict(size=6)
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#10B981', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        # Confidence interval (if available)
        if 'yhat_lower' in forecast_data.columns and 'yhat_upper' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'].tolist() + forecast_data['ds'].tolist()[::-1],
                y=forecast_data['yhat_upper'].tolist() + forecast_data['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(16, 185, 129, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Sales",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_product_performance_chart(product_data, metric='Sales', top_n=10):
        """Create bar chart for top performing products"""
        # Aggregate product data
        if 'Product' in product_data.columns and metric in product_data.columns:
            product_stats = product_data.groupby('Product')[metric].sum().sort_values(ascending=False).head(top_n)
            
            fig = px.bar(
                x=product_stats.values,
                y=product_stats.index,
                orientation='h',
                title=f"Top {top_n} Products by {metric}",
                labels={'x': metric, 'y': 'Product'},
                color=product_stats.values,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                height=400,
                template='plotly_white'
            )
            
            return fig
        
        return None
    
    @staticmethod
    def create_inventory_alert_chart(inventory_alerts):
        """Create visualization for inventory alerts"""
        alert_counts = {
            'Stockout': len(inventory_alerts.get('stockout', [])),
            'Overstock': len(inventory_alerts.get('overstock', [])),
            'Reorder': len(inventory_alerts.get('reorder', []))
        }
        
        fig = px.bar(
            x=list(alert_counts.keys()),
            y=list(alert_counts.values()),
            title="Inventory Alerts Summary",
            labels={'x': 'Alert Type', 'y': 'Count'},
            color=list(alert_counts.keys()),
            color_discrete_map={
                'Stockout': '#EF4444',
                'Overstock': '#F59E0B',
                'Reorder': '#3B82F6'
            }
        )
        
        fig.update_layout(
            height=300,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_gst_trend_chart(gst_data):
        """Create GST liability trend chart"""
        if 'Month' in gst_data.columns and 'Net_GST_Payable' in gst_data.columns:
            fig = px.line(
                gst_data,
                x='Month',
                y='Net_GST_Payable',
                title="Monthly GST Liability Trend",
                markers=True
            )
            
            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                height=400,
                template='plotly_white',
                xaxis_title="Month",
                yaxis_title="Net GST Payable"
            )
            
            return fig
        
        return None
    
    @staticmethod
    def create_seasonal_pattern_chart(monthly_sales):
        """Create seasonal pattern visualization"""
        if 'Month' in monthly_sales.columns and 'Sales' in monthly_sales.columns:
            # Convert month numbers to names
            monthly_sales['Month_Name'] = monthly_sales['Month'].apply(
                lambda x: pd.Timestamp(f'2023-{int(x)}-01').strftime('%b')
            )
            
            fig = px.line_polar(
                monthly_sales,
                r='Sales',
                theta='Month_Name',
                line_close=True,
                title="Seasonal Sales Pattern",
                template='plotly_white'
            )
            
            fig.update_traces(fill='toself')
            
            return fig
        
        return None
    
    @staticmethod
    def create_model_comparison_chart(model_results):
        """Create comparison chart for model performance"""
        models = list(model_results.keys())
        metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set2
        
        for idx, metric in enumerate(metrics):
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            values = [model_results[model].get(metric, 0) for model in models]
            
            fig.add_trace(
                go.Bar(
                    x=models,
                    y=values,
                    name=metric,
                    marker_color=colors[idx],
                    text=[f'{v:.2f}' for v in values],
                    textposition='auto'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="Model Performance Comparison",
            showlegend=False,
            height=600,
            template='plotly_white'
        )
        
        return fig