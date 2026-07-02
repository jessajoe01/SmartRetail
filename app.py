import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML Inference
from ml_pipeline.inference import run_multi_timeline_inference, get_timeline_summary, compare_timelines
from ml_pipeline.preprocessing import get_top_products, get_product_statistics

# Business Modules
from business_modules.inventory import product_inventory_status, calculate_reorder_quantities
from business_modules.pricing import product_pricing_strategy
from business_modules.gst import product_gst_summary
from business_modules.ai_assistant import RetailAIAssistant
from business_modules.insights import (
    sales_change_insights,
    seasonal_insights,
    discount_effectiveness,
    inventory_risk_insights,
    generate_all_insights
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Smart Retail Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ultra-modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeIn 1s ease-in;
    }
    
    .sub-header {
        font-size: 1.3rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
    }
    
    .product-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
    }
    
    .product-card:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        transform: translateX(5px);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        color: #333;
    }
    
    .alert-success {
        background: linear-gradient(135deg, #48c6ef 0%, #6f86d6 100%);
        color: white;
    }
    
    .chat-message {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .chat-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: right;
    }
    
    .insight-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 0.6rem 0;
        font-size: 1rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .insight-up {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        color: #155724;
    }
    
    .insight-down {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        color: #721c24;
    }
    
    .insight-info {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left: 5px solid #17a2b8;
        color: #0c5460;
    }
    
    .insight-warn {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        border-left: 5px solid #ffc107;
        color: #856404;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown('<div class="main-header">🛍️ Smart Retail Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Sales Forecasting | Inventory Optimization | Dynamic Pricing | GST Analytics</div>', unsafe_allow_html=True)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
st.sidebar.title("⚙️ Control Panel")

st.sidebar.markdown("### 🔮 Forecast Settings")

# ---- USER-FRIENDLY MONTH SELECTOR ----
MONTH_OPTIONS = {
    "3 Months": 12,
    "9 Months": 18,
    "1 Year 3 Months": 24,
}

selected_label = st.sidebar.selectbox(
    "📅 Forecast Duration",
    options=list(MONTH_OPTIONS.keys()),
    index=0,   # Default: 12 Months
    help="How many months ahead should the model forecast?"
)

future_periods = MONTH_OPTIONS[selected_label]

st.sidebar.success(f"Forecasting the next **{selected_label}** of sales")

# Timeline is always monthly (simplification as per original code)
timeline = "monthly"

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
st.markdown("---")
uploaded_file = st.file_uploader(
    "📤 Upload Your Sales Dataset (CSV)",
    type=["csv"],
    help="Upload CSV with columns: Date, Sales, Product (optional), Category (optional)"
)

if uploaded_file is None:
    st.info("👆 Upload a dataset to unlock product-level intelligence")
    
    with st.expander("📋 What makes this different?"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🎯 Product-Level Insights**
            - Individual product forecasts
            - SKU-wise inventory tracking
            - Product-specific pricing
            """)
        
        with col2:
            st.markdown("""
            **🤖 AI Assistant**
            - Ask questions in plain English
            - Get instant insights
            - Smart recommendations
            """)
        
        with col3:
            st.markdown("""
            **📊 Interactive Dashboards**
            - Drill down by product
            - Filter by category
            - Real-time updates
            """)
    
    st.stop()

# Load data
try:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset loaded! ({len(df):,} rows)")
    
    with st.expander("🔍 Quick Data Preview"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Columns:**", list(df.columns))
        
        with col2:
            st.dataframe(df.head(3))
            
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()

# -------------------------------------------------
# RUN ANALYSIS
# -------------------------------------------------
if st.button("🚀 Analyze My Business", type="primary", use_container_width=True):
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    try:
        status.text(f"🔄 Step 1/5: Preprocessing data for {selected_label} forecast...")
        progress_bar.progress(20)
        
        # Run monthly inference with the user-chosen number of months
        results = run_multi_timeline_inference(
            uploaded_df=df,
            timeline=timeline,
            future_periods=future_periods
        )
        
        status.text("📊 Step 2/5: Analysing products...")
        progress_bar.progress(40)
        
        product_preds = results.get('product_predictions', {})
        
        # Mock stock data (in real app, user would input this)
        stock_data = {product: np.random.randint(100, 1000) for product in product_preds.keys()}
        
        status.text("📦 Step 3/5: Inventory analysis...")
        progress_bar.progress(60)
        
        if product_preds:
            # Monthly predictions → divide by 30 for daily rate used in inventory module
            adjusted_preds = {
                product: {
                    'latest_prediction': pred_data['latest_prediction'] / 30,
                    'historical_predictions': pred_data['historical_predictions']
                }
                for product, pred_data in product_preds.items()
            }
            
            inventory_report = product_inventory_status(adjusted_preds, stock_data, period_days=30)
            reorder_plan = calculate_reorder_quantities(adjusted_preds, stock_data)
        else:
            inventory_report = {}
            reorder_plan = {}
        
        status.text("💰 Step 4/5: Pricing optimisation...")
        progress_bar.progress(80)
        
        if product_preds:
            current_prices = {product: np.random.uniform(50, 500) for product in product_preds.keys()}
            cost_prices = {product: price * 0.7 for product, price in current_prices.items()}
            pricing_report = product_pricing_strategy(product_preds, current_prices, cost_prices)
        else:
            pricing_report = {}
            current_prices = {}

        gst_report = product_gst_summary(product_preds, sales_gst_rate=18) if product_preds else {}
        
        status.text("✅ Step 5/5: Generating insights...")
        progress_bar.progress(100)
        
        timeline_summary = get_timeline_summary(results, timeline)

        # ---- Generate business insights from the insights module ----
        # Build a clean aggregated df for the insights module
        from ml_pipeline.preprocessing import parse_dates_robustly
        insights_df = df.copy()
        
        # Map date column
        date_col_candidates = ['Order Date', 'order_date', 'date', 'Date', 'sale_date', 'Sales Date']
        date_col = next((c for c in date_col_candidates if c in insights_df.columns), None)
        
        business_insights = []
        if date_col and 'Sales' in insights_df.columns or any(
            c.lower() in ['sales', 'revenue', 'amount'] for c in insights_df.columns
        ):
            try:
                # Normalise column names for insights module
                col_map = {}
                for c in insights_df.columns:
                    if c.lower() in ['order date', 'order_date', 'date', 'sale_date']:
                        col_map[c] = 'ds'
                    elif c.lower() in ['sales', 'revenue', 'amount', 'total']:
                        col_map[c] = 'Sales'
                    elif c.lower() in ['discount', 'discount_percent', 'disc']:
                        col_map[c] = 'Discount'

                ins_df = insights_df.rename(columns=col_map)
                ins_df['ds'] = parse_dates_robustly(ins_df['ds'])
                ins_df = ins_df.dropna(subset=['ds', 'Sales'])
                
                monthly_ins = ins_df.groupby(ins_df['ds'].dt.to_period('M').dt.to_timestamp()).agg(
                    {'Sales': 'sum', **({'Discount': 'mean'} if 'Discount' in ins_df.columns else {})}
                ).reset_index().rename(columns={'index': 'ds'})
                monthly_ins = monthly_ins.rename(columns={monthly_ins.columns[0]: 'ds'})

                predicted_sales = results.get('prophet_forecast', pd.DataFrame()).iloc[0]['Prophet_Forecast'] \
                    if results.get('prophet_forecast') is not None and len(results.get('prophet_forecast', [])) > 0 \
                    else monthly_ins['Sales'].mean()

                current_stock_total = sum(stock_data.values()) if stock_data else 500

                business_insights = generate_all_insights(monthly_ins, predicted_sales, current_stock_total)
            except Exception as e:
                business_insights = [f"⚠️ Could not generate some insights: {str(e)}"]
        
        # Store everything in session state
        st.session_state['results'] = results
        st.session_state['timeline'] = timeline
        st.session_state['selected_label'] = selected_label
        st.session_state['future_periods'] = future_periods
        st.session_state['timeline_summary'] = timeline_summary
        st.session_state['inventory_report'] = inventory_report
        st.session_state['pricing_report'] = pricing_report
        st.session_state['gst_report'] = gst_report
        st.session_state['reorder_plan'] = reorder_plan
        st.session_state['stock_data'] = stock_data
        st.session_state['current_prices'] = current_prices
        st.session_state['business_insights'] = business_insights
        st.session_state['analysis_complete'] = True
        st.session_state['raw_df'] = df
        
        assistant_data = {
            'product_predictions': product_preds,
            'inventory_report': inventory_report,
            'pricing_report': pricing_report,
            'gst_report': gst_report,
            'prophet_forecast': results.get('prophet_forecast'),
            'timeline': timeline
        }
        st.session_state['ai_assistant'] = RetailAIAssistant(assistant_data)
        
        progress_bar.empty()
        status.empty()
        
        st.success(f"✅ Analysis complete! Forecasting {selected_label} ahead. Explore the tabs below 👇")
        st.balloons()
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        
        with st.expander("🐛 Debug Info"):
            import traceback
            st.code(traceback.format_exc())
        
        st.stop()

# Check if analysis is complete
if 'analysis_complete' not in st.session_state or not st.session_state['analysis_complete']:
    st.stop()

# Get results from session state
results = st.session_state['results']
timeline = st.session_state.get('timeline', 'monthly')
selected_label = st.session_state.get('selected_label', '12 Months')
timeline_summary = st.session_state.get('timeline_summary', {})
inventory_report = st.session_state['inventory_report']
pricing_report = st.session_state['pricing_report']
gst_report = st.session_state['gst_report']
ai_assistant = st.session_state.get('ai_assistant')
business_insights = st.session_state.get('business_insights', [])

# -------------------------------------------------
# TABS  
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏠 Dashboard",
    "📅 Timeline Analysis",
    "📦 Product Intelligence",
    "📊 Inventory Control",
    "💰 Pricing Strategy",
    "🧾 GST Analytics",
    "💡 Insights",          
    "🤖 AI Assistant"
])

# -------------------------------------------------
# TAB 1: DASHBOARD
# -------------------------------------------------
with tab1:
    st.header("📊 Business Overview Dashboard")
    
    if ai_assistant:
        insights = ai_assistant.get_smart_insights()
        
        if insights:
            st.subheader("🔔 Smart Alerts")
            
            for insight in insights:
                alert_class = {
                    'alert': 'alert-critical',
                    'success': 'alert-success',
                    'opportunity': 'alert-warning'
                }.get(insight['type'], '')
                
                st.markdown(f"""
                <div class="insight-box {alert_class}">
                    <h4>{insight['title']}</h4>
                    <p>{insight['message']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    st.subheader("📈 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    product_preds = results.get('product_predictions', {})
    prophet_forecast = results.get('prophet_forecast')
    
    with col1:
        st.metric("Total Products", len(product_preds))
    
    with col2:
        if prophet_forecast is not None and len(prophet_forecast) > 0:
            next_month = prophet_forecast.iloc[0]['Prophet_Forecast']
            st.metric("Next Month Forecast", f"₹{next_month:,.0f}")
    
    with col3:
        critical_count = sum(1 for data in inventory_report.values() if data['status'] == 'CRITICAL')
        st.metric("Critical Stock Items", critical_count, delta=f"-{critical_count}" if critical_count > 0 else "0")
    
    with col4:
        total_gst = sum(data['net_gst_payable'] for data in gst_report.values()) if gst_report else 0
        st.metric("Total GST Payable", f"₹{total_gst:,.0f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if product_preds:
            top_10 = dict(sorted(product_preds.items(), key=lambda x: x[1]['latest_prediction'], reverse=True)[:10])
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(top_10.keys()),
                    y=[data['latest_prediction'] for data in top_10.values()],
                    marker=dict(
                        color=[data['latest_prediction'] for data in top_10.values()],
                        colorscale='Viridis'
                    )
                )
            ])
            
            fig.update_layout(
                title="🏆 Top 10 Products by Predicted Sales",
                xaxis_title="Product",
                yaxis_title="Predicted Sales (₹)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if inventory_report:
            status_counts = {}
            for data in inventory_report.values():
                status = data['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(status_counts.keys()),
                    values=list(status_counts.values()),
                    hole=0.4,
                    marker=dict(colors=['#ff6b6b', '#feca57', '#48c6ef', '#1dd1a1'])
                )
            ])
            
            fig.update_layout(
                title="📦 Inventory Status Distribution",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 2: TIMELINE ANALYSIS
# -------------------------------------------------
with tab2:
    st.header(f"📅 Sales Forecast — Next {selected_label}")
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 2rem;'>
        <h3>📆 Monthly Forecast</h3>
        <p>Showing predictions for the next <strong>{selected_label}</strong> ({future_periods} monthly periods)</p>
    </div>
    """, unsafe_allow_html=True)
    
    if timeline_summary:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Next Month", f"₹{timeline_summary.get('next_period_forecast', 0):,.0f}")
        
        with col2:
            st.metric("Avg per Month", f"₹{timeline_summary.get('avg_forecast', 0):,.0f}")
        
        with col3:
            st.metric(f"Total over {selected_label}", f"₹{timeline_summary.get('total_forecast', 0):,.0f}")
        
        with col4:
            st.metric("Products Analysed", timeline_summary.get('total_products', 0))
    
    st.subheader(f"🔮 {selected_label} Forecast Chart")
    
    prophet_forecast = results.get('prophet_forecast')
    total_predictions = results.get('total_predictions')
    
    if prophet_forecast is not None and len(prophet_forecast) > 0:
        fig = go.Figure()
        
        if total_predictions is not None:
            fig.add_trace(go.Scatter(
                x=total_predictions['ds'],
                y=total_predictions['Predicted_Sales'],
                mode='lines+markers',
                name='Historical Predictions',
                line=dict(color='#667eea', width=2),
                marker=dict(size=6)
            ))
        
        fig.add_trace(go.Scatter(
            x=prophet_forecast['ds'],
            y=prophet_forecast['Prophet_Forecast'],
            mode='lines+markers',
            name=f'Forecast ({selected_label})',
            line=dict(color='#ff6b6b', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
        
        fig.update_layout(
            title=f"Monthly Sales Forecast — Next {selected_label}",
            xaxis_title="Month",
            yaxis_title="Sales (₹)",
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Month-by-Month Forecast Table")
        
        forecast_display = prophet_forecast.copy()
        forecast_display['Month'] = forecast_display['ds'].dt.strftime('%B %Y')
        forecast_display['Forecast (₹)'] = forecast_display['Prophet_Forecast'].apply(lambda x: f"₹{x:,.0f}")
        forecast_display['Month-on-Month Growth'] = prophet_forecast['Prophet_Forecast'].pct_change() * 100
        forecast_display['Month-on-Month Growth'] = forecast_display['Month-on-Month Growth'].apply(
            lambda x: f"{x:+.1f}%" if not pd.isna(x) else "—"
        )
        
        st.dataframe(
            forecast_display[['Month', 'Forecast (₹)', 'Month-on-Month Growth']],
            use_container_width=True,
            height=400
        )
        
        csv = prophet_forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            f"⬇️ Download {selected_label} Forecast (CSV)",
            csv,
            f"forecast_{selected_label.replace(' ', '_').lower()}.csv",
            "text/csv",
            use_container_width=True
        )

# -------------------------------------------------
# TAB 3: PRODUCT INTELLIGENCE
# -------------------------------------------------
with tab3:
    st.header("🛍️ Product-Level Intelligence")
    
    if not product_preds:
        st.warning("⚠️ Product-level data not available. Dataset may not contain product information.")
    else:
        selected_product = st.selectbox(
            "Select Product to Analyse",
            options=list(product_preds.keys())
        )
        
        if selected_product:
            product_data = product_preds[selected_product]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Predicted Sales", f"₹{product_data['latest_prediction']:,.0f}")
            
            with col2:
                if selected_product in inventory_report:
                    st.metric("Stock Status", inventory_report[selected_product]['status'])
            
            with col3:
                if selected_product in pricing_report:
                    st.metric("Current Price", f"₹{pricing_report[selected_product]['current_price']:.2f}")
            
            with col4:
                if selected_product in gst_report:
                    st.metric("GST Payable", f"₹{gst_report[selected_product]['net_gst_payable']:.2f}")
            
            if 'data' in product_data:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=product_data['data']['ds'],
                    y=product_data['data']['Sales'],
                    mode='lines+markers',
                    name='Historical Sales',
                    line=dict(color='#667eea', width=3)
                ))
                
                fig.update_layout(
                    title=f"📈 Sales Trend: {selected_product}",
                    xaxis_title="Date",
                    yaxis_title="Sales (₹)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📦 Inventory Details")
                if selected_product in inventory_report:
                    st.json(inventory_report[selected_product])
            
            with col2:
                st.markdown("### 💰 Pricing Details")
                if selected_product in pricing_report:
                    st.json(pricing_report[selected_product])
        
        st.markdown("---")
        st.subheader("📋 All Products Summary")
        
        summary_data = []
        for product, pred_data in product_preds.items():
            row = {'Product': product, 'Predicted Sales': f"₹{pred_data['latest_prediction']:,.0f}"}
            
            if product in inventory_report:
                row['Stock Status'] = inventory_report[product]['status']
                row['Stock Days'] = f"{inventory_report[product]['stock_coverage_days']:.1f}"
            
            if product in pricing_report:
                row['Price Action'] = pricing_report[product]['suggested_action']
            
            summary_data.append(row)
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, height=400)

# -------------------------------------------------
# TAB 4: INVENTORY CONTROL
# -------------------------------------------------
with tab4:
    st.header("📦 Inventory Management & Control")
    
    if not inventory_report:
        st.warning("⚠️ Inventory data not available")
    else:
        critical_items = {k: v for k, v in inventory_report.items() if v['status'] == 'CRITICAL'}
        
        if critical_items:
            st.markdown("### 🚨 Critical Stock Alert")
            
            for product, data in critical_items.items():
                st.markdown(f"""
                <div class="product-card alert-critical">
                    <h4>⚠️ {product}</h4>
                    <p><strong>Current Stock:</strong> {data['current_stock']:.0f} units</p>
                    <p><strong>Days Remaining:</strong> {data['stock_coverage_days']:.1f} days</p>
                    <p><strong>Action:</strong> {data['action']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        if st.session_state.get('reorder_plan'):
            st.markdown("### 📋 Reorder Recommendations")
            reorder_df = pd.DataFrame(st.session_state['reorder_plan']).T
            st.dataframe(reorder_df, use_container_width=True)
        
        st.markdown("### 🗺️ Stock Coverage Heatmap")
        
        products = list(inventory_report.keys())
        coverage_days = [inventory_report[p]['stock_coverage_days'] for p in products]
        
        fig = go.Figure(data=go.Bar(
            y=products,
            x=coverage_days,
            orientation='h',
            marker=dict(color=coverage_days, colorscale='RdYlGn', colorbar=dict(title="Days"))
        ))
        
        fig.update_layout(
            title="Stock Coverage by Product",
            xaxis_title="Days of Stock",
            yaxis_title="Product",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 5: PRICING STRATEGY
# -------------------------------------------------
with tab5:
    st.header("💰 Dynamic Pricing Strategy")
    
    if not pricing_report:
        st.warning("⚠️ Pricing data not available")
    else:
        increase_products = {k: v for k, v in pricing_report.items() if v['suggested_action'] == 'INCREASE PRICE'}
        
        if increase_products:
            st.markdown("### 📈 Price Increase Opportunities")
            
            for product, data in list(increase_products.items())[:5]:
                potential_profit = (data['adjusted_price'] - data['cost_price']) * data['predicted_sales']
                current_profit = (data['current_price'] - data['cost_price']) * data['predicted_sales']
                profit_increase = potential_profit - current_profit
                
                st.markdown(f"""
                <div class="product-card alert-success">
                    <h4>💎 {product}</h4>
                    <p><strong>Current Price:</strong> ₹{data['current_price']:.2f}</p>
                    <p><strong>Can Increase To:</strong> ₹{data['current_price'] * 1.1:.2f}</p>
                    <p><strong>Additional Profit:</strong> ₹{profit_increase:,.0f}</p>
                    <p><strong>Reason:</strong> {data['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        reduce_products = {k: v for k, v in pricing_report.items() if v['suggested_action'] == 'REDUCE PRICE'}
        
        if reduce_products:
            st.markdown("### 💸 Discount Recommendations")
            
            for product, data in list(reduce_products.items())[:5]:
                st.markdown(f"""
                <div class="product-card alert-warning">
                    <h4>📉 {product}</h4>
                    <p><strong>Current Price:</strong> ₹{data['current_price']:.2f}</p>
                    <p><strong>Suggested Price:</strong> ₹{data['adjusted_price']:.2f}</p>
                    <p><strong>Discount:</strong> {data['suggested_discount']}%</p>
                    <p><strong>Reason:</strong> {data['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Pricing Analysis")
        
        products = list(pricing_report.keys())[:10]
        current_prices_list = [pricing_report[p]['current_price'] for p in products]
        suggested_prices_list = [pricing_report[p]['adjusted_price'] for p in products]
        
        fig = go.Figure(data=[
            go.Bar(name='Current', x=products, y=current_prices_list),
            go.Bar(name='Suggested', x=products, y=suggested_prices_list)
        ])
        
        fig.update_layout(
            barmode='group',
            title="Current vs Suggested Pricing",
            xaxis_title="Product",
            yaxis_title="Price (₹)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TAB 6: GST ANALYTICS
# -------------------------------------------------
with tab6:
    st.header("🧾 GST Analytics & Compliance")
    
    if not gst_report:
        st.warning("⚠️ GST data not available")
    else:
        total_output_gst = sum(data['output_gst'] for data in gst_report.values())
        total_input_gst = sum(data['input_tax_credit'] for data in gst_report.values())
        total_net_gst = sum(data['net_gst_payable'] for data in gst_report.values())
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Output GST", f"₹{total_output_gst:,.2f}")
        
        with col2:
            st.metric("Total Input Credit", f"₹{total_input_gst:,.2f}")
        
        with col3:
            st.metric("Net GST Payable", f"₹{total_net_gst:,.2f}")
        
        st.markdown("### 💰 Top GST Contributors")
        
        top_gst = sorted(gst_report.items(), key=lambda x: x[1]['net_gst_payable'], reverse=True)[:10]
        
        gst_products = [p[0] for p in top_gst]
        gst_amounts = [p[1]['net_gst_payable'] for p in top_gst]
        
        fig = go.Figure(data=[
            go.Bar(
                x=gst_amounts,
                y=gst_products,
                orientation='h',
                marker=dict(color=gst_amounts, colorscale='Blues')
            )
        ])
        
        fig.update_layout(
            title="Top 10 Products by GST Contribution",
            xaxis_title="Net GST Payable (₹)",
            yaxis_title="Product",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📋 Product-wise GST Details")
        
        gst_df = pd.DataFrame(gst_report).T.sort_values('net_gst_payable', ascending=False)
        st.dataframe(gst_df, use_container_width=True, height=400)
        
        csv = gst_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            "⬇️ Download GST Report",
            csv,
            "gst_report.csv",
            "text/csv",
            use_container_width=True
        )

# -------------------------------------------------
# TAB 7: INSIGHTS  
# -------------------------------------------------
with tab7:
    st.header("💡 Business Insights")
    
    st.markdown("""
    Automatically generated analysis of your sales patterns, seasonality, discount effectiveness,
    and inventory health — directly from your data.
    """)
    
    if not business_insights:
        st.info("ℹ️ Insights will appear here after analysis. Upload a dataset and click **Analyse My Business**.")
    else:
        # Categorise insights by emoji prefix for visual styling
        def classify_insight(text):
            if text.startswith("📈"):
                return "insight-up"
            elif text.startswith("📉"):
                return "insight-down"
            elif text.startswith("🚨") or text.startswith("⚠️"):
                return "insight-warn"
            elif text.startswith("✅"):
                return "insight-up"
            elif text.startswith("📦"):
                return "insight-warn"
            else:
                return "insight-info"

        st.subheader(f"📊 {len(business_insights)} Insight(s) Found")

        for insight in business_insights:
            css_class = classify_insight(insight)
            st.markdown(f'<div class="insight-card {css_class}">{insight}</div>', unsafe_allow_html=True)

# -------------------------------------------------
# TAB 8: AI ASSISTANT
# -------------------------------------------------
with tab8:
    st.header("🤖 AI Business Assistant")
    
    st.markdown("""
    Ask me anything about your business! Examples:
    - "What are my top performing products?"
    - "Which products need reordering?"
    - "Show me products that should be discounted"
    - "What's my sales forecast for next month?"
    - "What's my total GST liability?"
    """)
    
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    user_question = st.text_input("💬 Ask a question:", placeholder="e.g., Which products are running out of stock?")
    
    if st.button("Send") and user_question:
        if ai_assistant:
            response = ai_assistant.answer_question(user_question)
            st.session_state['chat_history'].append({
                'user': user_question,
                'assistant': response
            })
    
    if st.session_state['chat_history']:
        st.markdown("### 💬 Conversation")
        
        for chat in reversed(st.session_state['chat_history'][-10:]):
            st.markdown(f"""
            <div class="chat-message chat-user">
                <strong>You:</strong> {chat['user']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="chat-message">
                <strong>🤖 Assistant:</strong><br>{chat['assistant']}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Show Top Products", use_container_width=True):
            if ai_assistant:
                st.markdown(ai_assistant.answer_question("What are my top products?"))
    
    with col2:
        if st.button("Check Inventory", use_container_width=True):
            if ai_assistant:
                st.markdown(ai_assistant.answer_question("Show me inventory status"))
    
    with col3:
        if st.button("Get Forecast", use_container_width=True):
            if ai_assistant:
                st.markdown(ai_assistant.answer_question("What's my sales forecast?"))

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Smart Retail Intelligence System</strong> | Multi-Timeline Forecasting | Product-Level Analytics</p>
    <p>Built with ML & Advanced Business Intelligence</p>
</div>
""", unsafe_allow_html=True)