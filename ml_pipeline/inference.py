import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path

# Import from preprocessing - NO DUPLICATES!
from business_modules.insights import generate_all_insights
from ml_pipeline.preprocessing import (
    clean_dataset,
    map_columns,
    validate_columns,
    create_timeline_aggregation,
    add_timeline_features,
    get_top_products,
    get_product_statistics
)

# Business modules 
from business_modules.insights import generate_all_insights
from business_modules.inventory import (
    product_inventory_status, 
    calculate_reorder_quantities,
    get_inventory_summary,
    get_critical_products
)
from business_modules.pricing import (
    product_pricing_strategy,
    get_pricing_summary,
    get_top_pricing_opportunities,
    calculate_price_optimization_impact
)
from business_modules.gst import (
    product_gst_summary,
    get_gst_aggregate_summary,
    project_gst_timeline,
    get_gst_filing_alerts
)


def load_models():
    """Load trained ML models"""
    models = {}
    models_dir = Path("models")
    
    try:
        models["lr"] = load(models_dir / "linear_regression.pkl")
        models["rf"] = load(models_dir / "random_forest.pkl")
        models["xgb"] = load(models_dir / "xgboost.pkl")
        models["prophet"] = load(models_dir / "prophet_model.pkl")
        return models
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"❌ Model files not found. Please train models first using training_pipeline.ipynb\n{e}"
        )


def run_multi_timeline_inference(uploaded_df, timeline='monthly', future_periods=12, selected_products=None):
    """
    Complete product-level inference with flexible timeline
    
    Parameters:
    uploaded_df: DataFrame - raw uploaded data
    timeline: str - 'daily', 'weekly', 'monthly', 'yearly'
    future_periods: int - number of periods to forecast
    selected_products: list - specific products to analyze (optional)
    
    Returns:
    dict with predictions at chosen timeline
    """
    
    print(f"📊 Starting {timeline} analysis...")
    
    # --- STEP 1: PREPROCESSING ---
    df = clean_dataset(uploaded_df)
    df = map_columns(df)
    validate_columns(df)
    
    # --- STEP 2: TIMELINE AGGREGATION ---
    agg_result = create_timeline_aggregation(df, timeline)
    
    has_products = agg_result['has_products']
    product_level_data = agg_result['product_level']
    total_level_data = agg_result['total_level']
    
    # --- STEP 3: DATA VALIDATION ---
    min_periods = {
        'daily': 90,
        'weekly': 26,
        'monthly': 12,
        'yearly': 3
    }
    
    min_required = min_periods.get(timeline, 12)
    
    if len(total_level_data) < min_required:
        raise ValueError(
            f"❌ Insufficient data for {timeline} analysis. "
            f"Found {len(total_level_data)} periods, need at least {min_required}."
        )
    
    # --- STEP 4: LOAD MODELS ---
    models = load_models()
    
    results = {
        'timeline': timeline,
        'has_products': has_products,
        'product_predictions': {},
        'category_predictions': {},
        'total_predictions': None,
        'prophet_forecast': None
    }
    
    # --- STEP 5: TOTAL LEVEL PREDICTIONS ---
    print(f"📈 Generating {timeline} predictions...")
    
    total_features = add_timeline_features(total_level_data, timeline)
    
    # Get feature columns dynamically
    feature_cols = [col for col in total_features.columns 
                   if col.startswith('lag_') or col.startswith('ma_') or 
                   col in ['year', 'month', 'quarter', 'day_of_week', 'week_of_year', 
                          'day_of_month', 'is_weekend']]
    
    # Ensure basic features
    for col in ['year', 'month', 'quarter']:
        if col not in feature_cols and col in total_features.columns:
            feature_cols.append(col)
    
    # ADD MISSING FEATURES: Profit and Discount (if they were in training data)
    if 'Profit' in total_level_data.columns:
        total_features['Profit'] = total_level_data['Profit']
        feature_cols.append('Profit')
    
    if 'Discount' in total_level_data.columns:
        total_features['Discount'] = total_level_data['Discount']
        feature_cols.append('Discount')
    
    # CRITICAL FIX: Get the exact features the model was trained on
    try:
        # Try to get feature names from the trained model
        if hasattr(models['lr'], 'feature_names_in_'):
            model_features = list(models['lr'].feature_names_in_)
        else:
            # Fallback to common features
            model_features = feature_cols
    except:
        model_features = feature_cols
    
    # Create feature matrix with ALL required features
    X_total = pd.DataFrame()
    for col in model_features:
        if col in total_features.columns:
            X_total[col] = total_features[col]
        else:
            # Add missing feature with default value 0
            X_total[col] = 0
            print(f"⚠️ Adding missing feature '{col}' with default value 0")
    
    # Ensure columns are in the same order as training
    X_total = X_total[model_features]
    
    # ML predictions (ensemble)
    ml_predictions = total_features[['ds']].copy()
    ml_predictions['Linear_Regression'] = models['lr'].predict(X_total)
    ml_predictions['Random_Forest'] = models['rf'].predict(X_total)
    ml_predictions['XGBoost'] = models['xgb'].predict(X_total)
    ml_predictions['Predicted_Sales'] = ml_predictions[
        ['Linear_Regression', 'Random_Forest', 'XGBoost']
    ].mean(axis=1)
    
    results['total_predictions'] = ml_predictions
    
    # --- STEP 6: PROPHET FORECAST ---
    freq_map = {
        'daily': 'D',
        'weekly': 'W-MON',
        'monthly': 'MS',
        'yearly': 'YS'
    }
    
    prophet_input = total_level_data[['ds', 'Sales']].rename(columns={'Sales': 'y'})
    
    try:
        # Check if we have enough data for Prophet
        if len(prophet_input) < 2:
            print(f"⚠️ Insufficient data for Prophet ({len(prophet_input)} periods). Need at least 2.")
            results['prophet_forecast'] = pd.DataFrame(columns=['ds', 'Prophet_Forecast'])
        else:
            future = models['prophet'].make_future_dataframe(
                periods=future_periods,
                freq=freq_map[timeline]
            )
            
            forecast = models['prophet'].predict(future)
            prophet_forecast = forecast[['ds', 'yhat']].rename(columns={'yhat': 'Prophet_Forecast'})
            
            # Extract only future predictions
            prophet_forecast = prophet_forecast[
                prophet_forecast['ds'] > total_level_data['ds'].max()
            ].reset_index(drop=True)
            
            # Check if forecasts are realistic (not all zeros or negative)
            if prophet_forecast['Prophet_Forecast'].sum() == 0:
                print("⚠️ Prophet returned zero forecasts. Using ML model averages instead.")
                # Fallback: use average of ML predictions
                avg_prediction = ml_predictions['Predicted_Sales'].mean()
                prophet_forecast['Prophet_Forecast'] = avg_prediction
            
            # Ensure no negative forecasts
            prophet_forecast['Prophet_Forecast'] = prophet_forecast['Prophet_Forecast'].clip(lower=0)
            
            results['prophet_forecast'] = prophet_forecast
        
    except Exception as e:
        print(f"⚠️ Prophet forecast failed: {str(e)}")
        print("Using fallback: average of ML predictions")
        
        # Fallback: create forecast using ML model average
        if len(ml_predictions) > 0:
            avg_prediction = ml_predictions['Predicted_Sales'].mean()
            last_date = total_level_data['ds'].max()
            
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1 if timeline == 'daily' else 7 if timeline == 'weekly' else 30),
                periods=future_periods,
                freq=freq_map[timeline]
            )
            
            prophet_forecast = pd.DataFrame({
                'ds': future_dates,
                'Prophet_Forecast': [avg_prediction] * future_periods
            })
            
            results['prophet_forecast'] = prophet_forecast
        else:
            results['prophet_forecast'] = pd.DataFrame(columns=['ds', 'Prophet_Forecast'])
    
    # --- STEP 7: PRODUCT LEVEL PREDICTIONS ---
    if has_products and product_level_data is not None:
        print(f"🛍️ Generating product-level {timeline} predictions...")
        
        # Check if Product column actually exists
        if 'Product' not in product_level_data.columns:
            print("⚠️ No Product column found. Skipping product-level analysis.")
            has_products = False
        else:
            if selected_products is None:
                top_products = product_level_data.groupby('Product')['Sales'].sum().nlargest(20).index.tolist()
            else:
                top_products = selected_products
            
            for product in top_products:
                product_data = product_level_data[
                    product_level_data['Product'] == product
                ][['ds', 'Sales']].copy()
                
                if len(product_data) >= max(3, min_required // 3):
                    try:
                        product_features = add_timeline_features(product_data, timeline)
                        
                        if len(product_features) >= 3:
                            # Build feature matrix with ALL model features
                            X_product = pd.DataFrame()
                            for col in model_features:
                                if col in product_features.columns:
                                    X_product[col] = product_features[col]
                                else:
                                    X_product[col] = 0
                            
                            # Ensure column order matches training
                            X_product = X_product[model_features]
                            
                            lr_pred = models['lr'].predict(X_product)
                            rf_pred = models['rf'].predict(X_product)
                            xgb_pred = models['xgb'].predict(X_product)
                            
                            ensemble_pred = (lr_pred + rf_pred + xgb_pred) / 3
                            
                            results['product_predictions'][product] = {
                                'data': product_data,
                                'latest_prediction': ensemble_pred[-1],
                                'historical_predictions': ensemble_pred,
                                'avg_sales': product_data['Sales'].mean()
                            }
                            
                    except Exception as e:
                        print(f"⚠️ Skipping {product}: {str(e)}")
                        continue
        
        # --- STEP 8: CATEGORY LEVEL PREDICTIONS ---
        if has_products and 'Category' in product_level_data.columns:
            print(f"📂 Generating category-level {timeline} predictions...")
            
            categories = product_level_data['Category'].unique()
            
            for category in categories:
                cat_data = product_level_data[
                    product_level_data['Category'] == category
                ].groupby('ds').agg({'Sales': 'sum'}).reset_index()
                
                if len(cat_data) >= max(3, min_required // 3):
                    try:
                        cat_features = add_timeline_features(cat_data, timeline)
                        
                        if len(cat_features) >= 3:
                            # Build feature matrix with ALL model features
                            X_cat = pd.DataFrame()
                            for col in model_features:
                                if col in cat_features.columns:
                                    X_cat[col] = cat_features[col]
                                else:
                                    X_cat[col] = 0
                            
                            X_cat = X_cat[model_features]
                            
                            lr_pred = models['lr'].predict(X_cat)
                            rf_pred = models['rf'].predict(X_cat)
                            xgb_pred = models['xgb'].predict(X_cat)
                            
                            ensemble_pred = (lr_pred + rf_pred + xgb_pred) / 3
                            
                            results['category_predictions'][category] = {
                                'data': cat_data,
                                'latest_prediction': ensemble_pred[-1],
                                'historical_predictions': ensemble_pred
                            }
                            
                    except Exception as e:
                        print(f"⚠️ Skipping category {category}: {str(e)}")
                        continue
    
    print(f"✅ {timeline.capitalize()} inference complete!")
    
    return results


def get_timeline_summary(results, timeline):
    """Generates summary of predictions for the selected timeline"""
    
    timeline_labels = {
        'daily': 'Day',
        'weekly': 'Week',
        'monthly': 'Month',
        'yearly': 'Year'
    }
    
    label = timeline_labels.get(timeline, 'Period')
    
    summary = {
        'timeline': timeline,
        'label': label,
        'total_products': len(results.get('product_predictions', {})),
        'total_categories': len(results.get('category_predictions', {}))
    }
    
    prophet_forecast = results.get('prophet_forecast')
    
    if prophet_forecast is not None and len(prophet_forecast) > 0:
        summary['next_period_forecast'] = prophet_forecast.iloc[0]['Prophet_Forecast']
        summary['total_forecast'] = prophet_forecast['Prophet_Forecast'].sum()
        summary['avg_forecast'] = prophet_forecast['Prophet_Forecast'].mean()
        summary['forecast_periods'] = len(prophet_forecast)
    else:
        summary['next_period_forecast'] = 0
        summary['total_forecast'] = 0
        summary['avg_forecast'] = 0
        summary['forecast_periods'] = 0
    
    return summary


def compare_timelines(uploaded_df, timelines=['daily', 'weekly', 'monthly', 'yearly']):
    """Runs analysis on multiple timelines and compares results"""
    
    comparison = {}
    
    for timeline in timelines:
        try:
            period_map = {
                'daily': 30,
                'weekly': 12,
                'monthly': 12,
                'yearly': 3
            }
            
            periods = period_map.get(timeline, 12)
            
            results = run_multi_timeline_inference(
                uploaded_df,
                timeline=timeline,
                future_periods=periods
            )
            
            summary = get_timeline_summary(results, timeline)
            comparison[timeline] = summary
            
        except Exception as e:
            print(f"⚠️ Could not analyze {timeline} timeline: {str(e)}")
            comparison[timeline] = {
                'timeline': timeline,
                'error': str(e),
                'total_products': 0,
                'total_categories': 0
            }
    
    return comparison


def run_full_analysis(uploaded_df, 
                     timeline='monthly',
                     future_periods=12,
                     current_stock=500,
                     reorder_level=200,
                     overstock_level=1000,
                     current_price=100,
                     current_discount=10,
                     cost_price=70,
                     sales_gst_rate=18,
                     purchase_amount=0,
                     purchase_gst_rate=0):
    """
    Complete business analysis with all enhanced modules
    
    Returns:
    dict with all analysis results including enhanced summaries
    """
    
    # Run timeline inference
    results = run_multi_timeline_inference(
        uploaded_df,
        timeline=timeline,
        future_periods=future_periods
    )
    
    product_preds = results.get('product_predictions', {})
    
    # Convert timeline predictions to daily for inventory
    timeline_to_days = {
        'daily': 1,
        'weekly': 7,
        'monthly': 30,
        'yearly': 365
    }
    
    days_multiplier = timeline_to_days.get(timeline, 30)
    
    # Initialize reports
    inventory_report = {}
    inventory_summary = {}
    reorder_plan = {}
    pricing_report = {}
    pricing_summary = {}
    gst_report = {}
    gst_summary = {}
    gst_timeline = None
    stock_data = {}
    current_prices = {}
    
    if product_preds:
        # Generate mock data (in production, user provides this)
        stock_data = {product: np.random.randint(100, 1000) for product in product_preds.keys()}
        current_prices = {product: np.random.uniform(50, 500) for product in product_preds.keys()}
        cost_prices = {product: price * 0.7 for product, price in current_prices.items()}
        
        # Adjust predictions for daily rate (for inventory)
        adjusted_preds = {}
        for product, pred_data in product_preds.items():
            adjusted_preds[product] = {
                'latest_prediction': pred_data['latest_prediction'] / days_multiplier,
                'historical_predictions': pred_data['historical_predictions'],
                'avg_sales': pred_data.get('avg_sales', pred_data['latest_prediction'])
            }
        
        # ===== GENERATE BUSINESS REPORTS =====
        
        # Inventory
        inventory_report = product_inventory_status(adjusted_preds, stock_data, period_days=30)
        reorder_plan = calculate_reorder_quantities(adjusted_preds, stock_data)
        inventory_summary = get_inventory_summary(inventory_report)
        
        # Pricing
        pricing_report = product_pricing_strategy(product_preds, current_prices, cost_prices)
        pricing_summary = get_pricing_summary(pricing_report)
        
        # GST
        gst_report = product_gst_summary(product_preds, sales_gst_rate)
        gst_summary = get_gst_aggregate_summary(gst_report)
        
        # GST Timeline (if prophet forecast available)
        prophet_forecast = results.get('prophet_forecast')
        if prophet_forecast is not None and len(prophet_forecast) > 0:
            try:
                gst_timeline = project_gst_timeline(
                    prophet_forecast,
                    sales_gst_rate=sales_gst_rate,
                    purchase_ratio=0.6,
                    purchase_gst_rate=purchase_gst_rate
                )
            except Exception as e:
                print(f"⚠️ GST timeline projection failed: {str(e)}")
                gst_timeline = None
    
    # Timeline summary
    timeline_summary = get_timeline_summary(results, timeline)
    
    # ===== RETURN COMPLETE ANALYSIS =====
    return {
        'results': results,
        'timeline': timeline,
        'timeline_summary': timeline_summary,
        
        # Inventory
        'inventory_report': inventory_report,
        'inventory_summary': inventory_summary,
        'reorder_plan': reorder_plan,
        
        # Pricing
        'pricing_report': pricing_report,
        'pricing_summary': pricing_summary,
        
        # GST
        'gst_report': gst_report,
        'gst_summary': gst_summary,
        'gst_timeline': gst_timeline,     
        
        # Supporting data
        'stock_data': stock_data,
        'current_prices': current_prices
    }