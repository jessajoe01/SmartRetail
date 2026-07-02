"""
Configuration file for Smart Retail Analytics
"""
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent

# Paths
DATA_PATH = PROJECT_ROOT / "data"
MODELS_PATH = PROJECT_ROOT / "models"
BUSINESS_MODULES_PATH = PROJECT_ROOT / "business_modules"
ML_PIPELINE_PATH = PROJECT_ROOT / "ml_pipeline"

# GST Configuration
GST_SLABS = {
    '5%': 0.05,
    '12%': 0.12,
    '18%': 0.18,
    '28%': 0.28
}

# Inventory thresholds
STOCKOUT_THRESHOLD = 0.1    # 10% of monthly average
OVERSTOCK_THRESHOLD = 2.0   # 200% of monthly average
REORDER_POINT = 0.3         # 30% of monthly average

# Model settings
FORECAST_PERIODS = {
    'daily': 30,
    'weekly': 12,
    'monthly': 6,
    'yearly': 2
}

# UI Settings
THEME_CONFIG = {
    'primary_color': '#1E3A8A',
    'secondary_color': '#3B82F6',
    'success_color': '#10B981',
    'warning_color': '#F59E0B',
    'danger_color': '#EF4444'
}