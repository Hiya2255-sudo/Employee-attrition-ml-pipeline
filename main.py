from pathlib import Path
from datetime import datetime, timezone

from src.preprocessing import preprocess_data
from src.models.random_forest import train_rf_baseline, train_rf_tuned
from src.models.xgboost_model import train_xgb
from src.utils import save_model, save_json

# Ensure output folders exist
Path('dataset/processed').mkdir(parents=True, exist_ok=True)
Path('models').mkdir(parents=True, exist_ok=True)

raw_path = 'dataset/raw/Employee_Attrition_DataSet.csv'

# Step 1: Preprocess
cleaned_df, preprocessor, X_raw, y_raw = preprocess_data(raw_path)

# Save cleaned data
cleaned_df.to_csv('dataset/processed/cleaned_data.csv', index=False)

# Step 2: Train all candidate models
rf_pipeline, rf_metrics = train_rf_baseline(preprocessor, X_raw, y_raw)
rf_tuned_pipeline, rf_tuned_metrics = train_rf_tuned(preprocessor, X_raw, y_raw)
xgb_pipeline, xgb_metrics = train_xgb(preprocessor, X_raw, y_raw)

# Save all trained models
save_model(rf_pipeline, 'models/rf_baseline.pkl')
save_model(rf_tuned_pipeline, 'models/rf_tuned.pkl')
save_model(xgb_pipeline, 'models/xgb_model.pkl')

# Select the best model by F1 score
best_name = 'rf_baseline'
best_pipeline = rf_pipeline
best_metrics = rf_metrics

for name, metrics, pipeline in [
    ('rf_tuned', rf_tuned_metrics, rf_tuned_pipeline),
    ('xgboost', xgb_metrics, xgb_pipeline),
]:
    if metrics['f1'] > best_metrics['f1']:
        best_name = name
        best_metrics = metrics
        best_pipeline = pipeline

save_model(best_pipeline, 'models/best_model.pkl')

# Step 3: Save comparison metadata
model_info = {
    'created_at': datetime.now(timezone.utc).isoformat(),
    'cleaned_data_path': 'dataset/processed/cleaned_data.csv',
    'best_model': best_name,
    'best_model_path': 'models/best_model.pkl',
    'comparison': {
        'rf_baseline': rf_metrics,
        'rf_tuned': rf_tuned_metrics,
        'xgboost': xgb_metrics,
    },
}

save_json(model_info, 'models/model_info.json')