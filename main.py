from pathlib import Path

from src.preprocessing import preprocess_data
from src.model import train_model
from src.utils import save_model

# Ensure output folders exist
Path("dataset/processed").mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)

# Step 1: Preprocess
df = preprocess_data("dataset/raw/Employee_Attrition_DataSet.csv")

# Save cleaned data
df.to_csv("dataset/processed/cleaned_data.csv", index=False)

# Step 2: Train model
model = train_model(df)

# Step 3: Save model
save_model(model, "models/attrition_model.pkl")