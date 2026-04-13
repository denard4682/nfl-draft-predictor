from src.config import DATA_PATH, FEATURES, CLASSIFIER_PATH, REGRESSOR_PATH
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train import train_models
from src.evaluate import evaluate_classifer, evaluate_regressor
from src.utils import setup_logger


setup_logger()

df = load_data(DATA_PATH)
df = preprocess_data(df)
features = ["Height", "Weight", "40-yd Dash", "Bench Press", "Vertical Jump", "Broad Jump", "20-yd Shuttle", "3-Cone Drill", "Position", "School"]

clf, reg = train_models(df, features, CLASSIFIER_PATH, REGRESSOR_PATH)

print("✅ Models trained and saved successfully")








