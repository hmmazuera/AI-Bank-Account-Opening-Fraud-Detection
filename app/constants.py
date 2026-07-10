from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "dnn_model.keras"
SCALER_PATH = BASE_DIR / "data" / "processed" / "scaler.pkl"
X_TRAIN_PATH = BASE_DIR / "data" / "processed" / "X_train.pkl"

GITHUB_URL = ("https://github.com/hmmazuera/AI-Bank-Account-Opening-Fraud-Detection")

LINKEDIN_URL = ("https://www.linkedin.com/in/mauricio-mazuera-a0a7a933b/")

PAYMENT_TYPES = ["AB", "AA", "AC", "AD", "AE"]

EMPLOYMENT_STATUSES = ["CA","CB","CF","CC","CD","CE","CG"]

HOUSING_STATUSES = ["BC","BB","BA","BE","BD","BF","BG"]

APPLICATION_SOURCES = ["INTERNET","TELEAPP"]

DEVICE_OS_OPTIONS = ["other","linux","windows","macintosh","x11"]

INCOME_OPTIONS = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

CUSTOMER_AGE_OPTIONS = [10,20,30,40,50,60,70,80,90]

PROPOSED_CREDIT_LIMIT_OPTIONS = [190.0,200.0,210.0,490.0,500.0,510.0,990.0,1000.0,1500.0,1900.0,2000.0,2100.0]

BINARY_OPTIONS = [0, 1]

DEVICE_DISTINCT_EMAIL_OPTIONS = [-1, 0, 1, 2]

MONTH_OPTIONS = list(range(8))

CATEGORICAL_MAPPINGS = {
    "payment_type": {
        "AA": 0,
        "AB": 1,
        "AC": 2,
        "AD": 3,
        "AE": 4,
    },
    "employment_status": {
        "CA": 0,
        "CB": 1,
        "CC": 2,
        "CD": 3,
        "CE": 4,
        "CF": 5,
        "CG": 6,
    },
    "housing_status": {
        "BA": 0,
        "BB": 1,
        "BC": 2,
        "BD": 3,
        "BE": 4,
        "BF": 5,
        "BG": 6,
    },
    "source": {
        "INTERNET": 0,
        "TELEAPP": 1,
    },
    "device_os": {
        "linux": 0,
        "macintosh": 1,
        "other": 2,
        "windows": 3,
        "x11": 4,
    },
}

ELEVATED_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.50