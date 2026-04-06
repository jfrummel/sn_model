import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


df = pd.read_csv("data/cleaned_data.csv")


# 1. Prepare data
# Split serial number string into individual character columns,
# filling missing positions (e.g. short serials) with a placeholder string
# so CatBoost never receives a raw NaN in a categorical column.
SERIAL_LENGTH = 17
MISSING_CHAR = "?"  # placeholder for missing/short serial positions

def serial_to_chars(serial):
    chars = list(str(serial))
    # Pad or truncate to a fixed length
    chars = chars[:SERIAL_LENGTH] + [MISSING_CHAR] * max(0, SERIAL_LENGTH - len(chars))
    return pd.Series(chars)

X = df['serial'].apply(serial_to_chars)

# Ensure every cell is a plain Python string — eliminates any residual NaN
# that could sneak in from non-string serial values in the source CSV.
X = X.fillna(MISSING_CHAR).astype(str)

y = df['model']

# 2. All positions are categorical
cat_features = list(range(X.shape[1]))

# 3. Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    verbose=100,
)

model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_val, y_val),
)

# 5. Evaluate
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nDetailed Classification Report:")
print(classification_report(y_val, y_pred))


# 6. Inference

def predict_tractor_model(serial_number: str, model: CatBoostClassifier) -> str:
    """
    Takes a raw serial number string and returns the predicted tractor model.

    Applies the same preprocessing used at training time:
      - split into characters
      - pad / truncate to SERIAL_LENGTH
      - fill any gaps with MISSING_CHAR
      - cast everything to str
    """
    chars = list(str(serial_number))
    chars = chars[:SERIAL_LENGTH] + [MISSING_CHAR] * max(0, SERIAL_LENGTH - len(chars))

    formatted_input = pd.DataFrame([chars]).astype(str)

    # predict() returns a 2-D array, e.g. [['6125M']]
    prediction = model.predict(formatted_input)
    return prediction[0][0]


# 7. Persist
model.save_model("jd_tractor_model.cbm")
