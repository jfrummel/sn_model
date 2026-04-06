from catboost import CatBoostClassifier


model = CatBoostClassifier()
model.load_model("jd_tractor_model.cbm")

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

result = predict_tractor_model("1LV1023EPMM141146", model)
print(result)