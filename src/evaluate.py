from sklearn.metrics import accuracy_score, mean_absolute_error

def evaluate_classifer(model, x_test, y_test):
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {acc}")

def evaluate_regressor(model, x_test, y_test):
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    print(f"Regressor MAE: {mae}")