import joblib
import numpy as np

def load_model(clf_path, reg_path):
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path)
    return clf, reg

def predict_player(clf, reg, features):
    features = np.array(features).reshape(1, -1)

    drafted = clf.predict(features)[0]

    if drafted == 1:
        pick = reg.predict(features)[0]
        return drafted, pick
    
    return drafted, None
    
