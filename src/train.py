from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

def train_models(df, features, clf_path, reg_path):
    x = df[features]
    y = df["Drafted"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)

    joblib.dump(clf, clf_path)

    df_drafted = df[df["Drafted"] == 1]

    x_reg = df_drafted[features]
    y_reg = df_drafted["Pick"]

    x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x_reg, y_reg, test_size=0.2, random_state=42)

    reg = RandomForestRegressor()
    reg.fit(x_train_reg, y_train_reg)

    joblib.dump(reg, reg_path)
    
    return clf, reg
