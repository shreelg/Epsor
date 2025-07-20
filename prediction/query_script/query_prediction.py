import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import numpy as np
import shap


class predict_ticker:
    def __init__(self,
                 model_path="prediction\\xgb_model.json",
                 encoders_path="label_encoders.pkl",
                 csv_path="prediction\\query_script\\ticker_ds.csv"):
        self.model_path = model_path
        self.encoders_path = encoders_path
        self.csv_path = csv_path
        self.encoders = None
        self.model = None

    # load and save encoders
    def load_label_encoders(self):
        if os.path.exists(self.encoders_path):
            with open(self.encoders_path, "rb") as f:
                self.encoders = pickle.load(f)
        else:
            self.encoders = {}

    def save_label_encoders(self):
        with open(self.encoders_path, "wb") as f:
            pickle.dump(self.encoders, f)

    # preprocess
    def preprocess_numeric_columns(self, df):
        numeric_cols = ['EPS Actual', 'EPS Surprise', 'Surprise %']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

    def encode_categorical_columns(self, df):
        categorical_cols = ['Sector', 'Industry']
        if self.encoders is None:
            self.encoders = {}
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
                if col not in self.encoders:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.encoders[col] = le
                else:
                    le = self.encoders[col]
                    if "Unknown" not in le.classes_:
                        new_classes = list(le.classes_) + ["Unknown"]
                        le.classes_ = np.array(new_classes)
                    df[col] = df[col].apply(lambda x: x if x in le.classes_ else "Unknown")
                    df[col] = le.transform(df[col])
        return df

    def load_model(self):
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_path)

    # generate shap values
    def generate_shap_values(self, X):
        explainer = shap.Explainer(self.model, X)
        shap_values = explainer(X)
        return shap_values

    # main predict method
    def predict_from_csv(self, csv_path=None):
        df = pd.read_csv(self.csv_path)

        df = self.preprocess_numeric_columns(df)
        self.load_label_encoders()
        df = self.encode_categorical_columns(df)
        self.save_label_encoders()
        self.load_model()

        model_features = self.model.get_booster().feature_names
        if model_features is None:
            raise ValueError("Model does not have stored feature names.")

        for f in model_features:
            if f not in df.columns:
                print(f"Warning: Adding missing feature '{f}' as 0")
                df[f] = 0

        X = df[model_features]

        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]

        df['prediction'] = ['Buy' if p == 1 else 'No Buy' for p in preds]
        df['confidence'] = (probs * 100).round(4).astype(str) + '%'

        # FEATURE IMPORTANCE
        booster = self.model.get_booster()
        importance_dict = booster.get_score(importance_type='gain')
        feature_importance_sorted = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

        # SHAP EXPLANATIONS
        shap_values = self.generate_shap_values(X)
        explanations = []
        feature_names = X.columns.tolist()

        for i in range(len(X)):
            instance_contrib = dict(zip(feature_names, shap_values[i].values))
            explanations.append(instance_contrib)

        df['shap_explanation'] = explanations  # feature-level contribution per row

        
        self.feature_importance_global = feature_importance_sorted

        return df
