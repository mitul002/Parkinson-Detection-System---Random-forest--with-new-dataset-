import os
import joblib
import numpy as np
import pandas as pd
from feature_extraction import SpiralFeatureExtractor

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE, 'random_forest_model.pkl')
SCALER_PATH = os.path.join(BASE, 'scaler.pkl')
IMPUTER_PATH = os.path.join(BASE, 'imputer.pkl')
LE_PATH = os.path.join(BASE, 'label_encoder.pkl')
FN_PATH = os.path.join(BASE, 'feature_names.pkl')


def assemble_features(img_path):
    extractor = SpiralFeatureExtractor(use_cnn=True)
    f = extractor.extract_features(img_path)
    if f is None:
        raise RuntimeError('Feature extraction returned None')
    feat_dict = {}
    for i, v in enumerate(f['cnn_features']):
        feat_dict[f'cnn_feat_{i}'] = v
    feat_dict.update(f['handcrafted_features'])
    return pd.DataFrame([feat_dict])


def main(img_path):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    le = joblib.load(LE_PATH)
    feature_names = joblib.load(FN_PATH)

    df = assemble_features(img_path)

    # Add skeleton feature aliases to satisfy both naming variants if needed
    alias_pairs = [
        ("Skel_Endpoints", "skel_endpoints"),
        ("Skel_Branchpoints", "skel_branchpoints"),
        ("Skel_Length", "skel_length"),
    ]
    for a, b in alias_pairs:
        if a not in df.columns and b in df.columns:
            df[a] = df[b]
        if b not in df.columns and a in df.columns:
            df[b] = df[a]

    # Check for missing/extra columns
    missing = [c for c in feature_names if c not in df.columns]
    extra = [c for c in df.columns if c not in feature_names]
    print('Total features expected:', len(feature_names))
    print('Extracted columns:', len(df.columns))
    if missing:
        print('MISSING:', missing)
    if extra:
        print('EXTRA:', extra[:10], '... total extra:', len(extra))

    # Align columns
    df = df.reindex(columns=feature_names)

    X = imputer.transform(df)
    X = scaler.transform(X)

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]
    print('Prediction:', label)
    print('Probabilities:', dict(zip(le.classes_, proba)))


if __name__ == '__main__':
    test_img = os.path.join(BASE, 'abc.png')
    if not os.path.exists(test_img):
        raise SystemExit('Test image abc.png not found in model directory')
    main(test_img)
