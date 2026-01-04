"""
Retrain model with improved feature selection to avoid redundancy
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def remove_correlated_features(df, threshold=0.85):
    """Remove highly correlated features to reduce redundancy"""
    # Get numeric features only (exclude Image and Label)
    feature_cols = [col for col in df.columns if col not in ['Image', 'Label']]
    
    # Calculate correlation matrix
    corr_matrix = df[feature_cols].corr().abs()
    
    # Find highly correlated pairs
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Identify features to drop
    to_drop = set()
    
    # Prioritize keeping Mean_Thickness over Std_Thickness and Thickness_P95
    thickness_features = ['Mean_Thickness', 'Std_Thickness', 'Thickness_P95']
    thickness_in_data = [f for f in thickness_features if f in feature_cols]
    
    if len(thickness_in_data) > 1:
        print(f"\n⚠️  Found {len(thickness_in_data)} thickness features: {thickness_in_data}")
        # Keep Mean_Thickness, drop others
        for feat in thickness_in_data:
            if feat != 'Mean_Thickness':
                to_drop.add(feat)
                print(f"   Removing {feat} (highly correlated with Mean_Thickness)")
    
    # Remove other highly correlated features
    for column in upper_triangle.columns:
        if column in to_drop:
            continue
        correlated = upper_triangle[column][upper_triangle[column] > threshold].index.tolist()
        if correlated:
            # Keep the first, drop the rest
            for corr_feat in correlated:
                if corr_feat not in thickness_in_data or corr_feat != 'Mean_Thickness':
                    to_drop.add(corr_feat)
                    print(f"   Removing {corr_feat} (correlated with {column})")
    
    return list(to_drop)

def train_improved_model(data_path="spiral_feature_dataset.csv"):
    """Train Random Forest with improved feature selection"""
    print("="*70)
    print("IMPROVED PARKINSON'S DETECTION MODEL TRAINING")
    print("="*70)
    
    # Load dataset
    print("\n📊 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"   Original shape: {df.shape}")
    
    # Remove correlated features
    print("\n🔍 Analyzing feature correlations...")
    features_to_drop = remove_correlated_features(df)
    
    if features_to_drop:
        print(f"\n✂️  Dropping {len(features_to_drop)} redundant features")
        df = df.drop(columns=features_to_drop)
        print(f"   New shape: {df.shape}")
    else:
        print("\n✅ No highly correlated features found")
    
    # Separate features and target
    X = df.drop(columns=['Image', 'Label'])
    y = df['Label']
    
    # Store feature names
    feature_names = X.columns.tolist()
    print(f"\n📝 Final feature count: {len(feature_names)}")
    
    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"   Classes: {le.classes_}")
    
    # Handle missing values
    print("\n🔧 Preprocessing...")
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train Random Forest with better parameters
    print("\n🌲 Training improved Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',  # Limit features per split for diversity
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    
    print("\n📈 Model Performance:")
    print("-"*70)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    importances = rf_model.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n🎯 Top 10 Most Important Features:")
    print(feat_imp.head(10).to_string(index=False))
    
    # Check feature diversity
    top_3 = feat_imp.head(3)['Feature'].tolist()
    thickness_count = sum(1 for f in top_3 if 'Thickness' in f or 'thickness' in f)
    
    print("\n" + "="*70)
    if thickness_count <= 1:
        print("✅ GOOD: Feature diversity achieved!")
        print("   Top 3 features are from different measurements")
    else:
        print(f"⚠️  WARNING: {thickness_count}/3 top features are thickness-related")
        print("   Model may still be over-relying on one aspect")
    
    # Save model and preprocessing objects
    print("\n💾 Saving improved model files...")
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    
    print("\n✅ Training completed successfully!")
    print("="*70)
    
    return rf_model, scaler, imputer, le, feature_names

if __name__ == "__main__":
    train_improved_model()
