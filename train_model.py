"""
Train Random Forest model for Parkinson's Disease Detection
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

def train_model(data_path="spiral_feature_dataset.csv"):
    """Train and save Random Forest model"""
    print("="*60)
    print("PARKINSON'S DISEASE DETECTION - MODEL TRAINING")
    print("="*60)
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Separate features and target
    X = df.drop(columns=['Image', 'Label'])
    y = df['Label']
    
    # Store feature names
    feature_names = X.columns.tolist()
    
    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"\nClasses: {le.classes_}")
    
    # Handle missing values
    print("\nHandling missing values...")
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Initialize and train Random Forest
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_model.predict(X_test)
    
    print("\nModel Performance:")
    print("-"*30)
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
    
    print("\nTop 10 Most Important Features:")
    print(feat_imp.head(10))
    
    # Save model and preprocessing objects
    print("\nSaving model files...")
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    
    print("\n✅ Training completed successfully!")
    return rf_model, scaler, imputer, le, feature_names
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(self.data_path)
        
        # Basic cleaning
        print(f"Original dataset shape: {df.shape}")
        df = df.dropna(subset=['Label'])  # Remove rows with missing labels
        df = df[df['Label'] != '']        # Remove empty labels
        print(f"After cleaning: {df.shape}")
        
        # Separate features and target
        X = df.drop(columns=['Image', 'Label'])
        y = df['Label']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Class distribution:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            count = np.sum(y_encoded == i)
            print(f"  {class_name}: {count} ({count/len(y_encoded)*100:.1f}%)")
        
        # Handle missing values
        print("Handling missing values...")
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        print("Scaling features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        return X_scaled, y_encoded
    
    def initialize_models(self):
        """Initialize different models to compare"""
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            )
        }
    
    def train_and_evaluate_models(self, X, y):
        """Train and evaluate all models"""
        print("\\nTraining and evaluating models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Test set predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # For binary classification
            if len(self.label_encoder.classes_) == 2:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            
            results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'test_auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  Test Accuracy: {accuracy:.4f}")
            print(f"  Test AUC: {auc:.4f}")
        
        return results, X_test, y_test
    
    def select_best_model(self, results):
        """Select the best performing model"""
        print("\\n" + "="*50)
        print("MODEL COMPARISON SUMMARY")
        print("="*50)
        
        best_score = 0
        best_name = None
        
        for name, result in results.items():
            score = result['test_accuracy']  # You can change this to cv_mean or test_auc
            print(f"{name:20} | Accuracy: {result['test_accuracy']:.4f} | AUC: {result['test_auc']:.4f} | CV: {result['cv_mean']:.4f}")
            
            if score > best_score:
                best_score = score
                best_name = name
        
        print(f"\\nBest model: {best_name} (Accuracy: {best_score:.4f})")
        
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        
        return best_name, results[best_name]
    
    def generate_detailed_report(self, best_result, X_test, y_test):
        """Generate detailed classification report for best model"""
        print(f"\\n" + "="*50)
        print(f"DETAILED REPORT FOR {self.best_model_name}")
        print("="*50)
        
        y_pred = best_result['predictions']
        
        # Classification report
        report = classification_report(
            y_test, y_pred, 
            target_names=self.label_encoder.classes_,
            digits=4
        )
        print("\\nClassification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\\nConfusion Matrix:")
        print("Predicted ->", end="")
        for class_name in self.label_encoder.classes_:
            print(f"\\t{class_name}", end="")
        print()
        
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"Actual {class_name}\\t", end="")
            for j in range(len(self.label_encoder.classes_)):
                print(f"\\t{cm[i,j]}", end="")
            print()
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            print("\\nTop 10 Most Important Features:")
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i in range(min(10, len(indices))):
                idx = indices[i]
                print(f"  {i+1:2d}. {self.feature_names[idx]:25s} ({importances[idx]:.4f})")
    
    def save_model(self, model_dir="model"):
        """Save the trained model and preprocessing objects"""
        import os
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the best model
        model_path = os.path.join(model_dir, 'best_model.pkl')
        joblib.dump(self.best_model, model_path)
        print(f"\\nBest model saved to: {model_path}")
        
        # Save preprocessing objects
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
        
        imputer_path = os.path.join(model_dir, 'imputer.pkl')
        joblib.dump(self.imputer, imputer_path)
        print(f"Imputer saved to: {imputer_path}")
        
        label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        joblib.dump(self.label_encoder, label_encoder_path)
        print(f"Label encoder saved to: {label_encoder_path}")
        
        # Save feature names
        feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
        joblib.dump(self.feature_names, feature_names_path)
        print(f"Feature names saved to: {feature_names_path}")
        
        # Save model metadata
        metadata = {
            'best_model_name': self.best_model_name,
            'feature_count': len(self.feature_names),
            'classes': self.label_encoder.classes_.tolist(),
            'n_classes': len(self.label_encoder.classes_)
        }
        
        metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Model metadata saved to: {metadata_path}")
        
        print(f"\\nAll model files saved successfully!")
        return model_path
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("="*60)
        print("PARKINSON'S DISEASE DETECTION - MODEL TRAINING")
        print("="*60)
        
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Initialize models
        self.initialize_models()
        
        # Train and evaluate models
        results, X_test, y_test = self.train_and_evaluate_models(X, y)
        
        # Select best model
        best_name, best_result = self.select_best_model(results)
        
        # Generate detailed report
        self.generate_detailed_report(best_result, X_test, y_test)
        
        # Save model
        model_path = self.save_model()
        
        return model_path, best_name, best_result


if __name__ == "__main__":
    train_model()