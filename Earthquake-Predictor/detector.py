import numpy as np
import pandas as pd
import os
import sys
import pickle
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

print("\n=== Model Training Script ===")
print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Current working directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")

try:
    # Step 1: Load the dataset
    print("\nStep 1: Loading dataset...")
    if not os.path.exists("dataset.csv"):
        raise FileNotFoundError("dataset.csv not found in the current directory.")
    
    data = pd.read_csv("dataset.csv")
    print(f"Dataset loaded with shape: {data.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(data.head())
    
    # Step 2: Prepare features and target
    print("\nStep 2: Preparing features and target...")
    
    # Convert magnitude to integer classes by rounding to nearest integer
    # This creates fewer, more populated classes
    data['MagnitudeClass'] = data['Magnitude'].round().astype(int)
    
    # Print class distribution
    print("\nClass distribution (before filtering):")
    print(data['MagnitudeClass'].value_counts().sort_index())
    
    # Remove classes with too few samples (less than 5% of the data)
    min_samples = max(2, int(0.05 * len(data)))  # At least 2 samples or 5% of data
    class_counts = data['MagnitudeClass'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    
    # Filter the data to keep only valid classes
    filtered_data = data[data['MagnitudeClass'].isin(valid_classes)]
    
    if len(filtered_data) < len(data):
        print(f"\nFiltered out {len(data) - len(filtered_data)} samples with rare classes")
    
    # Prepare features and target
    X = filtered_data[['Latitude', 'Longitude', 'Depth']].values
    y = filtered_data['MagnitudeClass'].values
    
    print(f"\nFinal features shape: {X.shape}, Target shape: {y.shape}")
    print("\nClass distribution (after filtering):")
    print(pd.Series(y).value_counts().sort_index())
    
    # Step 3: Split the data
    print("\nStep 3: Splitting data into train and test sets...")
    
    # Check if we can use stratification
    unique_classes, class_counts = np.unique(y, return_counts=True)
    can_stratify = all(count > 1 for count in class_counts)
    
    if can_stratify and len(unique_classes) > 1:
        print("Using stratified split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        print("Using regular split (stratification not possible)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 4: Train the model
    print("\nStep 4: Training Random Forest Classifier...")
    rfc = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    rfc.fit(X_train, y_train)
    print("\nModel training completed!")
    
    # Step 5: Evaluate the model
    print("\nStep 5: Evaluating the model...")
    y_pred = rfc.predict(X_test)
    
    print("\n=== Model Evaluation ===")
    print("\nAccuracy:", metrics.accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Step 6: Save the model
    print("\nStep 6: Saving the model...")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(rfc, f)
    
    print(f"\nModel saved successfully to: {model_path}")
    print("\n=== Training completed successfully! ===")
    
except Exception as e:
    print(f"\n=== ERROR ===")
    print(f"Type: {type(e).__name__}")
    print(f"Error: {str(e)}")
    print("\nStack trace:")
    import traceback
    traceback.print_exc()
    sys.exit(1)