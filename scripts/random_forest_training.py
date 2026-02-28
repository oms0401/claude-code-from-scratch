import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import argparse
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load and preprocess the CSV data."""
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Check if target column exists
        if 'target' not in df.columns:
            logger.error("Target column 'target' not found in the dataset")
            raise ValueError("Target column 'target' is required in the dataset")
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Handle categorical variables if any
        le = LabelEncoder()
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = le.fit_transform(X[col])
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        logger.info(f"Data loaded successfully. Shape: {X.shape}")
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, cv_folds=5):
    """Train Random Forest classifier with GridSearchCV for hyperparameter tuning."""
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }
    
    # Create the Random Forest classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Create GridSearchCV object
    logger.info("Starting GridSearchCV for hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv_folds,
        n_jobs=-1,  # Use all available cores
        verbose=1,
        scoring='accuracy'
    )
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data."""
    logger.info("Evaluating model on test data...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    # Print classification report
    logger.info("Classification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Print confusion matrix
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    return accuracy

def save_model(model, file_path):
    """Save the trained model to disk."""
    try:
        joblib.dump(model, file_path)
        logger.info(f"Model saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Train Random Forest Classifier with GridSearchCV')
    parser.add_argument('--data', type=str, required=True, help='Path to the CSV data file')
    parser.add_argument('--output', type=str, default='random_forest_model.pkl', 
                       help='Path to save the trained model')
    parser.add_argument('--cv-folds', type=int, default=5, 
                       help='Number of cross-validation folds')
    parser.add_argument('--test-size', type=float, default=0.2, 
                       help='Test set size (default: 0.2)')
    
    args = parser.parse_args()
    
    try:
        # Load and preprocess data
        X, y = load_data(args.data)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size)
        
        # Train model with GridSearchCV
        best_model, grid_search = train_random_forest(X_train, y_train, cv_folds=args.cv_folds)
        
        # Evaluate model
        accuracy = evaluate_model(best_model, X_test, y_test)
        
        # Save model
        save_model(best_model, args.output)
        
        # Save grid search results
        grid_results_file = args.output.replace('.pkl', '_grid_results.pkl')
        save_model(grid_search, grid_results_file)
        
        logger.info(f"Training completed successfully. Final test accuracy: {accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()