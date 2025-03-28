import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self):
        # Selected features based on importance and usability
        self.required_features = [
            'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt',
            'Neighborhood', 'BsmtFullBath', 'BsmtHalfBath',
            'TotalBsmtSF', 'TotRmsAbvGrd',
            'MSZoning', 'HouseStyle', 'KitchenQual', 'GarageArea', 'GarageYrBlt',
            '1stFlrSF', 'Fireplaces', 'YearRemodAdd'
        ]

        # Always train a new model to ensure consistency
        self.model = None
        self.train_model()

    def train_model(self):
        """Train and save the prediction model"""
        # Load and prepare data
        train_data = pd.read_csv('C:/Master/classes/Machine learning/Predict_House/Predict_House/data/train.csv')
        train_data = self.preprocess_data(train_data)
        train_data = self.create_features(train_data)

        # Use only our selected features
        X = train_data[self.required_features]
        y = train_data['SalePrice']

        # Split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Build pipeline
        numeric_features = list(set(X.columns) - {'Neighborhood', 'MSZoning', 'HouseStyle', 'KitchenQual'})
        categorical_features = list(set(X.columns) & {'Neighborhood', 'MSZoning', 'HouseStyle', 'KitchenQual'})

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=42
            ))
        ])

        # Train and evaluate
        self.model.fit(X_train, y_train)
        self.evaluate_model(X_test, y_test)
        self.plot_feature_importance()

        # Save model
        joblib.dump(self.model, 'house_price_model.pkl')
        print("\nModel trained and saved successfully")

    def prepare_training_data(self):
        """Prepare X and y for training"""
        train_data = pd.read_csv('C:/Master/classes/Machine learning/Predict_House/Predict_House/data/train.csv')
        train_data = self.preprocess_data(train_data)
        train_data = self.create_features(train_data)
        X = train_data[self.required_features]
        y = train_data['SalePrice']
        return X, y

    def create_preprocessor(self, X):
        """Create a preprocessor for the given features"""
        numeric_features = list(set(X.columns) - {'Neighborhood', 'MSZoning', 'HouseStyle', 'KitchenQual'})
        categorical_features = list(set(X.columns) & {'Neighborhood', 'MSZoning', 'HouseStyle', 'KitchenQual'})

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

    def create_features(self, df):
        """Create useful derived features"""
        df['TotalBath'] = df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        return df

    def preprocess_data(self, df):
        """Handle missing values and basic cleaning"""
        df = df.copy()

        # Handle missing values
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            df[col].fillna(df[col].median(), inplace=True)

        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col].fillna('None', inplace=True)

        return df

    def evaluate_model(self, X_test, y_test):
        """Calculate and display model performance metrics"""
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        print(f"\n{' MODEL EVALUATION ':=^40}")
        print(f"Mean Absolute Error: ${mae:,.2f}")
        print(f"Sample Actual Value: ${y_test.iloc[0]:,.2f}")
        print(f"Sample Predicted Value: ${preds[0]:,.2f}")
        print(f"Error: ${abs(y_test.iloc[0] - preds[0]):,.2f}")

    def plot_feature_importance(self):
        """Visualize the most important features"""
        # Get feature names after preprocessing
        numeric_features = list(set(self.required_features) - {'Neighborhood', 'MSZoning', 'HouseStyle', 'KitchenQual'})
        categorical_features = list(
            set(self.required_features) & {'Neighborhood', 'MSZoning', 'HouseStyle', 'KitchenQual'})

        # Get one-hot encoded feature names
        ohe = self.model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        categorical_names = ohe.get_feature_names_out(categorical_features)

        all_features = numeric_features + list(categorical_names)
        importances = self.model.named_steps['regressor'].feature_importances_

        # Sort and get top 10
        sorted_idx = np.argsort(importances)[-10:]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), importances[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [all_features[i] for i in sorted_idx])
        plt.title("Top 10 Important Features")
        plt.xlabel("Feature Importance Score")
        plt.tight_layout()
        plt.show()

    def predict_price(self, input_data):
        """Make price prediction for input features"""
        try:
            input_df = pd.DataFrame([input_data], columns=self.required_features)

            # Convert numeric fields
            numeric_cols = list(set(self.required_features) - {'Neighborhood', 'MSZoning', 'HouseStyle', 'KitchenQual'})
            input_df[numeric_cols] = input_df[numeric_cols].apply(pd.to_numeric)

            prediction = self.model.predict(input_df)
            return prediction[0]
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None

    def create_submission(self, test_file_path='C:/Master/classes/Machine learning/Predict_House/Predict_House/data/test.csv', output_file='submission.csv'):
        """Generate submission file for competition"""
        test_data = pd.read_csv(test_file_path)
        test_data = self.preprocess_data(test_data)
        test_data = self.create_features(test_data)

        # Ensure we have all required columns
        for col in self.required_features:
            if col not in test_data.columns:
                test_data[col] = 0  # Default value for missing columns

        predictions = self.model.predict(test_data[self.required_features])

        submission = pd.DataFrame({
            'Id': test_data['Id'],
            'SalePrice': predictions
        })
        submission.to_csv(output_file, index=False)
        print(f"\nSubmission file '{output_file}' created successfully!")
        return submission

    def compare_models(self):
        """Compare multiple models to demonstrate fundamentals"""
        X, y = self.prepare_training_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }

        print("\n=== MODEL COMPARISON ===")
        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', self.create_preprocessor(X_train)),
                ('regressor', model)
            ])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            print(f"{name}: MAE = ${mae:,.2f}")

    def demonstrate_feature_engineering(self):
        """Show impact of feature engineering"""
        train_data = pd.read_csv('C:/Master/classes/Machine learning/Predict_House/Predict_House/data/train.csv')

        # Before feature engineering (using only original features)
        X_original = train_data[self.required_features]
        y = train_data['SalePrice']

        # After feature engineering (using original + engineered features)
        engineered_data = self.create_features(train_data)
        # Get all features - original required ones plus any new ones created
        engineered_features = self.required_features.copy()
        # Add any new features created in create_features() that aren't already included
        new_features = ['TotalBath', 'HouseAge', 'TotalSF']  # These come from create_features()
        for feat in new_features:
            if feat not in engineered_features and feat in engineered_data.columns:
                engineered_features.append(feat)

        X_engineered = engineered_data[engineered_features]

        print("\n=== FEATURE ENGINEERING IMPACT ===")
        for X_data, desc in [(X_original, "Original Features"), (X_engineered, "Engineered Features")]:
            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y, test_size=0.2, random_state=42
            )

            model = Pipeline(steps=[
                ('preprocessor', self.create_preprocessor(X_train)),
                ('regressor', GradientBoostingRegressor(random_state=42))
            ])

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            print(f"{desc}: MAE = ${mae:,.2f} (using {len(X_data.columns)} features)")

    def demonstrate_hyperparameter_tuning(self):
        """Show basic hyperparameter tuning"""
        X, y = self.prepare_training_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a fresh pipeline for tuning
        pipeline = Pipeline(steps=[
            ('preprocessor', self.create_preprocessor(X_train)),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ])

        param_grid = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 4, 5]
        }

        print("\n=== HYPERPARAMETER TUNING ===")
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error')
        grid_search.fit(X_train, y_train)

        print("Best parameters:", grid_search.best_params_)
        print("Best MAE:", -grid_search.best_score_)

def run_demo_mode(predictor):
    demo_cases = {
        "Basic affordable home": {
            "features": [5, 896, 1, 1961, 'NAmes', 0, 0, 882, 5, 'RH', '1Story', 'TA', 730, 1961, 896, 0, 1961],
            "expected_range": (120000, 140000),
            "key_points": [
                "Quality: 5/10",
                "Living Area: 896 sqft",
                "Bedrooms: 5",
                "Baths: 0",
                "Neighborhood: NAmes",
                "Year Built: 1961"
            ]
        },
        "Average family home": {
            "features": [7, 1500, 2, 2005, 'NAmes', 1, 0, 1000, 8, 'RL', '2Story', 'Gd', 480, 2005, 1500, 1, 2005],
            "expected_range": (180000, 220000),
            "key_points": [
                "Quality: 7/10",
                "Living Area: 1500 sqft",
                "Bedrooms: 8",
                "Baths: 1",
                "Neighborhood: NAmes",
                "Year Built: 2005"
            ]
        }
    }

    print(f"\n{' DEMONSTRATION MODE ':=^40}")
    for desc, case in demo_cases.items():
        pred = predictor.predict_price(case["features"])

        print(f"\n{desc.upper()}")
        print("Key Features:")
        print("\n".join(f"- {point}" for point in case["key_points"]))

        print(f"\nPredicted Price: ${pred:,.2f}")
        print(f"Expected Range: ${case['expected_range'][0]:,.2f}-${case['expected_range'][1]:,.2f}")

        if case['expected_range'][0] <= pred <= case['expected_range'][1]:
            print("✅ Prediction within expected range")
        else:
            deviation = pred - np.mean(case['expected_range'])
            print(f"⚠️ Deviation: ${deviation:,.2f} from expected")

def main():
    # Initialize predictor
    predictor = HousePricePredictor()

    # Main menu
    while True:
        print("\n=== MACHINE LEARNING FUNDAMENTALS DEMO ===")
        print("1. Model Comparison (Linear vs Tree-based)")
        print("2. Feature Engineering Demonstration")
        print("3. Hyperparameter Tuning Example")
        print("4. Run Demonstration Cases")
        print("5. Make Custom Prediction")
        print("6. Create Submission File")
        print("7. Exit")

        choice = input("Select demonstration (1-7): ")

        if choice == '1':
            predictor.compare_models()
        elif choice == '2':
            predictor.demonstrate_feature_engineering()
        elif choice == '3':
            predictor.demonstrate_hyperparameter_tuning()
        elif choice == '4':
            run_demo_mode(predictor)
        elif choice == '5':
            print(f"\n{' CUSTOM PREDICTION ':=^40}")
            print("Enter property details in this exact order:")
            print("[OverallQual, GrLivArea, GarageCars, YearBuilt, Neighborhood,")
            print("BsmtFullBath, BsmtHalfBath, TotalBsmtSF, TotRmsAbvGrd,")
            print("MSZoning, HouseStyle, KitchenQual, GarageArea, GarageYrBlt,")
            print("1stFlrSF, Fireplaces, YearRemodAdd]")
            print("\nExample: 7,1500,2,2005,NAmes,1,0,1000,8,RL,2Story,Gd,480,2005,1500,1,2005")

            user_input = input("\nEnter comma-separated values (or 'back'): ")
            if user_input.lower() == 'back':
                continue

            features = [x.strip() for x in user_input.split(',')]
            if len(features) != len(predictor.required_features):
                print(f"Error: Need exactly {len(predictor.required_features)} values")
                continue

            try:
                # Convert numeric features
                numeric_indices = [0, 1, 2, 3, 5, 6, 7, 8, 12, 13, 14, 15, 16]  # indices of numeric features
                for i in numeric_indices:
                    features[i] = float(features[i])

                prediction = predictor.predict_price(features)
                if prediction is not None:
                    print(f"\n{' RESULT ':=^40}")
                    print(f"Predicted Sale Price: ${prediction:,.2f}")
            except Exception as e:
                print(f"Error processing input: {str(e)}")
        elif choice == '6':
            print(f"\n{' CREATING SUBMISSION ':=^40}")
            predictor.create_submission()
        elif choice == '7':
            print("Exiting program...")
            break
        else:
            print("Invalid choice, please try again")

if __name__ == "__main__":
    main()