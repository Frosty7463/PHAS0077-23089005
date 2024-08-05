import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from parsing_smiles import MoleculeParser
import matplotlib.pyplot as plt


class ModelTrainingAndPrediction:
    """
    A class to predict the polarizability of molecules using various regression models.
    """

    def __init__(self, data_path):
        """
        Initializes the class and sets up the data and models.

        Parameters:
        data_path : str
            The path to the CSV file containing the data.
        """
        self.pol_data = pd.read_csv(data_path)
        self.models = {
            'Lasso Regression': Lasso(alpha=1),
            'Random Forest': RandomForestRegressor(),
            'SVR': SVR(),
            'Adaboost Model': AdaBoostRegressor(random_state=42),
            'Gradient Boosting Model': GradientBoostingRegressor(max_depth=5, n_estimators=100, learning_rate=0.5, random_state=42),
            'k-Nearest Neighbours': KNeighborsRegressor(),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror',
                                        colsample_bytree=0.8,
                                        learning_rate=0.1,
                                        max_depth=5,
                                        n_estimators=100,
                                        random_state=42),
            'Ensemble Model': VotingRegressor([
                ('RF', RandomForestRegressor()),
                ('gboost', GradientBoostingRegressor(max_depth=5, n_estimators=100, learning_rate=0.5, random_state=42)),
                ('xgboost', xgb.XGBRegressor(objective='reg:squarederror',
                                             colsample_bytree=0.8,
                                             learning_rate=0.1,
                                             max_depth=5,
                                             n_estimators=100,
                                             random_state=42))
            ])
        }
        
        self.scaler = StandardScaler()
        self.atoms_vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False, token_pattern=None)
        self.symbols_vectorizer = CountVectorizer(tokenizer=lambda x: x, lowercase=False, token_pattern=None)
        self.results = {}
        self.r2_scores = {}
        self.mae_scores = {}

    def parse_species(self, species):
        """
        Parses the species using MoleculeParser.

        Parameters:
        species : str
            The species to parse.

        Returns:
        tuple
            Parsed details of the species.
        """
        parser = MoleculeParser(species)
        return parser.get_species_details()

    def preprocess_data(self):
        """
        Preprocesses the data by parsing species and vectorizing features.
        """
        # Convert species column to string type
        self.pol_data['species'] = self.pol_data['species'].astype(str)

        # Extract features from the parsed species
        self.pol_data['parsed'] = self.pol_data['species'].apply(self.parse_species)
        self.pol_data['atoms'] = self.pol_data['parsed'].apply(lambda x: x[0])
        self.pol_data['symbols'] = self.pol_data['parsed'].apply(lambda x: x[1])
        self.pol_data['branches_bond'] = self.pol_data['parsed'].apply(lambda x: x[2])
        self.pol_data['cyclic_bond'] = self.pol_data['parsed'].apply(lambda x: x[3])
        self.pol_data['diff_dir_branch'] = self.pol_data['parsed'].apply(lambda x: x[4])

        pol_features = self.pol_data[['atoms', 'symbols', 'branches_bond', 'cyclic_bond', 'diff_dir_branch']]
        pol_target = self.pol_data['alpha']

        # Vectorize features
        X_pol = self.vectorize_molecule_features(pol_features)

        # Cross validation
        X_pol_train, X_pol_test, y_pol_train, y_pol_test, train_index, test_index = train_test_split(
            X_pol, pol_target, self.pol_data.index, test_size=0.3, random_state=42)

        # Standardize features
        self.X_pol_train_scaled = self.scaler.fit_transform(X_pol_train)
        self.X_pol_test_scaled = self.scaler.transform(X_pol_test)
        self.y_pol_train = y_pol_train
        self.y_pol_test = y_pol_test
        self.test_index = test_index

    def vectorize_molecule_features(self, molecule_features):
        """
        Vectorizes the molecule features into a matrix

        Parameters:
        molecule_features : DataFrame
            The molecule features to vectorize.

        Returns:
        ndarray
            The combined feature matrix.
        """
        atoms_vectorized = self.atoms_vectorizer.fit_transform(molecule_features['atoms'].apply(lambda x: ' '.join(x)))
        symbols_vectorized = self.symbols_vectorizer.fit_transform(molecule_features['symbols'].apply(lambda x: ' '.join(x)))
        
        # Combine all features into a single matrix
        features_combined = np.hstack([
            atoms_vectorized.toarray(),
            symbols_vectorized.toarray(),
            molecule_features[['branches_bond', 'cyclic_bond', 'diff_dir_branch']]
        ])
        return features_combined

    def train_and_evaluate_models(self):
        """
        Trains and evaluates the models, storing RMSE, R-squared, and MAE results.
        """
        for name, model in self.models.items():
            model.fit(self.X_pol_train_scaled, self.y_pol_train)
            y_pol_pred = model.predict(self.X_pol_test_scaled)
            mse_pol = mean_squared_error(self.y_pol_test, y_pol_pred)
            rmse = np.sqrt(mse_pol)
            r2 = r2_score(self.y_pol_test, y_pol_pred)
            mae = mean_absolute_error(self.y_pol_test, y_pol_pred)
            self.results[name] = rmse
            self.r2_scores[name] = r2
            self.mae_scores[name] = mae

    def plot_the_performance_of_models(self):
        """
        Displays the RMSE results and plots the error percentage compared to the mean of the training data.
        Additionally, plots the R-squared values of each model.
        """
        average_polarizability = 7.377
        model_names = list(self.results.keys())
        rmse_values = list(self.results.values())
        r2_values = list(self.r2_scores.values())
        mae_values = list(self.mae_scores.values())
        error_percentages = [(rmse / average_polarizability) * 100 for rmse in rmse_values]

        for name in self.results.keys():
            rmse = self.results[name]
            r2 = self.r2_scores[name]
            mae = self.mae_scores[name]
            err_por = rmse / average_polarizability
            err_per = "%.3f%%" % (err_por * 100)
            print(f"{name}: RMSE = {rmse:.3f}, MAE = {mae:.3f}, R² = {r2:.3f},\nerror percentage = {err_per}\n")

        # Plot Error Percentages
        plt.figure(figsize=(10, 6))
        plt.barh(model_names, error_percentages, color='skyblue')
        plt.xlabel('Error Percentage')
        plt.title('Error Percentage of Each Model')
        plt.show()

        # Plot MAE values
        plt.figure(figsize=(10, 6))
        plt.barh(model_names, mae_values, color='lightcoral')
        plt.xlabel('MAE')
        plt.title('Mean Absolute Error of Each Model')
        plt.show()

        # Plot R-squared values
        plt.figure(figsize=(10, 6))
        plt.plot(model_names, r2_values, marker='o', linestyle='-', color='b')
        plt.ylabel('R-squared')
        plt.title('R-squared of Each Model')
        plt.xticks(rotation=45)
        plt.show()

    def predict_polarizability_for_species(self, species):
        """
        Predicts the polarizability for a given species.

        Parameters:
        species : str
            The species to predict polarizability for.

        Returns:
        dict
            Predictions from each model.
        """
        species = str(species)
        parsed_details = self.parse_species(species)
        atoms, symbols, branches_bond, cyclic_bond, diff_dir_branch = parsed_details
        
        # Vectorize features for the given species
        atoms_vectorized = self.atoms_vectorizer.transform([' '.join(atoms)])
        symbols_vectorized = self.symbols_vectorizer.transform([' '.join(symbols)])
        features_combined = np.hstack([
            atoms_vectorized.toarray(),
            symbols_vectorized.toarray(),
            [[branches_bond, cyclic_bond, diff_dir_branch]]
        ])
     
        scaled_features = self.scaler.transform(features_combined)
        
        # Predict polarizability using each model
        predictions = {}
        for name, model in self.models.items():
            predicted_pol = model.predict(scaled_features)
            predictions[name] = predicted_pol[0]
        
        return predictions

    def predict_and_save_test_set(self):
        """
        Predicts polarizability for the test set and saves the results to testing.csv.
        """
        predictions = []

        for idx in self.test_index:
            species = self.pol_data.loc[idx, 'species']
            original_alpha = self.pol_data.loc[idx, 'alpha']
            pred = self.predict_polarizability_for_species(species)
            predicted_alpha = round(pred['Ensemble Model'], 3)
            predictions.append([species, original_alpha, predicted_alpha])

        predictions_df = pd.DataFrame(predictions, columns=['species', 'original', 'predicted'])
        predictions_df.to_csv('testing.csv', index=False)

        # Calculate and print RMSE for the test set
        test_mse = mean_squared_error(predictions_df['original'], predictions_df['predicted'])
        test_rmse = np.sqrt(test_mse)
        print(f"Test Set RMSE: {test_rmse:.3f}")
        print()


if __name__ == "__main__":
    predictor = ModelTrainingAndPrediction('pol_training_data.csv')
    predictor.preprocess_data()
    predictor.train_and_evaluate_models()
    predictor.predict_and_save_test_set()
    predictor.plot_the_performance_of_models()

    # Use the models to predict the polarizability of a certain species
    species_to_predict = ''
    predictions = predictor.predict_polarizability_for_species(species_to_predict)

    #print(f"Predictions for {species_to_predict}:")
    for model_name, prediction in predictions.items():
        pass
        #print(f"{model_name}: {prediction:.3f}")

