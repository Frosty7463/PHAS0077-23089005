import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class Element_prediction:
    """
    A class used to represent a model for predicting polarizability based on atomic radius
    for different groups and periods in the periodic table.

    Attributes:
    data : dict
        A dictionary containing data for various groups and periods in the periodic table.
        Each entry contains atomic radius and polarizability values.

    Methods:
    _prepare data():
        Prepares the data by converting it into pandas DataFrames and separating features and targets.
    cross_validation():
        Splits the data into training and testing sets for each group and period.
    train_models():
        Trains linear regression models for each group and period in the data.
    evaluate_models():
        Evaluates the trained models and calculates the RMSE for each.
    predict_polarizability(group, atomic_radius):
        Predicts the polarizability for a given group and atomic radius.
    print_evaluation():
        Prints the RMSE for each group and period model.
    print_predictions():
        Prints the predicted polarizability for predefined atomic radii.
    """
    def __init__(self, data):
        """
        Initializes the PeriodicTableModel with data and prepares the data for training.

        Parameters:
        data : dict
            A dictionary containing data for various groups and periods in the periodic table.
        """
        self.data = data
        self.models = {}
        self.rmse = {}
        self._prepare_data()

    def _prepare_data(self):
        """
        Prepares the data by converting it into pandas DataFrames and separating features and targets.
        """
        self.df_group1 = pd.DataFrame(self.data['group1'])
        self.df_group2 = pd.DataFrame(self.data['group2'])
        self.df_group17 = pd.DataFrame(self.data['group17'])
        self.df_noble_gas = pd.DataFrame(self.data['noble_gas'])
        self.df_period4 = pd.DataFrame(self.data['period4'])

        self.X_group1 = self.df_group1[['Atomic_Radius']]
        self.y_group1 = self.df_group1['Polarizability']

        self.X_group2 = self.df_group2[['Atomic_Radius']]
        self.y_group2 = self.df_group2['Polarizability']

        self.X_group17 = self.df_group17[['Atomic_Radius']]
        self.y_group17 = self.df_group17['Polarizability']

        self.X_noble = self.df_noble_gas[['Atomic_Radius']]
        self.y_noble = self.df_noble_gas['Polarizability']

        self.X_period4 = self.df_period4[['Atomic_Radius']]
        self.y_period4 = self.df_period4['Polarizability']

    def cross_validation(self):
        """
        Splits the data into training and testing sets for each group and period.
        """
        self.X_train_group1, self.X_test_group1, self.y_train_group1, self.y_test_group1 = train_test_split(
            self.X_group1, self.y_group1, test_size=0.2, random_state=42)
        
        self.X_train_group2, self.X_test_group2, self.y_train_group2, self.y_test_group2 = train_test_split(
            self.X_group2, self.y_group2, test_size=0.2, random_state=42)
        
        self.X_train_group17, self.X_test_group17, self.y_train_group17, self.y_test_group17 = train_test_split(
            self.X_group17, self.y_group17, test_size=0.2, random_state=42)
        
        self.X_train_noble, self.X_test_noble, self.y_train_noble, self.y_test_noble = train_test_split(
            self.X_noble, self.y_noble, test_size=0.2, random_state=42)
        
        self.X_train_period4, self.X_test_period4, self.y_train_period4, self.y_test_period4 = train_test_split(
            self.X_period4, self.y_period4, test_size=0.2, random_state=42)

    def train_models(self):
        """
        Trains linear regression models for each group and period in the data.
        """
        self.cross_validation()

        self.models['group1'] = LinearRegression()
        self.models['group2'] = LinearRegression()
        self.models['group17'] = LinearRegression()
        self.models['noble'] = LinearRegression()
        self.models['period4'] = LinearRegression()

        self.models['group1'].fit(self.X_train_group1, self.y_train_group1)
        self.models['group2'].fit(self.X_train_group2, self.y_train_group2)
        self.models['group17'].fit(self.X_train_group17, self.y_train_group17)
        self.models['noble'].fit(self.X_train_noble, self.y_train_noble)
        self.models['period4'].fit(self.X_train_period4, self.y_train_period4)

    def evaluate_models(self):
        """
        Evaluates the trained models and calculates the RMSE for each.
        """
        self.rmse['group1'] = np.sqrt(mean_squared_error(
            self.y_test_group1, self.models['group1'].predict(self.X_test_group1)))
        self.rmse['group2'] = np.sqrt(mean_squared_error(
            self.y_test_group2, self.models['group2'].predict(self.X_test_group2)))
        self.rmse['group17'] = np.sqrt(mean_squared_error(
            self.y_test_group17, self.models['group17'].predict(self.X_test_group17)))
        self.rmse['noble'] = np.sqrt(mean_squared_error(
            self.y_test_noble, self.models['noble'].predict(self.X_test_noble)))
        self.rmse['period4'] = np.sqrt(mean_squared_error(
            self.y_test_period4, self.models['period4'].predict(self.X_test_period4)))

    def predict_polarizability(self, group, atomic_radius):
        """
        Predicts the polarizability for a given set of data and atomic radius.

        Parameters:
        group : str
            The group or period for which the prediction is to be made (e.g., 'group1', 'group2', etc.).
        atomic_radius : float
            The atomic radius for which the polarizability is to be predicted.

        Returns:
        float
            The predicted polarizability.
        """
        input_data = pd.DataFrame([[atomic_radius]], columns=['Atomic_Radius'])
        return self.models[group].predict(input_data)[0]

    def print_evaluation(self):
        """
        Prints the RMSE for the prediction of each set of data.
        """
        for group, error in self.rmse.items():
            print(f"Root Mean Squared Error for {group} in periodic table: {error:.3f}")

    def print_predictions(self):
        """
        Prints the predicted polarizability for Van der Waals atomic radii for groups and periods.
        """

        # The Van der Waals atomic radius (in pm) for elements to predict 
        elements_to_predict = {
            'group1': {'Rb': 303, 'Cs': 343, 'Fr': 348}, 
            'group2': {'Sr': 249, 'Ba': 268, 'Ra': 283}, 
            'group17': {'I':198,'At': 202}, 
            'noble': {'Xe': 216, 'Rn': 220}, 
            'period4': {'Cr': 189, 'Mn': 197, 'Fe': 194, 'Co': 192, 'Ni': 163, 'Cu':140,
                        'Zn':139,'Ga': 187, 'Ge': 211, 'As': 185, 'Se': 190, 'Br': 183, 'Kr':202}  
        }  
        for group, elements in elements_to_predict.items():
            print(f"\nPredictions for {group}:")
            for element, radius in elements.items():
                polarizability = self.predict_polarizability(group, radius)
                print(f"Predicted polarizability for {element} (atomic radius {radius}): {polarizability:.3f} Å³")

# Data input
data = {
    'group1': {
        'Atomic_Radius': [120, 182, 227, 275], # Van der Waals atomic radius for H, Li, Na, K (in pm)
        'Polarizability': [4.51, 164, 163, 290]
    },
    'group2': {
        'Atomic_Radius': [153, 173, 231], # Van der Waals atomic radius for Be, Mg, Ca (in pm)
        'Polarizability': [37.7, 71.2, 161]
    },
    'group17': {
        'Atomic_Radius': [135, 175, 183], # Van der Waals atomic radius for F, Cl, Br (in pm)
        'Polarizability': [3.74, 13.6, 20]
    },
    'noble_gas': {
        'Atomic_Radius': [140, 154, 188, 202], # Van der Waals atomic radius for He, Ne, Ar, Kr (in pm)
        'Polarizability': [1.38, 2.66, 11.1, 16.8]
    },
    'period4': {
        'Atomic_Radius': [275, 231, 211, 187, 179], # Van der Waals atomic radius for K, Ca, Sc, Ti, V (in pm)
        'Polarizability': [290, 161, 97, 100, 87]
    }

}

if __name__ == "__main__":
   model = Element_prediction(data)
   model.train_models()
   model.evaluate_models()
   model.print_evaluation()
   model.print_predictions()
