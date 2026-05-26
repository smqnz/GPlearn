# -*- coding: utf-8 -*-
import numpy as np
from gplearn.genetic import SymbolicRegressor
from sklearn.impute import SimpleImputer

# ==================== Error metric calculation (optional) ====================
def calculate_metrics(actual: np.ndarray, predicted: np.ndarray):
    """Calculate RMSE, MAE, MSE, R2, MIE, MAPE"""
    actual = actual[~np.isnan(actual) & ~np.isnan(predicted)]
    predicted = predicted[~np.isnan(actual) & ~np.isnan(predicted)]
    if len(actual) == 0:
        return {k: np.nan for k in ['RMSE', 'MAE', 'MSE', 'R2', 'MIE', 'MAPE']}
    n = len(actual)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    mse = np.mean((actual - predicted) ** 2)
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    mie = np.max(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / np.clip(actual, 1e-10, None))) * 100
    return {'RMSE': rmse, 'MAE': mae, 'MSE': mse, 'R2': r2, 'MIE': mie, 'MAPE': mape}

# ==================== Core GP model ====================
def build_gp_model():
    """Construct a genetic programming regressor (parameters based on Section 3.3)"""
    return SymbolicRegressor(
        population_size=3000,          # Population size
        generations=50,               # Number of generations
        tournament_size=50,           # Tournament selection size
        stopping_criteria=0.001,      # Stopping criterion
        const_range=(-10., 10.),      # Range of constants
        init_depth=(2, 10),           # Initial tree depth range
        function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt'),  # Function set
        metric='mean absolute error', # Fitness metric
        parsimony_coefficient=0.002,  # Parsimony coefficient
        p_crossover=0.8,              # Crossover probability
        p_subtree_mutation=0.05,      # Subtree mutation probability
        p_hoist_mutation=0.05,        # Hoist mutation probability
        p_point_mutation=0.05,        # Point mutation probability
        p_point_replace=0.1,          # Point replacement probability
        max_samples=0.9,              # Fraction of samples per generation
        verbose=1,                    # Verbosity
        random_state=42,              # Random seed
        init_method='half and half',  # Initialization method
        warm_start=True,              # Warm start
        n_jobs=-1                     # Use all CPU cores
    )

def train_gp_model(X_train, y_train, X_valid=None, y_valid=None, special=False):
    """
    Train a GP model and return predictions, expression, and metrics.

    Parameters:
    X_train, y_train : Training feature matrix and target vector
    X_valid, y_valid : Validation set (optional, for monitoring)
    special          : Whether to use special configuration (e.g., for head column)

    Returns:
    model            : Trained SymbolicRegressor instance
    best_expr        : String representation of the best program
    y_train_pred     : Predictions on training set
    y_valid_pred     : Predictions on validation set (None if X_valid not provided)
    train_metrics    : Metrics dictionary for training set
    valid_metrics    : Metrics dictionary for validation set (None if y_valid not provided)
    """
    # Build model with standard or special configuration
    if special:
        model = SymbolicRegressor(
            population_size=5000, generations=80,
            function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'log', 'sqrt', 'exp', 'tan'),
            const_range=(-20., 20.), random_state=42,
            n_jobs=-1, **{k: v for k, v in build_gp_model().get_params().items()
                          if k not in ['population_size','generations','function_set','const_range','random_state','n_jobs']}
        )
    else:
        model = build_gp_model()

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    if X_valid is not None:
        X_valid_imp = imputer.transform(X_valid)

    # Train the model
    model.fit(X_train_imp, y_train)

    # Extract best expression
    best_expr = str(model._program)

    # Predictions
    y_train_pred = model.predict(X_train_imp)
    y_valid_pred = model.predict(X_valid_imp) if X_valid is not None else None

    # Compute metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    valid_metrics = calculate_metrics(y_valid, y_valid_pred) if X_valid is not None else None

    return model, best_expr, y_train_pred, y_valid_pred, train_metrics, valid_metrics

# ==================== Example usage (not runnable, only interface demonstration) ====================
if __name__ == "__main__":
    print("GP model core code loaded. Please call train_gp_model() with your actual data.")