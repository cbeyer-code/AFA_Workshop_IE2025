import abc
import collections
import math
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from river import (
    base,
    compose,
    datasets,
    evaluate,
    metrics,
    preprocessing,
    stats,
    stream,
    tree, linear_model, optim, forest,
)
import matplotlib.pyplot as plt

from afa_scores import AEDScorer,FeatureScorer, RandomScorer
from afa_strategies import ActiveFeatureAcquisition
from budget_managers import BudgetManager, IPFBudgetManager, SimpleBudgetManager


# --- 1. DATA PREPARATION ---

def create_missingness(dataset: pd.DataFrame, target_col: str, missingness_rate: float = 0.5, random_state: int = 42):
    """
    Introduces missing values into a dataset and creates two parallel streams.
    This version ensures the first instance is always complete in both streams
    to allow for model burn-in, and introduces missingness in all feature types.

    Args:
        dataset (pd.DataFrame): The complete input DataFrame.
        target_col (str): The name of the target variable column.
        missingness_rate (float): The proportion of feature values to make missing (0.0 to 1.0).
        random_state (int): Seed for reproducibility.

    Returns:
        tuple: A tuple containing two river stream generators:
               - stream_missing: Yields (features_with_missing_values, target).
               - stream_complete: Yields (complete_features, target).
    """
    if dataset.empty:
        return iter([]), iter([])

    rng = np.random.RandomState(random_state)

    # 1. Select a random complete instance for burn-in
    burn_in_idx = dataset.sample(n=1, random_state=rng).index
    burn_in_row = dataset.loc[burn_in_idx]

    # 2. Separate the rest of the data
    remaining_data = dataset

    # 3. Create a copy to introduce missingness into
    features_missing = remaining_data.copy()

    # 4. Introduce missingness in all columns of the remaining data
    for col in features_missing.drop(columns=target_col).columns:
        n_missing = int(len(features_missing) * missingness_rate)
        if n_missing > 0:
            # Choose random indices from the *remaining data* to modify
            missing_indices = rng.choice(features_missing.index, size=n_missing, replace=False)
            features_missing.loc[missing_indices, col] = np.nan

    # 5. Reassemble the DataFrames with the burn-in row at the top
    final_df_complete = pd.concat([burn_in_row, remaining_data]).reset_index(drop=True)
    final_df_missing = pd.concat([burn_in_row, features_missing]).reset_index(drop=True)

    # 6. Create the final streams
    stream_missing = stream.iter_pandas(
        X=final_df_missing.drop(columns=target_col),
        y=final_df_missing[target_col]
    )
    stream_complete = stream.iter_pandas(
        X=final_df_complete.drop(columns=target_col),
        y=final_df_complete[target_col]
    )

    return stream_missing, stream_complete


def one_hot_encode_df(df: pd.DataFrame, categorical_cols: List[str],*, drop_first: bool = False, dtype=np.uint8) -> pd.DataFrame:
    """
    One-hot-encode the specified categorical columns of `df`.

    Parameters
    ----------
    df : pd.DataFrame
        Input data frame (must already include the target column).
    categorical_cols : List[str]
        Names of columns to one-hot-encode.
    drop_first : bool, default False
        If True, drop the first dummy in each set (avoids collinearity).
    dtype : numpy dtype, default np.uint8
        Data type of the dummy columns; uint8 saves memory.

    Returns
    -------
    df_encoded : pd.DataFrame
        New data frame with dummies appended and originals dropped.
    feature_names : List[str]
        All predictor column names **after** encoding (target excluded).
    """

    df_encoded = pd.get_dummies(
        df,
        columns=categorical_cols,
        drop_first=drop_first,
        dtype=dtype,
    )

    return df_encoded

# --- 2. FEATURE COST GENERATION ---

def generate_feature_costs(feature_names: list, strategy: str = 'equal', base_cost: int = 1) -> dict:
    """
    Generates a dictionary of feature costs based on a given strategy.

    Args:
        feature_names (list): A list of the feature names.
        strategy (str): One of 'equal', 'increasing', or 'decreasing'.
        base_cost (int): The base cost unit.

    Returns:
        dict: A dictionary mapping feature names to their costs.
    """
    if strategy == 'equal':
        return {name: base_cost for name in feature_names}
    elif strategy == 'increasing':
        return {name: (i + 1) * base_cost for i, name in enumerate(feature_names)}
    elif strategy == 'decreasing':
        return {name: (len(feature_names) - i) * base_cost for i, name in enumerate(feature_names)}
    else:
        raise ValueError("Strategy must be one of 'equal', 'increasing', or 'decreasing'.")




def plot_results(history: dict, title: str):
    """Plots the results of the simulation."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(title, fontsize=16)

    # Plot 1: Performance (Kappa)
    ax1.plot(history['step'], history['kappa'], label='Kappa Score', color='blue')
    ax1.set_ylabel('Kappa Score')
    ax1.set_title(f'Model Performance - Running Cohen Kappa (final score: '+str(round(history['kappa'][-1],3))+')')
    ax1.grid(True)
    ax1.legend()
    ax1.set_ylim(-1, 1)

    # Plot 2: Budget Usage
    ax2.plot(history['step'], history['budget_received'], label='Total Budget Received', linestyle='--', color='green')
    ax2.plot(history['step'], history['budget_spent'], label='Total Budget Spent', color='red')
    ax2.set_xlabel('Instances')
    ax2.set_ylabel('Budget')
    ax2.set_title('Budget Usage Over Time')
    ax2.grid(True)
    ax2.legend()

    # Plot 3: Feature Merits
    features_to_plot = [
        (name, hist) for name, hist in history['merits'].items()
        if any(m > 0.001 for m in hist)
    ]
    num_features_to_plot = len(features_to_plot)

    if num_features_to_plot > 0:
        for feature_name, merit_history in features_to_plot:
            ax3.plot(history['step'], merit_history, label=feature_name, alpha=0.8)

    ax3.set_xlabel('Instances')
    ax3.set_ylabel('Merit Score (AED/Cost)')
    ax3.set_title('Feature Merits Over Time')
    ax3.grid(True)
    if num_features_to_plot > 10:
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    else:
        ax3.legend(fontsize='small')

    # Get the colors used in the time series plot to reuse them in the violin plot
    lines = ax3.get_lines()
    colors = [line.get_color() for line in lines]

    # Plot 4: Feature Merits (Violin Plot)
    if num_features_to_plot > 0:
        merit_data = [hist for name, hist in features_to_plot]
        merit_labels = [name for name, hist in features_to_plot]

        parts = ax4.violinplot(merit_data, showmeans=False, showmedians=True)
        # Customizing colors for the violin plot to match the time series
        for i, pc in enumerate(parts['bodies']):
            # Use modulo in case the number of colors is less than the number of bodies
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        parts['cmedians'].set_edgecolor('black')

        ax4.set_title('Distribution of Feature Merits')
        ax4.set_ylabel('Merit Score (AED/Cost)')
        ax4.set_xticks(np.arange(1, len(merit_labels) + 1))
        ax4.set_xticklabels(merit_labels, rotation=45, ha='right')
        ax4.grid(axis='y')

    plt.tight_layout()
    plt.show()



# --- 7. SIMULATION AND EVALUATION ---

def run_simulation(
        dataset,
        target_name,
        classifier_model,
        scorer,
        budget_manager,
        cost_strategy='equal',
        acquisition_strategy='k-best',
        k=1,
        missingness_rate=0.5,
        n_samples=5000
):
    """Main function to run the AFA simulation."""

    # 1. Prepare Data and Costs
    # Use the full dataset from the generator
    #data_gen = iter(dataset)
    #first_x, first_y = next(data_gen)
    #feature_names = list(first_x.keys())


    # 1. Prepare Data and Costs
    if isinstance(dataset, pd.DataFrame):
        if n_samples:
            data_sample = dataset.head(n_samples)
        else:
            data_sample = dataset
        feature_names = data_sample.drop(columns=target_name).columns.tolist()
    else:
        # Assumes a river-compatible iterable (like a river.datasets object)
        try:
            data_gen = iter(dataset)
            first_x, first_y = next(data_gen)
            feature_names = list(first_x.keys())

            # For generator-like datasets, n_samples should be provided or available
            if n_samples is None:
                try:
                    # river datasets have n_samples attribute
                    n_samples = dataset.n_samples
                except AttributeError:
                    raise ValueError("n_samples must be provided for datasets that are generators.")

            df_list = [{'y': first_y, **first_x}]
            # We already consumed one item with next(), so we iterate for n_samples - 1 more.
            for i, (x, y) in enumerate(data_gen):
                if i + 1 > n_samples:
                    break
                df_list.append({'y': y, **x})
            data_sample = pd.DataFrame(df_list).rename(columns={'y': target_name})
        except TypeError:
            raise TypeError("Input dataset must be a pandas DataFrame or a river-compatible iterable.")

    # Identify types of features to set up scaler for AED
    # x_sample = data_sample.drop(columns=target_name).iloc[0].to_dict()
    # numerical_features = [k for k, v in x_sample.items() if isinstance(v, (int, float))]
    # categorical_features = [k for k, v in x_sample.items() if not isinstance(v, (int, float))]

    # A robust way to identify feature types from the DataFrame
    numerical_features = data_sample.select_dtypes(include=np.number).columns.tolist()
    # Ensure the target column is not in the feature list
    if TARGET in numerical_features:
        numerical_features.remove(TARGET)

    categorical_features = data_sample.select_dtypes(include=['object', 'category']).columns.tolist()
    if TARGET in categorical_features:
        categorical_features.remove(TARGET)

    # Print the detected feature types
    print(feature_names)
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    if categorical_features:
        data_sample = one_hot_encode_df(
            data_sample,
            categorical_features,
            drop_first=False  # set True if you prefer k-1 dummies
        )
        print('Categorical features were One-Hot-Encoded for the classifiers')

    feature_names = data_sample.drop(columns=[target_name]).columns.tolist()
    categorical_features = [col for col in feature_names if col not in numerical_features]
    print("Categorical features after one-ho:", categorical_features)
    stream_miss, stream_true = create_missingness(data_sample, target_name, missingness_rate)







    # Combine streams to pass (x_miss, x_true) tuples
    parallel_stream = zip(stream_miss, stream_true)

    # Generate costs
    costs = generate_feature_costs(feature_names, strategy=cost_strategy)

    # 2. Build Pipeline
    afa_transformer = ActiveFeatureAcquisition(
        scorer=scorer,
        budget_manager=budget_manager,
        feature_costs=costs,
        acquisition_strategy=acquisition_strategy,
        k=k
    )



    #Preprocessing leaves categorical features as they are and min-max scales the numercial ones
    preprocessing_pipeline = compose.TransformerUnion(
        compose.Select(*numerical_features) | preprocessing.MinMaxScaler(),
        compose.Select(*categorical_features))


    imputer = preprocessing.StatImputer()
    classifier = classifier_model

    # 3. Setup Metrics
    metric_kappa = metrics.CohenKappa()
    metric_accuracy = metrics.Accuracy()

    # History tracking
    history = {
        'step': [],
        'kappa': [],
        'accuracy': [],
        'budget_spent': [],
        'budget_received': [],
        'features_acquired': [],
        'merits': {name: [] for name in feature_names}
    }

    # 4. Run Progressive Validation
    print(
        f"Running simulation with: {acquisition_strategy}(k={k}), {scorer.__class__.__name__}, {budget_manager.__class__.__name__}, cost: {cost_strategy}")

    step = 0
    # Custom loop to handle the parallel stream and metric tracking
    for (x_miss, y_miss), (x_true, y_true) in parallel_stream:
        # --- PREDICTION FLOW ---
        # a. Perform Active Feature Acquisition
        preprocessing_pipeline.learn_one(x_miss) #ToDo: This can lead to values which are out of bounds if x_true exceeds the ranges
        x_miss_norm = preprocessing_pipeline.transform_one(x_miss.copy())
        x_true_norm = preprocessing_pipeline.transform_one(x_true.copy())
        x_acquired = afa_transformer.transform_one((x_miss_norm, x_true_norm))

        # b. Impute remaining missing values
        x_imputed = imputer.transform_one(x_acquired)

        # c. Make a prediction
        y_pred = classifier.predict_one(x_imputed)

        # --- METRIC UPDATE ---
        if y_pred is not None:
            metric_kappa.update(y_true, y_pred)
            metric_accuracy.update(y_true, y_pred)

        # --- LEARNING FLOW ---
        # a. Learn the AFA components (using true data)
        afa_transformer.learn_one(x_acquired, y_true)

        # b. Learn the imputer (using the partially acquired data)
        imputer.learn_one(x_acquired)

        # c. Learn the classifier (using the fully imputed data)
        classifier.learn_one(x_imputed, y_true)

        # Record history
        if step % 10 == 0:
            history['step'].append(step)
            history['kappa'].append(metric_kappa.get())
            history['accuracy'].append(metric_accuracy.get())
            history['budget_spent'].append(budget_manager.get_spent_budget())
            history['budget_received'].append(budget_manager.get_received_budget())
            history['features_acquired'].append(afa_transformer.features_acquired_this_step)
            current_merits = afa_transformer.scorer.get_global_merits()
            for feature_name in history['merits']:
                history['merits'][feature_name].append(current_merits.get(feature_name, 0))
        step += 1
        if step >= n_samples:
            break

    print("Simulation finished.")
    return history



if __name__ == '__main__':
    # --- CONFIGURATION ---
    N_SAMPLES = 20000
    BUDGET_PER_INSTANCE = 1.0
    MISSINGNESS = 0.25
    K_PARAM = 2  # For k-best, etc.

    USE_PANDAS_DF = True  # Set to True to use a local pandas DataFrame

    if USE_PANDAS_DF:
        # Example of loading a pandas DataFrame
        # For this to work, you would need a 'my_dataset.csv' file in the same directory
        try:
            DATASET = pd.read_csv('dataset_generators/mixed_5k_shift')
            TARGET = 'label'  # Change this to your target column
        except FileNotFoundError:
            print("Creating a dummy pandas DataFrame because 'my_dataset.csv' was not found.")
            data = {
                'num_feat_1': np.random.rand(N_SAMPLES) * 10,
                'num_feat_2': np.random.rand(N_SAMPLES) * 5,
                'cat_feat_1': np.random.choice(['A', 'B', 'C'], N_SAMPLES),
                'cat_feat_2': np.random.choice(['X', 'Y', 'Z'], N_SAMPLES),
                'target': np.random.choice([0, 1], N_SAMPLES)
            }
            DATASET = pd.DataFrame(data)
            TARGET = 'target'

    else:
        # Use a river dataset
        DATASET = datasets.Higgs()
        TARGET = 'class'



    # --- CHOOSE YOUR COMPONENTS ---

    # 1. Classifier
    classifier = tree.HoeffdingTreeClassifier(grace_period=100)
    #classifier = tree.ExtremelyFastDecisionTreeClassifier(grace_period=100,delta=1e-5, min_samples_reevaluate=100)
    #classifier = linear_model.LogisticRegression(optimizer=optim.SGD(.1))
    #classifier = forest.AMFClassifier(n_estimators=10, use_aggregation=True, dirichlet=0.5, seed=1)

    # 2. Scorer
    #scorer = RandomScorer(seed=42)
    scorer = AEDScorer(window_size=200)

    # 3. Budget Manager
    #budget_mgr = SimpleBudgetManager(budget_per_instance=BUDGET_PER_INSTANCE)
    budget_mgr = IPFBudgetManager(budget_per_instance=BUDGET_PER_INSTANCE, window_size=200)

    # 4. Cost Strategy
    cost_strat = 'equal'  # 'equal', 'increasing', 'decreasing'

    # 5. Acquisition Strategy
    acq_strat = 'k-best'  # 'k-best', 'k-global-best', 'k-max-mean'

    # --- RUN ---

    history = run_simulation(
        dataset=DATASET,
        target_name=TARGET,
        classifier_model=classifier,
        scorer=scorer,
        budget_manager=budget_mgr,
        cost_strategy=cost_strat,
        acquisition_strategy=acq_strat,
        k=K_PARAM,
        missingness_rate=MISSINGNESS,
        n_samples=N_SAMPLES
    )

    # --- VISUALIZE ---
    plot_title = (
        f"{acq_strat}(k={K_PARAM}), {scorer.__class__.__name__}, "
        f"{budget_mgr.__class__.__name__}\n"
        f"Cost: {cost_strat}, Budget/Inst: {BUDGET_PER_INSTANCE}, Missingness: {MISSINGNESS}"
    )
    plot_results(history, plot_title)
