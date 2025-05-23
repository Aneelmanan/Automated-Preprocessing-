import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from fancyimpute import IterativeImputer

# Load your dataset
print("Step 1: Loading the dataset...")
df = pd.read_csv("D:/Education/Datacollection/RCADATA_filled_knn.csv")
print("Dataset loaded successfully. The dataset contains", df.shape[0], "rows and", df.shape[1], "columns.")

# List of columns in your dataset
columns = ["M.1", "M.2", "M.3", "M.4", "M.5", "M.6", "M.7", "M.8", "M.9", "M.10",
           "A.1", "A.2", "A.3", "A.4", "A.5", "A.6", "C.1", "C.2", "C.3", "C.4", "C.5"]
print("Columns defined: ", columns)


# Step 2: Handle Missing Data
def handle_missing_data(df, method='mean', k_neighbors=5, max_iter=10):
    """
    Handles missing data in the dataset using different imputation methods:
    - 'mean': Fills missing values with the mean of the column
    - 'median': Fills missing values with the median of the column
    - 'knn': Uses K-Nearest Neighbors to impute missing values
    - 'mice': Multiple Imputation by Chained Equations for missing values
    """
    print(f"Running Step 2: Handling missing data using the {method} method...")
    if method == 'mean':
        print("Using mean imputation method to fill missing values.")
        result = df.fillna(df.mean())
    elif method == 'median':
        print("Using median imputation method to fill missing values.")
        result = df.fillna(df.median())
    elif method == 'knn':
        print(f"Using KNN imputation method with {k_neighbors} neighbors to fill missing values.")
        knn_imputer = KNNImputer(n_neighbors=k_neighbors)
        result = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)
    elif method == 'mice':
        print(f"Using MICE (Multiple Imputation by Chained Equations) with {max_iter} iterations.")
        mice_imputer = IterativeImputer(max_iter=max_iter)
        result = pd.DataFrame(mice_imputer.fit_transform(df), columns=df.columns)
    else:
        print("Invalid method chosen, defaulting to mean imputation.")
        result = df.fillna(df.mean())

    print("Step 2 completed: Missing data handled successfully.")
    return result


# Step 3: Data Normalization
def normalize_data(df, method='minmax'):
    """
    Normalizes the dataset using different methods:
    - 'minmax': Min-Max Scaling to scale the data between 0 and 1
    - 'zscore': Z-Score normalization (Standardization)
    - 'robust': Robust scaling based on median and interquartile range
    """
    print(f"Running Step 3: Normalizing data using {method} method...")
    if method == 'minmax':
        print("Performing Min-Max scaling to normalize data between 0 and 1.")
        scaler = MinMaxScaler()
        result = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    elif method == 'zscore':
        print("Performing Z-Score normalization (standardization).")
        scaler = StandardScaler()
        result = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    elif method == 'robust':
        print("Performing Robust Scaling using median and IQR for normalization.")
        scaler = RobustScaler()
        result = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    else:
        print("Invalid method chosen, defaulting to Min-Max scaling.")
        scaler = MinMaxScaler()
        result = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    print(f"Step 3 completed: Data normalized using {method} method.")
    return result


# Step 4: Outlier Removal
def remove_outliers(df, method='iqr'):
    """
    Removes outliers using different methods:
    - 'boxplot': Visualizes outliers using a boxplot
    - 'iqr': Uses Interquartile Range to remove data points outside the 1.5*IQR range
    - 'zscore': Removes data points whose Z-Score is greater than 3
    """
    print(f"Running Step 4: Removing outliers using the {method} method...")
    if method == 'boxplot':
        print("Displaying boxplot for visual outlier detection.")
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[columns])
        plt.show()
        print("Boxplot displayed. Proceeding to next step...")
        result = df
    elif method == 'iqr':
        print("Using IQR method to remove outliers...")
        Q1 = df[columns].quantile(0.25)
        Q3 = df[columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3.5 * IQR
        upper_bound = Q3 + 3.5 * IQR
        result = df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]
        print(f"Step 4 (IQR method) completed: Outliers removed using IQR method.")
    elif method == 'zscore':
        print("Using Z-Score method to remove outliers (z > 3).")
        z_scores = np.abs(stats.zscore(df[columns]))
        result = df[(z_scores < 3).all(axis=1)]
        print(f"Step 4 (Z-Score method) completed: Outliers removed based on Z-Score.")
    else:
        print("Invalid method chosen for outlier removal, no outliers removed.")
        result = df

    return result


# Step 5: Check Data Quality
def check_data_quality(df, min_range=0.1, missing_threshold=0.05):
    """
    Checks the quality of the data:
    - Missing data threshold: if the missing percentage in any column exceeds this, the condition fails.
    - Range check: checks if the difference between max and min values in any column is too small.
    """
    print(f"Running Step 5: Checking data quality...")
    missing_percentage = df.isnull().mean().max()
    range_check = (df.max() - df.min()).max()

    print(f"Step 5: Checking missing data percentage... Found {missing_percentage * 100:.2f}% missing data.")
    if missing_percentage > missing_threshold:
        print(
            f"Warning: Missing data exceeds threshold ({missing_percentage * 100:.2f}%). Proceeding to handle missing data.")
        return False

    print(f"Step 5: Checking range of values in columns... Max-Min difference found: {range_check}.")
    if range_check < min_range:
        print(f"Warning: The range is too small ({range_check}), data quality is compromised.")
        return False

    print("Step 5 completed: Data quality check passed.")
    return True


# Step 6: Augmenting Data to Multiply by 4
def augment_data(df):
    """
    Augments the dataset by 4 times:
    - Method 1: Random Sampling (Bootstrap) to duplicate the data
    - Method 2: Adding Gaussian noise (ensuring non-negative values)
    - Method 3: Using Gaussian Mixture Model to generate synthetic data (ensuring non-negative values)
    """
    print("Running Step 6: Starting data augmentation...")

    # Method 1: Random Sampling (Bootstrap) to duplicate the data
    print("Method 1: Performing random sampling (bootstrap) to duplicate the data.")
    df_augmented = pd.concat([df] * 4, ignore_index=True)
    print(f"Method 1 completed: Data augmented by {4} times using random sampling.")

    # Method 2: Adding Gaussian noise to the dataset (Ensuring no negative values)
    print("Method 2: Adding Gaussian noise while ensuring no negative values.")
    noise = np.random.normal(loc=0, scale=0.01, size=df.shape)  # mean=0, std=0.01 noise
    df_with_noise = df + noise
    df_with_noise = np.maximum(df_with_noise, 0)  # Ensure no negative values
    df_augmented = pd.concat([df_augmented, df_with_noise], ignore_index=True)
    print("Method 2 completed: Gaussian noise added while ensuring all values are non-negative.")

    # Method 3: Using Gaussian Mixture Model to generate new synthetic data
    print("Method 3: Using Gaussian Mixture Model to generate synthetic data.")
    gmm = GaussianMixture(n_components=2)
    gmm.fit(df)
    gmm_samples, _ = gmm.sample(len(df))
    df_gmm = pd.DataFrame(gmm_samples, columns=df.columns)
    df_gmm = np.maximum(df_gmm, 0)  # Ensure no negative values
    df_augmented = pd.concat([df_augmented, df_gmm], ignore_index=True)
    print("Method 3 completed: Synthetic data generated using Gaussian Mixture Model, and negative values removed.")

    print("Step 6 completed: Data augmentation finished. The data is now augmented by 4 times.")
    return df_augmented


# Step 7: The Main Data Processing Flow
def process_data(df):
    """
    Main function to process the data:
    1. Handles missing data
    2. Normalizes the data
    3. Removes outliers
    4. Checks data quality
    5. Augments the dataset by 4 times
    """
    print("Running Step 7: Main Data Processing Flow started...")
    tuning_parameters = {
        'knn_neighbors': 5,
        'mice_max_iter': 10,
        'normalization_method': 'minmax',
        'outlier_removal_method': 'iqr'
    }

    # Step 1: Handle missing data using KNN imputation
    print("Running Step 1: Handling missing data.")
    df_handled = handle_missing_data(df, method='knn', k_neighbors=tuning_parameters['knn_neighbors'])

    # Step 2: Normalize the data using Min-Max Scaling
    print("Running Step 2: Normalizing the data.")
    df_normalized = normalize_data(df_handled, method=tuning_parameters['normalization_method'])

    # Step 3: Remove outliers using IQR method
    print("Running Step 3: Removing outliers.")
    df_cleaned = remove_outliers(df_normalized, method=tuning_parameters['outlier_removal_method'])

    # Step 4: Check if data quality is good, else reapply with adjusted parameters
    print("Running Step 4: Checking data quality.")
    if not check_data_quality(df_cleaned):
        print("Reapplying the steps with adjusted parameters...")
        tuning_parameters['knn_neighbors'] = 10
        tuning_parameters['mice_max_iter'] = 15
        tuning_parameters['normalization_method'] = 'zscore'
        tuning_parameters['outlier_removal_method'] = 'zscore'

        print("Re-running Step 1...")
        df_handled = handle_missing_data(df, method='knn', k_neighbors=tuning_parameters['knn_neighbors'])
        print("Re-running Step 2...")
        df_normalized = normalize_data(df_handled, method=tuning_parameters['normalization_method'])
        print("Re-running Step 3...")
        df_cleaned = remove_outliers(df_normalized, method=tuning_parameters['outlier_removal_method'])

    # Step 5: Augment the data by 4 times
    print("Running Step 5: Augmenting the data.")
    df_augmented = augment_data(df_cleaned)

    return df_augmented


# Step 8: Apply the data processing and save the final dataset
print("Running Step 8: Applying data processing and saving the final augmented dataset...")
df_final = process_data(df)

# Step 9: Save the final augmented dataset
df_final.to_csv("D:/Education/Datacollection/final_augmented_dataset_no_negatives.csv", index=False)

print("Data processing and augmentation completed successfully!")
