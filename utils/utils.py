# Data Processing and Analysis
import pandas as pd  
import numpy as np        
from scipy import stats   
from sklearn.metrics import f1_score
import logging

import json
import os
import sys

# Visualization
import geopandas as gpd
import matplotlib.pyplot as plt 
import seaborn as sns    
from matplotlib.colors import LinearSegmentedColormap       

# Others
import holidays
from sklearn.preprocessing import OneHotEncoder
from scipy.cluster.hierarchy import linkage, dendrogram
import joblib

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Setup logger
log = logging.getLogger(__name__)

# Color of plots
plot_color = '#568789'
custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", ['#568789', '#efb440'])

# Visualizations

def plot_missing_values_dendrogram(df, figsize=(10, 6), leaf_rotation=90, leaf_font_size=10, title='Dendrogram of Missing Values'):
    missing_matrix = df.isnull().astype(int)
    
    linkage_matrix = linkage(missing_matrix.T, method='ward', metric='euclidean')
    
    plt.figure(figsize=figsize)
    dendrogram(
        linkage_matrix,
        labels=missing_matrix.columns,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size
    )
    
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Euclidean Distance')
    
    plt.tight_layout()
    plt.show()

def plot_numerical_features(df, numerical_features, figsize=(15, 5), plot_color=plot_color):
    fig, axes = plt.subplots(len(numerical_features), 2, figsize=(figsize[0], figsize[1] * len(numerical_features)))
    fig.suptitle('Distribution of Numerical Features', fontsize=16, y=1.02)

    for idx, feature in enumerate(numerical_features):
        # Histogram with KDE
        sns.histplot(data=df, x=feature, kde=True, ax=axes[idx, 0], color=plot_color)
        axes[idx, 0].set_title(f'Distribution of {feature}')
        axes[idx, 0].set_xlabel(feature)

        # Box plot
        sns.boxplot(data=df, y=feature, ax=axes[idx, 1], color=plot_color)
        axes[idx, 1].set_title(f'Box Plot of {feature}')

    plt.tight_layout()
    plt.show()

def plot_feature_distribution(df, feature, figsize=(15, 5), plot_color=plot_color, xlim=(0, 1500)):

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Distribution Analysis of {feature}', fontsize=16, y=1.02)

    sns.histplot(data=df, x=feature, kde=False, ax=axes[0], color=plot_color)
    axes[0].set_title(f'Distribution of {feature}')
    axes[0].set_xlabel(feature)

    axes[0].set_xlim(xlim)  

    sns.boxplot(data=df, y=feature, ax=axes[1], color=plot_color)
    axes[1].set_title(f'Box Plot of {feature}')

    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, 
                              figsize=(10, 8), 
                              title='Correlation Heatmap of Numerical Features', 
                              custom_cmap=None, 
                              center=0, 
                              fmt='.2f', 
                              cbar_shrink=0.8):
    
    plt.figure(figsize=figsize)
    
    correlation_matrix = df.corr()
    
    if custom_cmap is None:
        custom_cmap = plt.cm.coolwarm
    
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap=custom_cmap,
                center=center,
                fmt=fmt,
                cbar_kws={"shrink": cbar_shrink})
    
    plt.title(title)
    
    plt.tight_layout()
    plt.show()

def plot_feature_relationships(df, feature_pairs, figsize=(15, 15), plot_color=plot_color):
    n_plots = len(feature_pairs)
    n_rows = (n_plots + 1) // 2  
    n_cols = 2 if n_plots > 1 else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('Key Feature Relationships', fontsize=16, y=1.02)
    
    if n_rows * n_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    for i, (x_feature, y_feature) in enumerate(feature_pairs):
        sns.scatterplot(data=df, 
                        x=x_feature, 
                        y=y_feature, 
                        ax=axes[i],
                        color=plot_color)
        axes[i].set_title(f'{x_feature} vs {y_feature}')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_top_categorical_values(df, features, top_n=10, figsize=(10, 6), plot_color=plot_color):
    for feature in features:
        if feature in df.columns:  # Check if the feature exists in the DataFrame
            # Get the top N values for the feature
            top_values = df[feature].value_counts().head(top_n)

            # Create a bar plot
            plt.figure(figsize=figsize)
            sns.barplot(x=top_values.values, y=top_values.index, color=plot_color)
            plt.title(f'Top {top_n} Most Frequent Values in {feature}')
            plt.xlabel('Frequency')
            plt.ylabel(feature)
            plt.show()
        else:
            print(f"Warning: Feature '{feature}' not found in the DataFrame.")

def plot_categorical_features(categorical_df, max_features_per_plot=6, figsize=(20, 15),
                              exclude_features=['Carrier Name', 'County of Injury', 'Zip Code']):
    feature_pairs = {
        'WCIO Cause of Injury Code': 'WCIO Cause of Injury Description',
        'WCIO Nature of Injury Code': 'WCIO Nature of Injury Description',
        'WCIO Part Of Body Code': 'WCIO Part Of Body Description',
        'Industry Code': 'Industry Code Description'
    }
    
    filtered_features = [
        feat for feat in categorical_df.columns 
        if feat not in exclude_features and feat not in feature_pairs.values()
    ]

    num_features = len(filtered_features)
    num_plots = (num_features + max_features_per_plot - 1) // max_features_per_plot

    for plot_idx in range(num_plots):
        start_idx = plot_idx * max_features_per_plot
        end_idx = min((plot_idx + 1) * max_features_per_plot, num_features)
        current_features = filtered_features[start_idx:end_idx]

        fig, axes = plt.subplots(
            nrows=(len(current_features) + 2) // 3,
            ncols=3,
            figsize=(20, 5 * ((len(current_features) + 2) // 3))
        )
        axes = axes.flatten()

        for i, feature in enumerate(current_features):
            value_counts = categorical_df[feature].value_counts()
            title = feature
            if len(value_counts) > 10:
                value_counts = value_counts.head(10)  
                title = f"{feature} (Top 10 Values)"
            sns.barplot(x=value_counts.index, y=value_counts.values, color=plot_color, ax=axes[i])
            axes[i].set_title(title)
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
            axes[i].set_xlabel(None)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

# Function to create bar plots for features with fewer unique values
def plot_value_counts(df, features, max_categories=10, n_cols=2):
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, feature in enumerate(features):
            row = idx // n_cols
            col = idx % n_cols
            
            value_counts = df[feature].value_counts()
            if len(value_counts) > max_categories:
                # Keep top categories and group others
                other_count = value_counts[max_categories:].sum()
                value_counts = value_counts[:max_categories]
                value_counts['Others'] = other_count
            
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[row, col])
            axes[row, col].set_title(f'Distribution of {feature}')
            axes[row, col].set_xlabel('Count')
            # Remove empty subplots
        for idx in range(idx + 1, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])
            
        plt.tight_layout()
        plt.show()


def plot_cases_by_county(df_cleaned):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    shapefile_path = os.path.join(current_dir, '..', 'data', '03_context', 'Counties.shp')
    shapefile_path = os.path.normpath(shapefile_path)
    
    ny_counties = gpd.read_file(shapefile_path)
    
    cases_per_county_df = df_cleaned['County of Injury'].value_counts().reset_index()
    
    cases_per_county_df.columns = ['NAME', 'Count']
    
    cases_per_county_df['NAME'] = cases_per_county_df['NAME'].str.capitalize()
    
    ny_counties = ny_counties.merge(cases_per_county_df, on='NAME', how='right')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ny_counties.plot(column='Count', ax=ax, legend=True,
                     legend_kwds={'label': "Number of Cases by County",
                                  'orientation': "horizontal"},
                     cmap=custom_cmap)
    
    ax.set_axis_off()
    
    plt.title('Number of Cases by County')
    plt.show()

def plot_distribution_and_boxplot(df, column_name, color='#568789'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.histplot(df[column_name], kde=True, bins=30, color=color, ax=axes[0])
    axes[0].set_title(f"Distribution of {column_name}")
    axes[0].set_xlabel(column_name)
    axes[0].set_ylabel("Frequency")

    sns.boxplot(x=df[column_name], color=color, ax=axes[1])
    axes[1].set_title(f"Boxplot of {column_name}")
    axes[1].set_xlabel(column_name)

    plt.tight_layout()

    plt.show()

def plot_claim_counts_heatmap(df, 
                               date_column='Accident Date', 
                               figsize=(10, 8), 
                               title='Claim Counts by Year and Month', 
                               ):
    
    df['Accident Year'] = df[date_column].dt.year
    df['Accident Month'] = df[date_column].dt.month
    
    claim_counts = df.groupby(['Accident Year', 'Accident Month']).size().unstack(fill_value=0)
    
    claim_counts.index = claim_counts.index.astype(int)
    claim_counts.columns = claim_counts.columns.astype(int) 
    
    plt.figure(figsize=figsize)
    sns.heatmap(claim_counts, annot=False, cmap=custom_cmap)
    
    plt.title(title)
    plt.xlabel('Accident Month')
    plt.ylabel('Accident Year')
    
    plt.tight_layout()
    plt.show()

# Other

def datatype_changes(df_list):
    for df in df_list:
        # Select date columns
        date_cols = df.columns[df.columns.str.contains('Date')]
        # Transform into date using pandas
        df[date_cols] = df[date_cols].apply(pd.to_datetime)

        # Select code columns
        code_cols = df.columns[df.columns.str.contains('Code')]
        # Transform code columns to float
        df[code_cols] = df[code_cols].astype('str')

        float_to_int(df, ['Age at Injury', 'Birth Year'])


def limit_feature(df_list, feature, minimum=None, maximum=None, verbose=True):
    for i, df in enumerate(df_list):
        if verbose:
            print(f"DataFrame {i+1}:")
            print(f'Number of rows with 0: {len(df[df[feature] == 0])}.')
            if minimum is not None:
                print(f'Number of rows below {minimum}: {len(df[df[feature] < minimum])}.')
            if maximum is not None:
                print(f'Number of rows above {maximum}: {len(df[df[feature] > maximum])}.')
            print()
            
        # Apply the limits to the feature, setting out-of-bounds values to NaN
        if minimum is not None:
            df[feature] = df[feature].where(df[feature] >= minimum, np.nan)
        if maximum is not None:
            df[feature] = df[feature].where(df[feature] <= maximum, np.nan)

def extract_dates_components(df_list, date_columns):
    # Loop through each dataframe
    for df in df_list:
        # Loop through each date column and extract year, month, day, and day of the week
        for col in date_columns:
            if col in df.columns:
                # Convert to datetime if not already
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    # Extract components from each date column, filling missing values with -1
                    df[f'{col}_Year'] = df[col].dt.year.fillna(-1).astype(int)
                    df[f'{col}_Month'] = df[col].dt.month.fillna(-1).astype(int)
                    df[f'{col}_Day'] = df[col].dt.day.fillna(-1).astype(int)
                    df[f'{col}_DayOfWeek'] = df[col].dt.dayofweek.fillna(-1).astype(int)
                except Exception as e:
                    print(f"Could not process date column {col}: {e}")
            else:
                print(f"Date column {col} not found in dataframe")

def dist_dates(df_list):
    for df in df_list:
        # Calculate the time difference between relevant dates
        df['Days_to_First_Hearing'] = (df['First Hearing Date'] - df['Accident Date']).dt.days
        df['Days_to_C2'] = (df['C-2 Date'] - df['Accident Date']).dt.days
        df['Days_to_C3'] = (df['C-3 Date'] - df['Accident Date']).dt.days

        df['Days_to_First_Hearing'] = df['Days_to_First_Hearing'].fillna(-1).astype(int)
        df['Days_to_C2'] = df['Days_to_C2'].fillna(-1).astype(int)
        df['Days_to_C3'] = df['Days_to_C3'].fillna(-1).astype(int)



# Define the function that assigns a season based on the exact date
def get_season(date):
    # Extract the month and day from the date
    month_day = (date.month, date.day)
    
    # Define the season boundaries (start of each season)
    winter_start = (12, 21)
    spring_start = (3, 20)
    summer_start = (6, 21)
    fall_start = (9, 23)
    
    # Determine the season based on the month and day
    if (month_day >= winter_start) or (month_day < spring_start):
        return 'Winter'
    elif (month_day >= spring_start) and (month_day < summer_start):
        return 'Spring'
    elif (month_day >= summer_start) and (month_day < fall_start):
        return 'Summer'
    else:
        return 'Fall'
    
def flag_public_holiday_accidents(df, date_column, state='NY'):
    """
    Flags accidents that happened on public holidays in a given state.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the accident data.
    date_column (str): The name of the column containing the accident dates.
    state (str): The state for which public holidays are to be checked (default is 'NY').
    
    Returns:
    pd.DataFrame: The input DataFrame with an additional column flagging public holiday accidents.
    """
    
    # Determine the year range from the 'Accident Date' column
    start_year = df[date_column].dt.year.min()  # Get the earliest year
    end_year = df[date_column].dt.year.max()  # Get the latest year

    # Initialize an empty list to hold all the public holidays for the specified state
    holiday_dates = []

    # Loop through all years from the minimum to the maximum year in the dataset
    for year in range(start_year, end_year + 1):  # Including the end year
        # Get the public holidays for the current year in the specified state
        holidays_in_year = holidays.US(years=year, state=state)
        holiday_dates.extend(holidays_in_year.keys())  # Add holidays for the current year to the list

    # Convert the list of public holiday dates to a pandas datetime format
    holiday_dates = pd.to_datetime(holiday_dates)

    # Create a new column in the dataframe to flag accidents that occurred on public holidays
    df['Holiday_Accident'] = df[date_column].isin(holiday_dates).astype(int)

    return df

def flag_weekend_accidents(df, date_column):
    """
    Flags accidents that happened on weekends (Saturday or Sunday).
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the accident data.
    date_column (str): The name of the column containing the accident dates.
    
    Returns:
    pd.DataFrame: The input DataFrame with an additional column flagging weekend accidents.
    """
    
    # Create a new column to flag accidents that occurred on weekends
    df['Weekend_Accident'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)  # 5=Saturday, 6=Sunday

    return df
    

def frequency_encoding(df, column_name, test_df, save_encoding, **kwargs):
    # Only proceed if column exists in both DataFrames
    if column_name not in df.columns or column_name not in test_df.columns:
        print(f"[frequency_encoding] Skipping '{column_name}' (not found in both DataFrames)")
        return
    new_column_name = f"Enc {column_name}"
    df[column_name] = df[column_name].fillna("Unknown")
    test_df[column_name] = test_df[column_name].fillna("Unknown")
    value_counts = df[column_name].value_counts()
    freq_mapping = value_counts / len(df)
    df[new_column_name] = df[column_name].map(freq_mapping)
    test_df[new_column_name] = test_df[column_name].map(freq_mapping).fillna(0)

def apply_frequency_encoding(df, test_df, save_encoding=False, **kwargs):
    frequency_encoder_vars = [
        'County of Injury',
        'District Name',
        'Industry Code',
        'WCIO Cause of Injury Code',
        'WCIO Nature of Injury Code',
        'WCIO Part Of Body Code',
        'Zip Code'
    ]
    for col in frequency_encoder_vars:
        if col in df.columns and col in test_df.columns:
            frequency_encoding(df, col, test_df, save_encoding, **kwargs)
        else:
            print(f"[apply_frequency_encoding] Skipping '{col}' (not found in both DataFrames)")
    # Only drop columns that exist
    df = df.drop(columns=[col for col in frequency_encoder_vars if col in df.columns])
    test_df = test_df.drop(columns=[col for col in frequency_encoder_vars if col in test_df.columns])
    return df, test_df

def apply_one_hot_encoding(train_df, other_df, features, save_encoder=False):
    """
    Applies one-hot encoding to the specified features in the DataFrame,
    and optionally saves the encoder categories to a JSON file.

    Parameters:
    - train_df (pd.DataFrame): The training DataFrame.
    - other_df (pd.DataFrame): Another DataFrame to apply the same encoding.
    - features (list): List of column names to be encoded.
    - save_encoder (Boolean): Optional. Save encoder flag.

    Returns:
    - pd.DataFrame: Transformed training DataFrame.
    - pd.DataFrame: Transformed other DataFrame.
    """
    # Initialize the encoder with drop='first' option for avoiding multicollinearity
    oh_enc = OneHotEncoder(drop='first', sparse_output=False)

    # Fit the encoder on the train dataset and transform both datasets
    train_encoded_features = oh_enc.fit_transform(train_df[features]).astype(int)
    other_encoded_features = oh_enc.transform(other_df[features]).astype(int)

    # Save the encoder
    if save_encoder:
        # Create folder if it does not exist
        folder_path = "../data/05_encoders/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # Save encoder
        joblib.dump(oh_enc, '../data/05_encoders/OneHotEncoder.pkl')

    # Create encoded DataFrame with proper feature names
    encoded_feature_names = oh_enc.get_feature_names_out(features)
    train_encoded_df = pd.DataFrame(train_encoded_features, columns=encoded_feature_names, index=train_df.index)
    other_encoded_df = pd.DataFrame(other_encoded_features, columns=encoded_feature_names, index=other_df.index)

    # Combine the encoded features with the rest of the original DataFrames (dropping the original features)
    new_train = pd.concat([train_df.drop(columns=features), train_encoded_df], axis=1)
    new_other = pd.concat([other_df.drop(columns=features), other_encoded_df], axis=1)

    return new_train, new_other


def categorize_impact(impact):
    if impact > 50000:
        return 0 # Low
    elif 1000 <= impact <= 50000:
        return 1 # Medium
    else:
        return 2 # High

def financial_impact(df):
    adjusted_dependents = df['Number of Dependents'].replace(0, 1)
    
    financial_impact = df['Average Weekly Wage'] / adjusted_dependents

    df['Financial Impact Category'] = financial_impact.apply(categorize_impact)


def sine_cosine_encoding(df, column, mapping):
    df[f"{column}_Ordinal"] = df[column].map(mapping)
    df[f"{column}_Sin"] = np.sin(2 * np.pi * df[f"{column}_Ordinal"] / 4)
    df[f"{column}_Cos"] = np.cos(2 * np.pi * df[f"{column}_Ordinal"] / 4)
    return df.drop(columns=[f"{column}_Ordinal", column])

def calculate_birth_year(df):
    # Ensure the correct format of 'Birth Year'
    df['Accident Date'] = pd.to_datetime(df['Accident Date'], errors='coerce')

    # Filter the rows where 'Birth Year' is NaN, but 'Age at Injury' and 'Accident Date' are not NaN
    condition = df['Birth Year'].isna() & df['Age at Injury'].notna() & df['Accident Date'].notna()

    # Replace missing 'Birth Year' with the difference between 'Accident Date' year and 'Age at Injury'
    df.loc[condition, 'Birth Year'] = df.loc[condition, 'Accident Date'].dt.year - df.loc[condition, 'Age at Injury']


def save_results_csv(model, features, y_train, y_train_pred, y_val, y_val_pred):
    # Define the model name
    model_name = type(model).__name__

    # Calculate F1 scores
    f1_train = f1_score(y_train, y_train_pred, average='macro')
    f1_val = f1_score(y_val, y_val_pred, average='macro')

    # Get model parameters
    model_params = model.get_params()

    # Create a dictionary of results, including the model name in the third column
    result = {
        'Model Name': model_name,
        'F1 Train': f1_train,
        'F1 Validation': f1_val,
        **model_params, 
        'Feature Group': features,
    }

    # Convert dictionary to DataFrame
    result_df = pd.DataFrame([result])

    # Define the file name (CSV in this case)
    filename = "model_results.csv"

    # Check if the file already exists
    try:
        # If the file exists, append the new results without the header
        existing_df = pd.read_csv(filename)
        result_df.to_csv(filename, index=False, header=False, mode='a')  # Append to existing file
    except FileNotFoundError:
        # If the file does not exist, create a new file and write the header
        result_df.to_csv(filename, index=False)

    print(f"Results added to {filename}")

def NA_imputer(train_df, test_df, save_median=False, **kwargs):

    columns = ["Age at Injury","Average Weekly Wage"]
    imputation_value = train_df[columns].median()

    for col in columns:
        train_df[col] = train_df[col].fillna(imputation_value[col])
        test_df[col] = test_df[col].fillna(imputation_value[col])

    # Handle Accident Date if it exists
    if 'Accident Date' in train_df.columns and 'Accident Date' in test_df.columns:
        train_df['Accident Date'] = pd.to_datetime(train_df['Accident Date'], errors='coerce')
        test_df['Accident Date'] = pd.to_datetime(test_df['Accident Date'], errors='coerce')

        # Calculate birth_year from accident_date and age_at_injury if both exist
        if 'Birth Year' in train_df.columns and 'Age at Injury' in train_df.columns:
            condition = train_df['Birth Year'].isna() & train_df['Age at Injury'].notna() & train_df['Accident Date'].notna()
            train_df.loc[condition, 'Birth Year'] = train_df.loc[condition, 'Accident Date'].dt.year - train_df.loc[condition, 'Age at Injury']

            # Filter the rows where 'Birth Year' is NaN, but 'Age at Injury' and 'Accident Date' are not NaN
            condition = test_df['Birth Year'].isna() & test_df['Age at Injury'].notna() & test_df['Accident Date'].notna()
            # Replace missing 'Birth Year' with the difference between 'Accident Date' year and 'Age at Injury'
            test_df.loc[condition, 'Birth Year'] = test_df.loc[condition, 'Accident Date'].dt.year - test_df.loc[condition, 'Age at Injury']

        # Drop accident_date as it's not needed for modeling
        train_df.drop('Accident Date', axis=1, inplace=True)
        test_df.drop('Accident Date', axis=1, inplace=True)

def create_new_features(train_df, test_df, calculate=True):
    median_wage = train_df['Average Weekly Wage'].median()
    
    train_df['Relative_Wage'] = np.where(train_df['Average Weekly Wage'] > median_wage, 1, 0)
    test_df['Relative_Wage'] = np.where(test_df['Average Weekly Wage'] > median_wage, 1, 0)

    financial_impact(train_df)
    financial_impact(test_df)

    age_bins = [0, 25, 40, 55, 70, 100]  # Define bins
    age_labels = [0, 1, 2, 3, 4]         # Define labels

    train_df['Age_Group'] = pd.cut(
        train_df['Age at Injury'], bins=age_bins, labels=age_labels, right=False, include_lowest=True
    ).astype('category').cat.codes

    test_df['Age_Group'] = pd.cut(
        test_df['Age at Injury'], bins=age_bins, labels=age_labels, right=False, include_lowest=True
    ).astype('category').cat.codes

def version_control():

    file_path = 'version.txt'

    try:
        with open(file_path, 'r') as file:
            count = int(file.read().strip())
    except FileNotFoundError:
        count = 0

    count += 1

    with open(file_path, 'w') as file:
        file.write(str(count))

    return count

def custom_trial_dirname(trial, ):
    return f"../GridSearch/trial_{trial.trial_id}"

def float_to_int(df, columns):
    for col in columns:
        df[col] = df[col].astype('Int64')


# Function to get the current experiment count
def get_experiment_count(file_name):
    if not os.path.exists(file_name):
        return 0
    with open(file_name, "r") as file:
        data = file.readlines()
    return sum(1 for line in data if line.strip().startswith("{"))  # Count JSON objects

def save_scores(model_name, best_config, best_f1_score, hours_passed):

    # File name for saving results
    file_name = "Runs.txt"

    experiment_count = get_experiment_count(file_name) + 1

    # Prepare data to save
    output_data = {
        "experiment_count": experiment_count,
        "model_name": model_name,
        "best_trial_config": best_config,
        "best_f1_score": best_f1_score,
        "time":hours_passed
    }

    # Append the output to the file
    with open(file_name, "a") as file:
        file.write(json.dumps(output_data, indent=4))
        file.write("\n\n")  # Add a blank line between runs for readability

def remove_outliers(df):
    df = df[df['Age at Injury'].le(90) | df['Age at Injury'].isna()]
    df = df[df['Average Weekly Wage'].lt(100000) | df['Average Weekly Wage'].isna()]
    df = df[df['Birth Year'].gt(1938) | df['Birth Year'].isna()]
    df = df[df['IME-4 Count'].lt(40)] # Has no missing values