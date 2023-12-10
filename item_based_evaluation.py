import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm

def filter_top_n(dataFrame, n):
    """
    Filters the top N rows for each user based on the 'champion_pts' column.

    Parameters:
    - dataFrame (pd.DataFrame): Input DataFrame containing user champion proficiency data.
    - n (int): Number of top rows to retain for each user.

    Returns:
    - pd.DataFrame: DataFrame with the top N rows for each user.
    """
    df_filtered = dataFrame.groupby('user_name').apply(lambda x: x.nlargest(n, 'champion_pts')).reset_index(drop=True)
    return df_filtered

def filter_by_level(dataFrame, level):
    """
    Filters rows where 'champion_lvl' is greater than or equal to the specified level.

    Parameters:
    - dataFrame (pd.DataFrame): Input DataFrame containing user champion proficiency data.
    - level (int): Minimum champion level to retain.

    Returns:
    - pd.DataFrame: DataFrame with rows where 'champion_lvl' is greater than or equal to the specified level.
    """
    df_filtered = dataFrame[dataFrame['champion_lvl'] >= level]
    return df_filtered

def preprocessing():
    """
    Perform data preprocessing steps, including filtering top rows and one-hot encoding of champion names.

    Returns:
    - pd.DataFrame: Transposed DataFrame with one-hot encoded champion names, filled with 0 for missing values.
    """
    data = pd.read_csv("champion_proficiency.csv")
    data = filter_top_n(data, 7)
    df_encoded = pd.get_dummies(data['champion_name'])
    df_encoded = pd.concat([data['user_name'], df_encoded], axis=1)
    df_encoded = df_encoded.groupby('user_name').apply(lambda x: x.max()).reset_index(drop=True)
    df_final = df_encoded.set_index('user_name').T
    df_final = df_final.fillna(0)
    return df_final

df = preprocessing()

def itemBasedFiltering(target_user_data):
    """
    Perform item-based filtering to recommend champions for a new user.

    Parameters:
    - target_user_data (dict): Dictionary containing user data with champion preferences.

    Returns:
    - pd.Series: Top 5 recommended champions based on item-based filtering.
    """
    # Calculate cosine similarity
    cosine_sim_matrix = cosine_similarity(df)
    cosine_sim_matrix = pd.DataFrame(cosine_sim_matrix, index=df.index, columns=df.index)
    target_df = cosine_sim_matrix.loc[list(target_user_data.keys())[1:]]

    average_df = pd.DataFrame(target_df.mean(axis=0), columns=['Average']).T
    top5_recommendation = average_df.drop(columns=target_df.index).squeeze().nlargest(5)
    return top5_recommendation

# Generate dummy data for a new user
new_user_data = {'user_name': 'new_user', 'Akali': 1, 'Draven': 1, 'Irelia': 1, 'Sylas': 1, 'Kalista': 1}
print(itemBasedFiltering(new_user_data))

def evaluation(df_test):
    """
    Evaluate the item-based recommendation algorithm using Mean Squared Error (MSE), accuracy, and overall accuracy.

    Parameters:
    - df_test (pd.DataFrame): Test DataFrame with user preferences.
    """
    # Load data
    data = pd.read_csv("champion_proficiency.csv")

    # Keep the top 7 values based on 'champion_pts'
    df_top_N = data.groupby('user_name').apply(lambda x: x.nlargest(7, 'champion_pts')).reset_index(drop=True)
    df_filtered = pd.merge(data, df_top_N, how='right',
                           on=['user_name', 'champion_name', 'champion_id', 'champion_lvl', 'champion_pts'])

    # One-hot encode champion names
    df_encoded = pd.get_dummies(df_filtered['champion_name'])
    df_final = pd.concat([df_filtered['user_name'], df_encoded], axis=1)
    # Remove rows where all non-name columns are the same
    df_final = df_final.groupby('user_name').apply(lambda x: x.max()).reset_index(drop=True)

    # Extract 20% of the dataset as the test dataset
    df_sample = df_final.sample(frac=0.2)
    df_test = df_sample.copy()

    # Remove the test dataset from the original dataset
    df_final.drop(df_test.index, axis=0, inplace=True)

    for index, row in df_test.iterrows():
        # Find the indices where the value is 1 in the row
        one_indices = row[row == 1].index
        if not one_indices.empty:
            # Change the value of the first 1 to 0
            df_test.at[index, one_indices[0]] = 0

        # Convert the row to a dictionary
        row_data = row[row != 0].to_dict()

        # Get recommendations using item-based filtering
        result = itemBasedFiltering(row_data)
        champion = result.idxmax()
        # print(result.max())
        df_test.loc[index, champion] = 1

    # Calculate MSE between the recommended champions and the actual champions in the test dataset
    mse = mean_squared_error(df_sample.iloc[:, 1:], df_test.iloc[:, 1:])

    # Calculate accuracy using the correct formula
    correct_predictions = np.sum(df_sample.iloc[:, 1:] == df_test.iloc[:, 1:])
    total_samples = len(df_test)  # Total test samples

    correct_predictions_df = pd.DataFrame(correct_predictions, columns=['Correct Predictions'])

    # Total predictions for each champion
    correct_predictions_df['Total Predictions'] = total_samples
    # Calculate accuracy for each champion
    correct_predictions_df['Accuracy'] = correct_predictions_df['Correct Predictions'] / total_samples

    # Calculate overall accuracy
    total_correct_predictions = correct_predictions_df['Correct Predictions'].sum()
    overall_accuracy = total_correct_predictions / (total_samples * len(correct_predictions_df))

    # Print results
    print("MSE: {:.5f}\n".format(mse))
    print("<Champion-wise Accuracy>")
    print(correct_predictions_df[['Correct Predictions', 'Total Predictions', 'Accuracy']])

    print("\nOverall Accuracy: {:.5f}".format(overall_accuracy))

# Evaluate the item-based recommendation algorithm
evaluation(df_sample.copy())
