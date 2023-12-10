import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocessing():
    """
    Perform data preprocessing steps:
    1. Load champion proficiency data.
    2. Keep the top 7 rows for each user based on 'champion_pts'.
    3. Merge the filtered data to retain relevant columns.
    4. One-hot encode champion names and create a transposed DataFrame.
    5. Remove rows where all non-name columns are the same.

    Returns:
    - pd.DataFrame: Transposed DataFrame with one-hot encoded champion names.
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
    return df_final

# Load preprocessed data
df = preprocessing()

# Sample 20% of the entire dataset as the test dataset
df_sample = df.sample(frac=0.2)
# Remove the test dataset from the original dataset
df.drop(df_sample.index, axis=0, inplace=True)

# Generate data for a new user (provide data for champions with a value of 1)
new_user_data = {'user_name': 'new_user', 'Akali': 1, 'Draven': 1, 'Irelia': 1, 'Kalista': 1, 'Sylas': 1}
existing_columns = df.columns[1:]
# Ensure the new user DataFrame has the same columns as the existing dataset
new_user_data.update({col: 0 for col in existing_columns if col not in new_user_data})
# Dataframe creation
new_user = pd.DataFrame([new_user_data])

def userBasedFiltering(target_user_df):
    """
    Perform user-based filtering to recommend champions for a new user.

    Parameters:
    - target_user_df (pd.DataFrame): DataFrame containing new user data with champion preferences.

    Returns:
    - list: List of tuples containing recommended champions and their weighted average scores.
    """
    # Remove the 'user_name' column from both the original and target DataFrames
    original_df_vector = df.drop('user_name', axis=1).values
    target_user_vector = target_user_df.drop('user_name', axis=1).values

    # Calculate cosine similarity
    similarities = cosine_similarity(original_df_vector, target_user_vector)

    # Find indices of the top 5 similar users
    similar_users_indices = np.argsort(similarities[:, 0])[::-1][:5]

    # Calculate weighted average scores for the top 5 similar users
    weighted_average_scores = np.average(original_df_vector[similar_users_indices], axis=0,
                                         weights=similarities[similar_users_indices, 0])

    # Exclude champions that the new user already has a value of 1 and sort in descending order
    recommendations = [(champion, score) for champion, score in zip(df.columns[1:], weighted_average_scores) if
                       target_user_df[champion].iloc[0] == 0]
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations

def evaluation(df_test):
    """
    Evaluate the recommendation algorithm using Mean Squared Error (MSE) and accuracy.

    Parameters:
    - df_test (pd.DataFrame): Test DataFrame with user preferences.
    """
    # Iterate over rows in the test DataFrame
    for index, row in df_test.iterrows():
        # Save indices where the row has a value of 1
        one_indices = row[row == 1].index
        if not one_indices.empty:
            # Change the value of the first 1 to 0
            df_test.at[index, one_indices[0]] = 0
        # Convert the row to a DataFrame
        row_df = pd.DataFrame([row], columns=df_test.columns)
        max_score = 0
        best_champions = []
        # Filter with the row DataFrame and save the recommended champion with the highest score
        for champion, score in userBasedFiltering(row_df):
            if score > max_score:
                max_score = score
                best_champions = [(champion, score)]
            elif score == max_score:
                best_champions.append((champion, score))

        # Save the champion with the highest score
        for champion, score in best_champions:
            df_test.loc[index, champion] = 1

    # Calculate MSE between the test dataset with recommendations and the original dataset
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

# Evaluate the recommendation algorithm
evaluation(df_sample.copy())
