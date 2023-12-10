import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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
    result_df = data.groupby('user_name').apply(lambda x: x.nlargest(7, 'champion_pts')).reset_index(drop=True)
    df_filtered = pd.merge(data, result_df, how='right',
                           on=['user_name', 'champion_name', 'champion_id', 'champion_lvl', 'champion_pts'])
    # Keep only rows where champion level is 5 or higher
    # df_filtered = df_filtered[df_filtered['champion_lvl'] >= 5].drop(
    #     columns=['champion_pts', 'champion_lvl', 'champion_id'])

    # One-hot encode champion names
    df_encoded = pd.get_dummies(df_filtered['champion_name'])
    df_final = pd.concat([df_filtered['user_name'], df_encoded], axis=1)
    # Remove rows where all non-name columns are the same
    df_final = df_final.groupby('user_name').apply(lambda x: x.max()).reset_index(drop=True)
    return df_final
    
def clustering(df_user):
    """
    Perform KMeans clustering on the input DataFrame and visualize the result using PCA.

    Parameters:
    - df_user (pd.DataFrame): DataFrame containing user champion proficiency data.

    Returns:
    - None
    """
    # Drop the 'user_name' column
    features = df_user.drop('user_name', axis=1)

    # Standardize features (optional)
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)

    # Apply PCA to reduce features to 2 dimensions
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_standardized)

    n_clusters = 5

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_user['cluster'] = kmeans.fit_predict(features_standardized)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue='cluster', data=df_user, palette='viridis',
                    s=100)
    plt.title('K-Means Clustering Result (PCA)')
    plt.show()

# Load the dataset after preprocessing
df = preprocessing()

# Generate data for a new user (provide data for champions with a value of 1)
new_user_data = {'user_name': 'new_user', 'Akali': 1, 'Draven': 1, 'Irelia': 1, 'Kalista': 1, 'Sylas': 1}
existing_columns = df.columns[1:]
new_user_data.update({col: 0 for col in existing_columns if col not in new_user_data})
new_user = pd.DataFrame([new_user_data])

def userBasedFiltering(new_user_df):
    """
    Perform user-based filtering to recommend champions for a new user.

    Parameters:
    - new_user_df (pd.DataFrame): DataFrame containing new user data with champion preferences.

    Returns:
    - list: List of tuples containing recommended champions and their weighted average scores.
    """
    # Calculate weighted average without using weights
    user_vector_matrix = df.drop('user_name', axis=1).values
    new_user_vector = new_user_df.drop('user_name', axis=1).values

    # Calculate cosine similarity
    similarities = cosine_similarity(user_vector_matrix, new_user_vector)

    # Find indices of the top 5 similar users
    similar_users_indices = np.argsort(similarities[:, 0])[::-1][:5]

    # Calculate weighted average scores for the top 5 similar users
    weighted_average_scores = np.average(user_vector_matrix[similar_users_indices], axis=0,
                                         weights=similarities[similar_users_indices, 0])

    # Exclude champions that the new user already has a value of 1 and sort in descending order
    filtered_recommendations = [(champion, score) for champion, score in zip(df.columns[1:], weighted_average_scores) if
                                new_user_data[champion] != 1]
    filtered_recommendations.sort(key=lambda x: x[1], reverse=True)
    return filtered_recommendations

# Print recommended champions and their weighted average scores
for champion, score in userBasedFiltering(new_user):
    if score != 0:
        print(f"Champion: {champion}, Weighted Average Score: {score}")

# Visualize clustering results after preprocessing
clustering(df)

