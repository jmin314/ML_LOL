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

def kMeanClustering(target_df):
    """
    Perform KMeans clustering on the input DataFrame and visualize the result using PCA.

    Parameters:
    - target_df (pd.DataFrame): DataFrame containing user champion proficiency data.

    Returns:
    - None
    """
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(target_df)

    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_standardized)

    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    target_df['cluster'] = kmeans.fit_predict(features_standardized)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue='cluster', data=target_df, palette='viridis', s=100)
    plt.title('K-Means clustering Result (PCA)')
    plt.show()

df = preprocessing()
kMeanClustering(df)

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

