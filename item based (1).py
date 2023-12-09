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
    df_filtered = dataFrame.groupby('user_name').apply(lambda x: x.nlargest(n, 'champion_pts')).reset_index(drop=True)
    return df_filtered

def filter_by_level(dataFrame, level):
    df_filtered = dataFrame[dataFrame['champion_lvl'] >= 5]
    return df_filtered

def preprocessing():
    # 데이터 불러오기
    data = pd.read_csv("/Users/jangminseong/Desktop/start1/ML/TermP/champion_proficiency.csv")
    # 필터링 방법을 선택하여 필터링
    data = filter_top_n(data, 7)
    df_encoded = pd.get_dummies(data['champion_name'])
    df_encoded = pd.concat([data['user_name'], df_encoded], axis=1)
    df_encoded = df_encoded.groupby('user_name').apply(lambda x: x.max()).reset_index(drop=True)
    df_final = df_encoded.set_index('user_name').T
    df_final = df_final.fillna(0)
    return df_final

def kMeanClustering(target_df):

    # feature 표준화 (제외해도 무관)
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(target_df)

    # PCA 사용, 다수의 feature를 2차원으로 축소
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_standardized)

    n_clusters = 5

    # KMean 클러스터링 적용
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    target_df['cluster'] = kmeans.fit_predict(features_standardized)

    # 시각화
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue='cluster', data=target_df, palette='viridis',
                    s=100)
    plt.title('K-Means clustering Result (PCA)')
    plt.show()

df = preprocessing()
kMeanClustering(df)

def itemBasedFiltering(target_user_data):
    # 코사인 유사도 계산
    cosine_sim_matrix = cosine_similarity(df)
    cosine_sim_matrix = pd.DataFrame(cosine_sim_matrix, index=df.index, columns=df.index)
    target_df = cosine_sim_matrix.loc[list(target_user_data.keys())[1:]]

    average_df = pd.DataFrame(target_df.mean(axis=0), columns=['평균']).T
    top5_recommendation = average_df.drop(columns=target_df.index).squeeze().nlargest(5)
    # print(top5_recommendation)
    return top5_recommendation

# 새로운 사용자의 더미 데이터 생성
new_user_data = {'user_name': 'new_user', 'Akali': 1, 'Draven': 1, 'Irelia': 1, 'Sylas': 1, 'Kalista': 1}

print(itemBasedFiltering(new_user_data))

def testMSE():
    # 데이터 불러오기
    data = pd.read_csv("/Users/jangminseong/Desktop/start1/ML/TermP/champion_proficiency.csv")
    # champion_pts 기준으로 상위 7개 값만 남기기
    df_top_N = data.groupby('user_name').apply(lambda x: x.nlargest(7, 'champion_pts')).reset_index(drop=True)
    df_filtered = pd.merge(data, df_top_N, how='right',
                           on=['user_name', 'champion_name', 'champion_id', 'champion_lvl', 'champion_pts'])
    # 챔피언 이름을 원-핫 인코딩
    df_encoded = pd.get_dummies(df_filtered['champion_name'])
    df_final = pd.concat([df_filtered['user_name'], df_encoded], axis=1)
    # 이름을 제외한 데이터가 모두 같은 경우 제거
    df_final = df_final.groupby('user_name').apply(lambda x: x.max()).reset_index(drop=True)

    # 전체 데이터셋의 20%를 테스트 데이터셋으로 추출
    df_sample = df_final.sample(frac=0.2)
    df_test = df_sample.copy()
    # 기존 기존 데이터셋에서 테스트 데이터셋 제거
    df_final.drop(df_test.index, axis=0, inplace=True)

    for index, row in df_test.iterrows():
        # row에서 1의 값을 갖는 위치를 저장
        one_indices = row[row == 1].index
        if not one_indices.empty:
            # 첫 번째 1이 있는 위치의 값을 0으로 변경
            df_test.at[index, one_indices[0]] = 0
        # row를 데이터프레임으로 변환
        max_score = 0
        best_champions = []
        row_data = row[row != 0].to_dict()
        # print(row_data)
        result = itemBasedFiltering(row_data)
        champion = result.idxmax()
        # print(result.max())
        df_test.loc[index, champion] = 1
    # 추천 받은 챔피언을 반영한 데이터셋과 기존 데이터셋 간의 MSE 계산
    mse = mean_squared_error(df_sample.iloc[:, 1:], df_test.iloc[:, 1:])
    return mse

print(testMSE())