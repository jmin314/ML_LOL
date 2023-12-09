import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def preprocessing():
    # 데이터 불러오기
    data = pd.read_csv("/Users/jangminseong/Desktop/start1/ML/TermP/champion_proficiency.csv")
    # champion_pts 기준으로 상위 7개 값만 남기기
    df_top_N = data.groupby('user_name').apply(lambda x: x.nlargest(7, 'champion_pts')).reset_index(drop=True)
    df_filtered = pd.merge(data, df_top_N, how='right',
                           on=['user_name', 'champion_name', 'champion_id', 'champion_lvl', 'champion_pts'])
    # 숙련도 레벨 5 이상만 남김
    # df_filtered = df_filtered[df_filtered['champion_lvl'] >= 5].drop(columns=['champion_pts', 'champion_lvl', 'champion_id'])

    # 챔피언 이름을 원-핫 인코딩
    df_encoded = pd.get_dummies(df_filtered['champion_name'])
    df_final = pd.concat([df_filtered['user_name'], df_encoded], axis=1)
    # 이름을 제외한 데이터가 모두 같은 경우 제거
    df_final = df_final.groupby('user_name').apply(lambda x: x.max()).reset_index(drop=True)
    return df_final


def clustering(df_user):
    # user_name 컬럼 제거
    features = df_user.drop('user_name', axis=1)

    # feature 표준화 (제외해도 무관)
    scaler = StandardScaler()
    features_standardized = scaler.fit_transform(features)

    # PCA 사용, 다수의 feature를 2차원으로 축소
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_standardized)

    n_clusters = 3

    # KMean 클러스터링 적용
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_user['cluster'] = kmeans.fit_predict(features_standardized)

    # 시각화
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue='cluster', data=df_user, palette='viridis',
                    s=100)
    plt.title('K-Means clustering Result (PCA)')
    plt.show()


# 데이터셋 전처리 후 불러옴
df = preprocessing()
# 전처리 결과를 클러스터링, 시각화
clustering(df)
# 전체 데이터셋의 20%를 테스트 데이터셋으로 추출
df_sample = df.sample(frac=0.2)
# 기존 기존 데이터셋에서 테스트 데이터셋 제거
df.drop(df_sample.index, axis=0, inplace=True)

# 새로운 사용자의 데이터 생성 (1인 경우만 제공)
new_user_data = {'user_name': 'new_user', 'Akali': 1, 'Draven': 1, 'Irelia': 1, 'Kalista': 1, 'Sylas': 1}
existing_columns = df.columns[1:]
# 기존 데이터셋과 같도록 컬럼 추가
new_user_data.update({col: 0 for col in existing_columns if col not in new_user_data})
# 데이터프레임 생성
new_user = pd.DataFrame([new_user_data])


def userBasedFiltering(target_user_df):
    # 기존 데이터프레임과 타겟 데이터프레임에서 'user_name'을 드랍
    original_df_vector = df.drop('user_name', axis=1).values
    target_user_vector = target_user_df.drop('user_name', axis=1).values

    # 코사인 유사도 계산
    similarities = cosine_similarity(original_df_vector, target_user_vector)

    # 상위 5명의 사용자 인덱스 찾기
    similar_users_indices = np.argsort(similarities[:, 0])[::-1][:5]

    # 상위 5명의 사용자와 가중 평균 점수 계산
    weighted_average_scores = np.average(original_df_vector[similar_users_indices], axis=0,
                                         weights=similarities[similar_users_indices, 0])

    # 이미 new user가 1의 값을 갖는 챔피언을 제외하고 내림차순으로 정렬
    recommendations = [(champion, score) for champion, score in zip(df.columns[1:], weighted_average_scores) if
                       target_user_df[champion].iloc[0] == 0]
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations


def testMSE(df_test):
    # row 별로 테스트 데이터프레임을 순회
    for index, row in df_test.iterrows():
        # row에서 1의 값을 갖는 위치를 저장
        one_indices = row[row == 1].index
        if not one_indices.empty:
            # 첫 번째 1이 있는 위치의 값을 0으로 변경
            df_test.at[index, one_indices[0]] = 0
        # row를 데이터프레임으로 변환
        row_df = pd.DataFrame([row], columns=df_test.columns)
        max_score = 0
        best_champions = []
        # row_df로 필터링 실행 후 최고 점수를 갖는 추천 챔피언 저장
        for champion, score in userBasedFiltering(row_df):
            if score > max_score:
                max_score = score
                best_champions = [(champion, score)]
            elif score == max_score:
                best_champions.append((champion, score))

        # 최고 점수를 갖는 챔피언 출력
        # for champion, score in best_champions:
        #    print(f"Champion: {champion}, Weighted Average Score: {score}")
        # row_df[[champion for champion, score in best_champions]] = 1
        # print(row_df)
        # 최고 점수를 갖는 추천 챔피언 위치에 1 저장
        for champion, score in best_champions:
            df_test.loc[index, champion] = 1
        # df_test.loc[index, [champion for champion, score in best_champions if row_df[champion].iloc[0] == 0]] = 1
    # 추천 받은 챔피언을 반영한 데이터셋과 기존 데이터셋 간의 MSE 계산
    mse = mean_squared_error(df_sample.iloc[:, 1:], df_test.iloc[:, 1:])
    return mse


print(testMSE(df_sample.copy()))
