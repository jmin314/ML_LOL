import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error


def filter_top_n(dataFrame, n):
    df_filtered = dataFrame.groupby('user_name').apply(lambda x: x.nlargest(n, 'champion_pts')).reset_index(drop=True)
    return df_filtered


def filter_by_level(dataFrame, level):
    df_filtered = dataFrame[dataFrame['champion_lvl'] >= 5]
    return df_filtered


def preprocessing():
    # 데이터 불러오기
    data = pd.read_csv("champion_proficiency.csv", encoding='ANSI')
    # 필터링 방법을 선택하여 필터링
    data = filter_top_n(data, 7)
    df_encoded = pd.get_dummies(data['champion_name'])
    df_encoded = pd.concat([data['user_name'], df_encoded], axis=1)
    df_encoded = df_encoded.groupby('user_name').apply(lambda x: x.max()).reset_index(drop=True)
    df_final = df_encoded.set_index('user_name').T
    df_final = df_final.fillna(0)
    return df_final


df = preprocessing()


def itemBasedFiltering(target_user_data):
    # 코사인 유사도 계산
    cosine_sim_matrix = cosine_similarity(df)
    cosine_sim_matrix = pd.DataFrame(cosine_sim_matrix, index=df.index, columns=df.index)

    target_df = cosine_sim_matrix.loc[list(target_user_data.keys())[1:]]

    average_df = pd.DataFrame(target_df.mean(axis=0), columns=['평균']).T
    top5_recommendation = average_df.drop(columns=target_df.index).squeeze().nlargest(5)
    print(top5_recommendation)
    return top5_recommendation


# 새로운 사용자의 더미 데이터 생성
new_user_data = {'user_name': 'new_user', 'Akali': 1, 'Draven': 1, 'Irelia': 1, 'Sylas': 1, 'Kalista': 1}

itemBasedFiltering(new_user_data)
