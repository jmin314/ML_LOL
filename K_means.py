import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def preprocessing():
    # 데이터 불러오기
    data = pd.read_csv("/Users/jangminseong/Desktop/start1/ML/TermP/champion_proficiency.csv")
    
    # champion_pts 기준으로 상위 7개 값만 남기기
    df_top_N = data.groupby('user_name').apply(lambda x: x.nlargest(7, 'champion_pts')).reset_index(drop=True)
    df_filtered = pd.merge(data, df_top_N, how='right',
                           on=['user_name', 'champion_name', 'champion_id', 'champion_lvl', 'champion_pts'])

    # 챔피언 이름을 원-핫 인코딩
    df_encoded = pd.get_dummies(df_filtered['champion_name'])
    df_cont = pd.concat([df_filtered['user_name'], df_encoded], axis=1)
    
    # 이름을 제외한 데이터가 모두 같은 경우 제거
    df_cont = df_cont.groupby('user_name').apply(lambda x: x.max()).reset_index(drop=True)
    
    features = df_cont.drop('user_name', axis=1)

    return features

# 데이터셋 전처리 후 불러옴
df = preprocessing()

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(df)

# 훈련할 때 사용된 feature의 순서를 확인
training_feature_order = df.columns

df_test = df.sample(frac=0.05)
df_train = df.drop(df_test.index, axis=0)

df_test.to_csv('df_test.csv', index = False)
df_train.to_csv('df_train.csv', index = False)

new_user_data = {'user_name': 'new_user', 'Akali': 1, 'Draven': 1, 'Irelia': 1, 'Kalista': 1, 'Sylas': 1}
existing_columns = df.columns[0:]

# 기존 데이터셋과 같도록 컬럼 추가
new_user_data.update({col: 0 for col in existing_columns if col not in new_user_data})

# 데이터프레임 생성
new_user = pd.DataFrame([new_user_data])

# 훈련할 때 사용된 feature의 순서를 기반으로 컬럼 순서 재조정
new_user_final = new_user[training_feature_order]

# 'user_name' 열 추가
new_user_final.insert(0, 'user_name', new_user['user_name'])

print(new_user_final)

new_user_final.to_csv('asdfasdfasf.csv', index = False)

new_user_final.drop('user_name', axis=1, inplace=True)
# new_user = df_test.copy()
label = kmeans.predict(new_user_final)

print(label)

silAvg = silhouette_score(new_user_final, label)

print(silAvg)