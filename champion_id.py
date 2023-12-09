import os
import re
import subprocess
import csv

line_list = ['top', 'jungle', 'mid', 'adc', 'support']
champion_id_dict_list = []
# for line in line_list :
#     os.system("curl 'https://www.op.gg/champions?region=global&tier=all&position=" + line + "' > ./champion_id/" + line)

# 1번 방법
for line in line_list :
    break
    with open("./champion_id/" + line, 'r') as file:
        content = file.read()

        # 정규표현식 패턴
        pattern = r'"key":"(.*?)","champion_id":(\d+)'

        # 정규표현식을 사용하여 값을 추출
        matches = re.findall(pattern, content)

        # 결과 출력
        for match in matches:
            champion_name = match[0]  # "name" 값
            champion_id = int(match[1])  # "champion_id" 값을 정수로 변환

            if len(champion_name) > 20 : # 키가 20글자 넘어가면 이상 값이여서 무시 -> 코드 다시 확인
                continue

            champion_id_dict = {}
            champion_id_dict['champion_name'] = champion_name
            champion_id_dict['champion_id'] = champion_id

            champion_id_dict_list.append(champion_id_dict)

# 2번 방법
url_list = []

for line in line_list:
    with open("./champion_id/" + line, 'r') as file:
        content = file.read()

        # 정규표현식 패턴
        pattern = r'champions/([^/]+)/build/([^/]+)\?'

        # findall을 사용하여 모든 URL 추출
        matches = re.findall(pattern, content)

        for match in matches:
            champion_name = match[0]
            champion_line = match[1]
            
            url = f"https://www.op.gg/champions/{champion_name}/build/{champion_line}"
            url_list.append(url)

url_list = list(set(url_list))

# for i, url in enumerate(url_list) :
#     print(i, url)

for url in url_list[:1] :
    curl_command = "curl " + url
    response = subprocess.check_output(curl_command, shell=True, text=True)
    
    print("response size : ", len(response))
    print(response)
    
    # 정규표현식 패턴
    pattern = r'"id":(\d+),"key":"([^"]+)"'

    # 정규표현식을 사용하여 값을 추출
    matches = re.findall(pattern, response)

    # 결과 출력
    for match in matches:
        champion_id = int(match[0])  # "champion_id" 값을 정수로 변환
        champion_name = match[1]  # "name" 값
        
        if len(champion_name) > 20 or "Summoner" in champion_name: # 키가 20글자 넘어가면 이상 값이여서 무시 -> 코드 다시 확인
            continue

        champion_id_dict = {}
        champion_id_dict['champion_name'] = champion_name
        champion_id_dict['champion_id'] = champion_id

        champion_id_dict_list.append(champion_id_dict)

# 중복 제거
unique_champion_id_dict_list = []
for champion_id_dict in champion_id_dict_list:
    if champion_id_dict not in unique_champion_id_dict_list:
        unique_champion_id_dict_list.append(champion_id_dict)

for champion_id_dict in unique_champion_id_dict_list :
    # print(champion_id_dict)
    print(f"name : {champion_id_dict['champion_name']}, id : {champion_id_dict['champion_id']}")

print("total count :", len(unique_champion_id_dict_list))

# unique_champion_id_dict_list의 딕셔너리를 'champion_name'을 기준으로 오름차순으로 정렬
sorted_champion_id_dict_list = sorted(unique_champion_id_dict_list, key=lambda x: x['champion_name'])

csv_file_path = 'champion_id.csv'  # 저장할 CSV 파일 경로

with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = ['champion_name', 'champion_id']  # 필드 이름 리스트
    
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()  # 헤더 행(필드 이름) 작성
    
    for champion_id_dict in sorted_champion_id_dict_list:
        writer.writerow({'champion_name': champion_id_dict['champion_name'], 'champion_id': champion_id_dict['champion_id']})