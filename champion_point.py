import subprocess
import re
import pandas as pd
import os

# Code to obtain user's proficiency per champion

def getChampionsData(name):
    # Code to fetch proficiency
    url = "https://masterychart.com/profile/kr/" + name.replace(" ", "%20")
    print(url)
    curl_command = "curl " + url

    # Request
    response = subprocess.check_output(curl_command, shell=True, text=True)

    # Regular expression to obtain proficiency per champion
    pattern = r'{"name":"([^"]+)","disp":"([^"]+)","id":(\d+),"lvl":(\d+),"pts":(\d+),"class":"([^"]+)"'
    matches = re.findall(pattern, response)

    champion_dict_list = []

    # Saving values obtained through regular expressions
    for match in matches:
        champion_dict = {}
        
        champion_dict['user_name'] = name
        champion_dict['champion_name'] = match[0]
        champion_dict['champion_id'] = match[2]
        champion_dict['champion_lvl'] = match[3]
        champion_dict['champion_pts'] = match[4]

        champion_dict_list.append(champion_dict)

    return champion_dict_list

df = pd.read_csv('data.csv')  # File containing user nicknames, etc.
df2 = pd.read_csv('champion_proficiency.csv')  # File to store proficiency data

name_list = df[df['position'].isin([4, 5])]['user_name'].tolist()  # Fetching user nicknames

name_list = list(set(name_list))  # Removing duplicates from name_list

success_count = 0
fail_count = 0
continue_count = 0

# Processing one user at a time
for i, name in enumerate(name_list):
    # Continue if already saved
    if name in df2['user_name'].values:
        print(f"{name} -> continue")
        continue_count += 1
    else:
        # Fetch proficiency per champion for the user
        championsDataDict = getChampionsData(name)
        
        if championsDataDict != []:
            df = pd.DataFrame(championsDataDict)
            df.to_csv('champion_proficiency.csv', index=False, mode='a', header=False)  # Save proficiency
            
            print(f"{name} -> success")
            success_count += 1
        else:
            print(f"{name} -> fail")  # If unable to fetch proficiency per champion (e.g., due to user's nickname change)
            fail_count += 1
    
    print(f"percentage : {i+1}/{len(name_list)}, Success : {success_count}, Fail : {fail_count}, Continue : {continue_count}")