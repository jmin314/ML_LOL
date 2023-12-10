import subprocess
import re
import pandas as pd
import os

def getChampionsData(name):
    """
    Retrieves champion proficiency data from the Mastery Chart website for a given user name.

    Parameters:
    - name (str): User name to fetch data for.

    Returns:
    - list: List of dictionaries containing champion proficiency data.
    """
    url = "https://masterychart.com/profile/kr/" + name.replace(" ", "%20")
    print(url)
    curl_command = "curl " + url

    # Execute the curl command to fetch the HTML response
    response = subprocess.check_output(curl_command, shell=True, text=True)

    # Define a regular expression pattern to extract champion data from the HTML response
    pattern = r'{"name":"([^"]+)","disp":"([^"]+)","id":(\d+),"lvl":(\d+),"pts":(\d+),"class":"([^"]+)"'
    matches = re.findall(pattern, response)

    champion_dict_list = []

    for match in matches:
        champion_dict = {}
        
        champion_dict['user_name'] = name
        champion_dict['champion_name'] = match[0]
        champion_dict['champion_id'] = match[2]
        champion_dict['champion_lvl'] = match[3]
        champion_dict['champion_pts'] = match[4]

        champion_dict_list.append(champion_dict)

    return champion_dict_list

# Read the existing data from 'data.csv' and 'champion_proficiency.csv'
df = pd.read_csv('data.csv')
df2 = pd.read_csv('champion_proficiency.csv')

# Extract user names for players in the Support (position 4) and ADC (position 5) roles
name_list = df[df['position'].isin([4, 5])]['user_name'].tolist()

# Remove duplicate user names
name_list = list(set(name_list))

success_count = 0
fail_count = 0
continue_count = 0

# Iterate through each user name
for i, name in enumerate(name_list):
    # Check if the user name already exists in 'champion_proficiency.csv'
    if name in df2['user_name'].values:
        print(f"{name} -> continue")
        continue_count += 1
    else:
        # Fetch champion proficiency data for the user
        championsDataDict = getChampionsData(name)

        # Check if the fetched data is not empty
        if championsDataDict != []:
            # Convert the data to a DataFrame and append it to 'champion_proficiency.csv'
            df = pd.DataFrame(championsDataDict)
            df.to_csv('champion_proficiency.csv', index=False, mode='a', header=False)

            print(f"{name} -> success")
            success_count += 1
        else:
            print(f"{name} -> fail")
            fail_count += 1

    # Print progress information
    print(f"percentage : {i+1}/{len(name_list)}, Success : {success_count}, Fail : {fail_count}, Continue : {continue_count}")

