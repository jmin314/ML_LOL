import os
import re
import csv
import time

# Directory path
directory_path = "./record"

def info(pattern, data):
    """
    Extracts information from the given data based on the provided pattern.

    Parameters:
    - pattern (str): Regular expression pattern for extracting information.
    - data (str): Data to search and extract information from.

    Returns:
    - list: Extracted values based on the pattern.
    """
    value_list = []
    preprocess_data = r"'" + pattern + "':\s+'(.*?)'"
    values = re.findall(preprocess_data, data)
    
    if values == []:
        preprocess_data = r"'" + pattern + "': (\d+)"
        values = re.findall(preprocess_data, data)
        for value in values:
            value_list.append(value)
    else:
        for value in values:
            value_list.append(value)

    if len(values) > 20 * 10:  # If more than 20 games with 10 players each, clean up the data
        return_values = []
        for i in range(0, 20):
            return_values += values[:10]  # Get 10 values
            values = values[11:]  # Discard the first 11 values
        return return_values
    else:
        return values

def edit_dict(dict):
    """
    Edits the given dictionary to create a list of dictionaries with organized information.

    Parameters:
    - dict (dict): Original dictionary with unorganized information.

    Returns:
    - list: List of dictionaries with organized information.
    """
    dict_list = []
    for _ in range(10 * 20):  # 10 players * 20 games
        dict_list.append({})

    for key in dict.keys():
        if len(dict[key]) == 20:  # If there are 20 pieces of information, assume it's for a single game
            for i in range(0, 20):
                for j in range(i * 10, i * 10 + 10):
                    dict_list[j][key] = dict[key][i]
        else:
            for i in range(0, 200):
                dict_list[i][key] = dict[key][i]

    return dict_list

# Initialize a list to store dictionaries containing player information
info_dict_list = []
# Specify the filters for information extraction
filter_list = ['id', 'game_type', 'game_map', 'result', 'team_key', 'name', 'position', 'champion_id', 'kill', 'death', 'assist']

# Traverse the directory and read the contents of files
for root, directories, files in os.walk(directory_path):
    for filename in files:
        print(filename)
        # Initialize dictionary for storing information
        info_dict = {}

        file_path = os.path.join(root, filename)

        # Read the content of the file
        with open(file_path, 'r') as file:
            json_data = file.read()

            # Extract information based on specified filters
            for filter in filter_list:
                info_dict[filter] = info(filter, json_data)
        
        dict_list = edit_dict(info_dict)
        break

# Post-processing
# Filter entries with 'game_map' as 'SUMMONERS_RIFT' and 'game_type' as 'SOLORANKED'
filtered_dict_list = [entry for entry in dict_list if entry['game_map'] == 'SUMMONERS_RIFT' and entry['game_type'] == 'SOLORANKED']

# Calculate KDA (Kill/Death/Assist ratio) for each entry
for entry in filtered_dict_list:
    kill_value = int(entry.get('kill'))
    death_value = int(entry.get('death'))
    assist_value = int(entry.get('assist'))

    if death_value == 0:
        entry['kda'] = (kill_value + assist_value)
    else:
        entry['kda'] = round((kill_value + assist_value) / death_value, 2)  # Calculate KDA

# Shorten player IDs
used_id_list = []
used_id_count = 0

for entry in filtered_dict_list:
    if entry['id'] not in used_id_list:
        used_id_list.append(entry['id'])
        used_id_count += 1
        entry['id'] = used_id_count
    else:
        entry['id'] = used_id_count

# Calculate kill influence ratio
result_dict = {}
for item in filtered_dict_list:
    id_key = item['id']
    team_key = item['team_key']
    kill = int(item['kill'])

    unique_key = (id_key, team_key)

    if unique_key in result_dict:
        result_dict[unique_key] += kill
    else:
        result_dict[unique_key] = kill

sum_kill_dict = [{'id': key[0], 'team_key': key[1], 'sum_kill': value} for key, value in result_dict.items()]

for item in filtered_dict_list:
    kill_value = int(entry.get('kill'))
    assist_value = int(entry.get('assist'))

    for item2 in sum_kill_dict:
        if item['id'] == item2['id'] and item['team_key'] == item2['team_key']:
            sum_kill = item2['sum_kill']
            break
    
    item['kill_influence'] = round(sum_kill / (kill_value + assist_value), 2)

# Remove and replace unnecessary information
for item in filtered_dict_list:
    del item['game_type']
    del item['game_map']
    del item['kill']
    del item['death']
    del item['assist']

    if item['result'] == 'WIN':
        item['result'] = 1
    else:
        item['result'] = 0

    if item['team_key'] == 'BLUE':
        item['team_key'] = 1
    else:
        item['team_key'] = 0

    if item['position'] == 'TOP':
        item['position'] = 1
    elif item['position'] == 'JINGLE':
        item['position'] = 1
    elif item['position'] == 'MID':
        item['position'] = 1
    elif item['position'] == 'ADC':
        item['position'] = 1
    elif item['position'] == 'SUPPORT':
        item['position'] = 1

# Print the final list of dictionaries
print(filtered_dict_list)

# CSV file path
csv_file_path = 'data.csv'

# Open the CSV file in write mode and write the data
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = filtered_dict_list[0].keys()  # Use the keys of the first dictionary as field names
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()  # Write the header row (field names)
    
    for row in filtered_dict_list:
        writer.writerow(row)
