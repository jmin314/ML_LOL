import os
import re
import csv

# Directory path
directory_path = "./record"

def info(pattern, data):
    if pattern not in ['kill', 'death', 'assist', 'champion_id'] : # String
        new_pattern = r"'" + pattern + "': '(.*?)'"
    else : # Number
        new_pattern = r"'" + pattern + "': (\d+)"
    
    result = re.findall(new_pattern, data)

    # Delete duplicate value
    if len(result) == 1 :
        result = result[0]
    elif len(result) == 11 :
        result = result[:-1]
    elif len(result) == 13 :
        result = result[:-3]
    return result

info_dict_list = []
success_file_count = 0
filter_list = ['id', 'game_type', 'game_map', 'result', 'team_key', 'summoner_id', 'tagline', 'game_name', 'position', 'champion_id', 'kill', 'death', 'assist']
filter_list = ['id', 'game_type', 'game_map', 'result', 'team_key', 'tagline', 'game_name', 'position', 'champion_id', 'kill', 'death', 'assist']
game_id_count = 1
game_id_list = []
file_count = 0

# Traverse the directory and read file contents
for root, directories, files in os.walk(directory_path):
    for filename in files:
        file_count += 1
        print(file_count, filename[:-1])

        file_path = os.path.join(root, filename)

        # Read file contents
        with open(file_path, 'r') as file:
            data = file.read()

        # Extracting data for each game
        pattern = r"\{'id': '.*?', 'memo':.*?\.png'\}\}\}"
        matches = re.findall(pattern, data)
        
        # Analyzing one game at a time
        for match in matches:
            info_dict = {}
            for filter in filter_list:
                info_dict[filter] = info(filter, match)
            
            # Skip if it's not the desired game type
            if info_dict['game_map'] == 'SUMMONERS_RIFT' and info_dict['game_type'] == 'SOLORANKED':
                
                if info_dict['id'] not in game_id_list:
                    game_id_list.append(info_dict['id'])
                    info_dict['id'] = [game_id_count]*10
                    game_id_count += 1

                    del info_dict['game_map']
                    del info_dict['game_type']

                    for i in range(0, 10):
                        # Filtering unnecessary information
                        new_info_dict = {}
                        for filter in filter_list:
                            if filter not in ['game_map', 'game_type', 'tagline', 'game_name']:
                                new_info_dict[filter] = info_dict[filter][i]
                            else:
                                continue
                        
                        # Combine username and tag
                        user_name = f"{info_dict['game_name'][i]}-{info_dict['tagline'][i]}"
                        new_info_dict['user_name'] = user_name
                        
                        info_dict_list.append(new_info_dict)

filtered_dict_list = info_dict_list.copy()

# Calculate KDA
for entry in filtered_dict_list:
    kill_value = int(entry.get('kill'))
    death_value = int(entry.get('death'))
    assist_value = int(entry.get('assist'))

    if death_value == 0:
        entry['kda'] = (kill_value + assist_value)
    else:
        entry['kda'] = round((kill_value + assist_value) / death_value, 2)

# Deletion and substitution
for item in filtered_dict_list:
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
    elif item['position'] == 'JUNGLE':
        item['position'] = 2
    elif item['position'] == 'MID':
        item['position'] = 3
    elif item['position'] == 'ADC':
        item['position'] = 4
    elif item['position'] == 'SUPPORT':
        item['position'] = 5

# CSV file path
csv_file_path = 'data.csv'

# Write data to the CSV file in write mode
with open(csv_file_path, 'w', newline='') as csvfile:
    fieldnames = filtered_dict_list[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for row in filtered_dict_list:
        writer.writerow(row)
