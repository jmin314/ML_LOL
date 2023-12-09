import json
import os
import requests

url = "https://www.op.gg/summoners/kr/"

# Open a file containing user names
with open("./member_name") as file:
    for name in file:
        name = name.strip()  # Remove leading/trailing whitespaces
        
        # Check if a record for this user already exists
        if os.path.exists('./record/' + name):
            continue
        
        # Encode the name to include in the URL
        encode_name = name.replace(" ", "%20")
        total_url = url + encode_name

        # Make an HTTP GET request to fetch user data
        response = requests.get(total_url)
        
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Find the starting index of the game data in the response
            data_start_index = response.text.find('"games":{"data"')
            if data_start_index != -1:
                # Trim the response to get relevant game data
                trimmed_response = "{" + response.text[data_start_index:]
                
                try:
                    # Load trimmed JSON data
                    json_data = json.loads(trimmed_response)
                    
                    # Write JSON data to a file in the 'record' directory
                    with open("./record/" + name, 'w') as file2:
                        file2.write(str(json_data))
                        
                except json.JSONDecodeError as e:
                    # Handle JSON decoding errors
                    print(f"Error decoding JSON for {name}: {e}")
            else:
                # Handle cases where game data is not found
                print(f"Could not find game data for {name}")
        else:
            # Handle failed HTTP requests
            print(f"Failed to fetch data for {name}. Status code: {response.status_code}")
