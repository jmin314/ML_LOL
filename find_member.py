import re
import os

# Directory path
directory_path = "./member"

# Traverse directory and read file contents
for root, directories, files in os.walk(directory_path):
    for filename in files:
        file_path = os.path.join(root, filename)
        
        # Read file contents
        with open(file_path, 'r') as file:
            content = file.read()

        # Use regular expression to find all "internal_name" values
        matches = re.findall(r'"name":"(.*?)".*?"tier":"(.*?)"', content, re.DOTALL)

        # Retrieve values for specific tiers
        if matches:
            for match in matches:
                if match[1] == "CHALLENGER" or match[1] == "GRANDMASTER":
                    print(match[0])
        else:
            print("Cannot find internal_name.")
