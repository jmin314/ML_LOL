import re

with open("./record/EDGqx1457794367", 'r') as file:
    data = file.read()

pattern = r"'id': '(.*?)'"
matches = re.findall(pattern, data)

print(matches, len(matches))