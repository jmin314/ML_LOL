import os

# Website to obtain nicknames
url = "https://www.op.gg/leaderboards/tier?page="

# Crawling nicknames of users on page 30
for index in range(1, 30 + 1):
    total_url = url + str(index)
    file_path = "./member/" + str(index)

    # Sending a request to the address
    os.system("curl " + total_url + " > " + file_path)

    with open(file_path, 'r') as file:
        content = file.read()
    
    # Trimming as needed
    data_index = content.find("\"data\"")

    if data_index != -1:
        content = content[data_index:]

    with open(file_path, 'w') as file:
        file.write(content)
