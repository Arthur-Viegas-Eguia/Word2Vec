import re
def filter_data(string):
    filter_links = r"\b(http|https):\/\/[^\s/$.?#].[^\s]*\b"
    data_no_links = re.sub(filter_links, '', string)
    filter_names = r"<[\S]*>"
    data_no_names = re.sub(filter_names, '', data_no_links)
    data_no_commands = r"[!+-]\S*"
    data_no_commands = re.sub(data_no_commands, '', data_no_names)
    filter_no_punctuation = r'[!"#$%&\'()*+,-./;<=>?@[\]^_`{|}~]'
    data_no_punctuation = re.sub(filter_no_punctuation, '', data_no_commands)
    filter_non_ascii = r'[^\x00-\x7F]+'
    data_non_ascii = re.sub(filter_non_ascii, '', data_no_punctuation)
    filter_non_emoji = r'(?<!\:)[^a-zA-Z0-9](?=\:)'
    data_no_colon = re.sub(filter_non_emoji, '', data_non_ascii)
    return data_no_colon
    


file = open("output.txt", "r")
output = open("answer.txt", "w")
for line in file:
    filtered_line = filter_data(line)
    if re.match(r'^\s*$', filtered_line) == None:
        output.write(filtered_line)
file.close()
file2 = open("output_2.txt", "r")
for line in file2:
    filtered_line = filter_data(line)
    if re.match(r'^\s*$', filtered_line) == None:
        output.write(filtered_line)

