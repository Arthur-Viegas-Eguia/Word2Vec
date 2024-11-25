import re
def filter_data(string):
    #This re
    filter_links = r"\b(http|https):\/\/[^\s].[^\s]*\b"
    data_no_links = re.sub(filter_links, '', string)
    filter_names = r"<[\S]*>"
    data_no_names = re.sub(filter_names, '', data_no_links)
    data_no_commands = r"[!+-]\S*"
    data_no_commands = re.sub(data_no_commands, '', data_no_names)
    filter_no_punctuation = r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]'
    data_no_punctuation = re.sub(filter_no_punctuation, '', data_no_commands)
    #We got the ASCII character values from https://stackoverflow.com/questions/2124010/grep-regex-to-match-non-ascii-characters
    filter_non_ascii = r'[^\x00-\x7F]+'
    data_non_ascii = re.sub(filter_non_ascii, '', data_no_punctuation)
    filter_multiple_spaces = r'\s+'
    data_single_spaced = re.sub(filter_multiple_spaces, ' ', data_non_ascii).strip()
    return data_single_spaced

def stopword_eraser(sentence):
    new_sentence = []
    sentence = sentence.lower()
    sentence = sentence.split(" ")
    stopwords_file = open("stopwords.txt", "r")
    stopwords = set()
    for stopword in stopwords_file:
        stopwords.add(stopword[:-1])
    for word in sentence:
        if word in stopwords:
            continue
        else:
            new_sentence.append(word)
    new_sentence = " ".join(new_sentence)
    return new_sentence

file = open("modified_output_bryce.txt", "r")
output = open("answer.txt", "w")
for line in file:
    filtered_line = filter_data(line)
    if re.match(r'^\s*$', filtered_line) == None:
        output.write(filtered_line+"\n")
    output.write("\n")
file.close()
file2 = open("modified_output_jimmy.txt", "r")
for line in file2:
    filtered_line = filter_data(line)
    line_no_names = stopword_eraser(filtered_line)
    if re.match(r'^\s*$', line_no_names) == None:
        output.write(filtered_line+"\n")
    output.write("\n")

file_to_remove = open("input_movies.txt", "r")
file_to_write_in = open("clean_movies.txt", "w")
i = 0
for line in file_to_remove:
    file_to_write_in.write(stopword_eraser(line))

    



