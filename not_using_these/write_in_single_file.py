import os


output_file = open("output_movies.txt", "w")
file =[]
for file_path in os.listdir("data/"):
    file = open("data/" + file_path, "r")
    for line in file:
        file.append(line+"\n")
    

    