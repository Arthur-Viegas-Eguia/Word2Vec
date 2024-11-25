data = None
with open('output_jimmy.txt', 'r') as f:
    data = f.readlines()

data = ''.join(data).split('f9w\n')[:-1]
print(len(data))

data.reverse()

data = '\n'.join(data)

while '  ' in data:
    data = data.replace('  ', ' ')

with open('modified_output_jimmy.txt', 'w') as f:
    f.write(data)