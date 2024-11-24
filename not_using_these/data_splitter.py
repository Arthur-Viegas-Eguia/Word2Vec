data = None

with open('output.txt') as f:
    data = f.readlines()

data = ''.join(data).split('f9w\n')[:-1]
data.reverse()
data = ' '.join(data).replace('\n', ' ')

while '  ' in data:
    data = data.replace('  ', ' ')

with open('modified_output.txt', 'w') as f:
    f.write(data)