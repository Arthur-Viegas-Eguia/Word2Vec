data = None
with open('answer.txt') as f:
    data = f.readline()
while '  ' in data:
    data = data.replace('  ', ' ')
with open('fixed.txt', 'w') as f:
    f.write(data)