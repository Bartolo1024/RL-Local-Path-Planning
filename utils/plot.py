from matplotlib import pyplot as plt

with open('log.txt', 'r') as file:
    lines = file.readlines()
    lines = lines[:-1]
    lines = [int(l.replace('\n', '').split(' ')[-1]) for l in lines]
plt.plot(lines)
plt.show()
