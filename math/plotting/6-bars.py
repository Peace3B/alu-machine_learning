
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

# my codes
people = ['Farrah', 'Fred', 'Felicia']
fname_color = {
    'apples' : 'red',
    'bananas' : 'yellow',
    'oranges' : '#ff8000',
    'peaches' : '#ffe5b4'
}

i = 0
for fname, color in sorted(fname_color.items()):
    bottom = 0
    for j in range(i):
        bottom += fruit[j]
    plt.bar(np.arange(len(people)), fruit[i], width=0.5, bottom=bottom, color=color, label=fname)
    i += 1
plt.xticks(np.arange(len(people)), people)
plt.yticks(np.arange(0, 81, 10))
plt.ylabel('Quantity of Fruit')
plt.title("Number of Fruit per Person")
plt.legend()
plt.show()
