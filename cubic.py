import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
df = pd.read_csv('data1.csv')
y = np.array(df['Research'])
x = np.array(df['Profit'])

# Normalize the input variables
x = (x - np.mean(x)) / np.std(x)
y = (y - np.mean(y)) / np.std(y)

m = y.shape[0]

def gradient(w1_now, w2_now, w3_now, b_now, x, y, L):
    w_gradient1 = 0
    w_gradient2 = 0
    w_gradient3 = 0
    b_gradient = 0 
    for i in range(m):
        w_gradient1 += (x[i]**3) * ((w1_now * (x[i]**3)) + (w2_now * (x[i]**2)) + (w3_now * x[i]) + b_now - y[i]) 
        w_gradient2 += (x[i]**2) * ((w1_now * (x[i]**3)) + (w2_now * (x[i]**2)) + (w3_now * x[i]) + b_now - y[i]) 
        w_gradient3 += x[i] * ((w1_now * (x[i]**3)) + (w2_now * (x[i]**2)) + (w3_now * x[i]) + b_now - y[i])
        b_gradient += ((w1_now * (x[i]**3)) + (w2_now * (x[i]**2)) + (w3_now * x[i]) + b_now - y[i])
    
    w1_now = w1_now - (L * (1/m) * (w_gradient1))
    w2_now = w2_now - (L * (1/m) * (w_gradient2))
    w3_now = w3_now - (L * (1/m) * (w_gradient3))
    b_now = b_now - (L * (1/m) * (b_gradient)) 

    return w1_now, w2_now, w3_now, b_now

def cost_function(w1, w2, w3, b, x, y):
    cost = 0.0 
    for i in range(m):
        C = (((w1 * (x[i]**3)) + (w2 * (x[i]**2)) + (w3 * x[i]) + b) - y[i]) ** 2 
        cost += C
    cost = cost / (2*m) 
    return cost 

w1 = 0.0
w2 = 0.0
w3 = 0.0
b = 0.0 
L = 0.1
epoch = 1000

for i in range(epoch):
    w1, w2, w3, b = gradient(w1, w2, w3, b, x, y, L) 
    Cost = cost_function(w1, w2, w3, b, x, y) 
    print(f'The epoch is {i+1} and cost is {Cost}') 

# Plotting the original data and the regression curve
plt.scatter(x, y)
x_range = np.linspace(np.min(x), np.max(x), 100)
plt.plot(x_range, (w1 * (x_range**3) + (w2 * (x_range**2)) + (w3 * x_range) + b),color='Red')
plt.xlabel('Profit')
plt.ylabel('Research')
plt.show()