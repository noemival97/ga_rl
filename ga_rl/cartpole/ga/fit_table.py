import numpy as np
from matplotlib import pyplot as plt 

fit= np.load('ga_copy/fit_table.npy')
# Creazione del grafico
plt.figure(figsize=(10, 6))  # You can change the width and height as needed
plt.plot(np.arange(len(fit[:23].flatten())), fit[:23].flatten())
print(sum(fit[:23]))

print(sum(sum(fit[:23])))
#t = 0
#t_values=[]
#x_values=[]
#
#for j in range(0,len(fit),15):
#
#        x_valore = np.mean(fit[j:j+15])
#        t = t + x_valore
#        t_values.append(t)
#        x_values.append(x_valore)
#
#
#
### Creazione del grafico
#plt.plot(t_values, x_values)

plt.xlabel('t')
plt.ylabel('Average Rewards')
plt.show() # f

