import numpy as np
from matplotlib import pyplot as plt 

fit= np.load('./cartpole/a2c/rew_table_ac.npy')
plt.figure(figsize=(10, 6))  # You can change the width and height as needed
#print(fit)
plt.plot(np.arange(len(fit[:239])), fit[:239])
print(sum(fit[:239]))
#t = 0
#t_values=[]
#x_values=[]
#fit[240]=200
#print(fit[:240])
#for j in range(0,240,15):
#
#        x_valore = np.mean(fit[j:j+15])
#        t = t + x_valore
#        t_values.append(t)
#        x_values.append(x_valore)
#


## Creazione del grafico
#plt.plot(t_values, x_values)

plt.xlabel('t')
plt.ylabel('Average Rewards')
plt.show() # f


#plt.plot(np.arange(len(fit[:130].flatten())), fit[:130].flatten())
#
#fit[130:135]=200
#t = 0
#t_values=[]
#x_values=[]
#
#for j in range(0,135,15):
#
#        x_valore = np.mean(fit[j:j+15])
#        t = t + x_valore
#        t_values.append(t)
#        x_values.append(x_valore)
#
#
#plt.figure(figsize=(10, 6))  # You can change the width and height as needed
#