import numpy as np
from matplotlib import pyplot as plt 

fit= np.load('./cartpole/dqn/rew_table_dqn.npy')
print(fit[:548 ])
plt.plot(np.arange(len(fit[:548].flatten())), fit[:548].flatten())
print(sum(fit[:548]))

#fit[548:555]=200
#plt.figure(figsize=(10, 6))  # You can change the width and height as needed
#
##plt.plot(np.arange(len(fit)), fit)
#t = 0
#t_values=[]
#x_values=[]
#
#for j in range(0,555,15):
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
