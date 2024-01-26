import cartpole_ga as ga
import numpy as np
import time
import random 
import psutil
import tensorflow as tf

class Base_Ga():
  def __init__(self,num_gen,num_pop,sigma,t):
    self.num_gen=num_gen
    self.num_pop=num_pop
    self.sigma=sigma
    self.t=t
    self.tau_table=np.zeros((self.num_gen,self.num_pop),dtype=int)
    self.fit_table=np.zeros((self.num_gen,self.num_pop), dtype=int)
   
    self.best_g=0
    self.initialization()

  def initialization(self):
    print("Inizialization of the ga")

    for n in range(self.num_pop):
      tau=ga.gen_tau()
      self.tau_table[0,n]=tau
      np.save("cartpole/ga/tau_table.npy", self.tau_table)
      self.fit_table[0,n]= self.fitness(0,n)
    self.tau_table[:1,:],self.fit_table[0,:]=ga.sort(self.tau_table[:1,:],self.fit_table[0,:])
    np.save("cartpole/ga/tau_table", self.tau_table)
    np.save("cartpole/ga/fit_table.npy", self.fit_table)

    print("Generazione: ",0, "Fitness: ", self.fit_table[0,:])
    print("Saved tau_table")
    return 

  def fitness(self,g,n):
    f=0
    for _ in range(15):
      f+= ga.run_agent(g,n,self.sigma)
    return f/15

  def fitness_fun(self,g):
      print("Fitness calculation")
      fit=[]

      for n in range(1,self.num_pop):
        fit.append(self.fitness(g,n))
      self.fit_table[g,1:self.num_pop]=fit

      print("Fitness End")

  def choose_elite(self,g):
    self.fit_table[g,0]=self.fit_table[g-1,0]



  def selection_and_mutation(self,g):
    print("Selection and Mutation")
    lista=[]
    for n in range(1, self.num_pop):
      k=random.randint(0,self.t)
      parent=self.tau_table[:g+1,k].copy()
      tau=ga.gen_tau()
      parent[g]=tau
      lista.append(parent)
    lista = np.transpose(lista, (1, 0))
    self.tau_table[:g+1,1:self.tau_table.shape[1]]=lista
    np.save("cartpole/ga/tau_table.npy", self.tau_table)

  
 
  def base_ga(self):

    for g in range(1,self.num_gen):
        self.choose_elite(g)
        self.selection_and_mutation(g)
        self.fitness_fun(g)
        self.tau_table[:g+1,:],self.fit_table[g,:]=ga.sort(self.tau_table[:g+1,:],self.fit_table[g,:])

        np.save("cartpole/ga/tau_table.npy", self.tau_table)
        np.save("cartpole/ga/fit_table.npy", self.fit_table)

        print("Generazione: ",g, "Fitness: ", self.fit_table[g,:])

        if(self.fit_table[g,0]>=200): 
          self.best_g=g
          break

    return self.best_g

from memory_profiler import profile

@profile
def run_ga():
  num_gen=200
  num_pop=20
  sigma=0.05
  t=3


  g=Base_Ga(num_gen, num_pop,sigma,t).base_ga()

  with open('cartpole/ga/output__cartpole_ga.txt', 'w') as file:
      file.write(f"Nun_gen : {num_gen}\n")
      file.write(f"Nun_pop : {num_pop}\n")
      file.write(f"sigma : {sigma}\n")
      file.write(f"t : {t}\n")
      file.write(f"finish in : {g} generation")
  return 


if __name__ == "__main__":
  run_ga()

