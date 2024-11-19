#!/usr/bin/env python3
from navground import sim
import h5py
import numpy as np
import yaml
import itertools as itr
import os
from os.path import exists
import shutil
import pandas as pd
from scipy.stats import iqr
from utilsTopological import *
from scipy.spatial import distance_matrix
from tqdm import tqdm

CONFIG_FILE = 'configTopological.yaml'
RESULTS_FILE = 'simulationVictorTopological/dataset.csv'

def obj_to_yaml(obj): return yaml.dump(obj)

def yaml_load(filename):
	r = ''
	with open(filename, 'r') as f:
		r = yaml.safe_load(f)
	return r

configYAML = yaml_load(CONFIG_FILE)
print(configYAML)
n_runs = configYAML['runs']

exp = sim.load_experiment(obj_to_yaml(configYAML))
exp.run(number_of_threads=12)
print("Duration: ", exp.duration)
print("Experiment path: ", exp.path)


# data = h5py.File(exp.path)
# print(data.items())
# 	#analyse each run: average?

collisions = []
sms = []
etalist=[]
taulist = []
meanEntropy=[]
medianEntropy=[]
stdsEntropy=[]
iqrsEntropy=[]

for i, run in tqdm(exp.runs.items(), desc="Procesando runs"): # in data.items()
    world = run.world # sim.load_world(g.attrs['world'])

    sm = np.unique([agent.behavior.safety_margin for agent in world.agents])
    eta = np.unique([agent.behavior.eta for agent in world.agents])
    #print([agent.behavior.eta for agent in world.agents])
    #print(eta)
    etalist+=list(eta)
    tau = np.unique([agent.behavior.tau for agent in world.agents])
    taulist+=list(tau)
    assert len(sm) == 1
    sms += list(sm)
    collisions.append(len(run.collisions))

    ps=run.poses
    maxd = [float(np.max(distance_matrix(X,X))) for X in ps[:,:,:2]]
    entropies=[]
    for j in range(ps.shape[0]):
        persistence = ComputePersistenceDiagram(ps,j,0,maxd[j],"rips")
        persistenceL = limitingDiagram(persistence,maxd[j])
        entropies.append(EntropyCalculationFromBarcode(persistenceL))
    entropies = np.array(entropies)
    meanEntropy.append(entropies.mean())
    medianEntropy.append(np.median(entropies))
    stdsEntropy.append(entropies.std())
    iqrsEntropy.append(iqr(entropies))

print(len(sms),len(etalist),len(taulist),len(collisions),len(meanEntropy),len(medianEntropy),len(stdsEntropy),len(iqrsEntropy))
dataset = pd.DataFrame(zip(sms,etalist,taulist,collisions,meanEntropy,medianEntropy,stdsEntropy,iqrsEntropy),columns = ["SafetyMargin","Eta","Tau","NumberOfCollisions","meanEntropy","medianEntropy","stdsEntropy","iqrsEntropy"])
print(dataset.head())
dataset.to_csv(RESULTS_FILE,index = False)