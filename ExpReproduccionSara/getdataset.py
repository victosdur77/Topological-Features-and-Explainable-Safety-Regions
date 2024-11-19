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
from tqdm import tqdm

CONFIG_FILE = 'config.yaml'
RESULTS_FILE = 'simulationVictor/dataset.csv'

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
for i, run in tqdm(exp.runs.items()): # in data.items()
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

print(len(sms),len(etalist),len(taulist),len(collisions))
dataset = pd.DataFrame(zip(sms,etalist,taulist,collisions),columns = ["SafetyMargin","Eta","Tau","NumberOfCollisions"])
print(dataset)
dataset.to_csv(RESULTS_FILE,index = False)