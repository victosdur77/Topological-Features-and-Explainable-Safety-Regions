import numpy as np
from gudhi import RipsComplex
from gudhi import AlphaComplex
from gudhi.representations import DiagramSelector
import gudhi as gd
from scipy.spatial import distance_matrix
import math
from scipy import sparse
from scipy.stats import pearsonr

## calculating persistence features and diagram
def ComputePersistenceDiagram(ps,moment,dimension,maximumFiltr,complex="alpha",robotsSelected="all"):
    # maximumFiltration = [float(np.max(distance_matrix(X,X))) for X in ps[:,:,:2]]
    if robotsSelected == "all":
        points=ps[moment,:,:2]
    else:
        points=ps[moment,robotsSelected,:2]
    if complex not in ["rips","alpha"]:
        raise ValueError("The selected complex must be rips or alpha")
    elif complex=="alpha":
        alpha_complex = AlphaComplex(points=points) # 0ption 1: Using alpha complex
        simplex_tree = alpha_complex.create_simplex_tree(max_alpha_square=maximumFiltr)
    else:
        rips_complex = RipsComplex(points=points,max_edge_length=maximumFiltr) # Option 2: Using Vietoris-Rips complex
        simplex_tree = rips_complex.create_simplex_tree()
    persistence_features = simplex_tree.persistence()
    persistence = simplex_tree.persistence_intervals_in_dimension(dimension)
    return persistence

## removing infinity bars or limiting this bars
def limitingDiagram(Diagram,maximumFiltr,remove=False):
    if remove is False:
        infinity_mask = np.isinf(Diagram) #Option 1:  Change infinity by a fixed value
        Diagram[infinity_mask] = maximumFiltr 
    elif remove is True:
        Diagram = DiagramSelector(use=True).fit_transform([Diagram])[0] #Option 2: Remove infinity bars
    return Diagram

## calculating entropy
def EntropyCalculationFromBarcode(persistentBarcode):
    l=[]
    for i in persistentBarcode:
        l.append(i[1]-i[0])
    L = sum(l)
    p=l/L
    entropy=-np.sum(p*np.log(p))
    return round(entropy,4)