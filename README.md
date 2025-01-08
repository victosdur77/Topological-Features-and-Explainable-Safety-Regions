# ExplainableSafetyRegions

This repository contains data and experiments associated to the paper Toscano-Durán, V. Gonzalez-Diaz, R. Narteni, S. Mongelli, M. (2025) "Safe Social Navigation through Explainable Safety Regions Based on Topological Features". Submited to The 3rd World Conference on eXplainable Artificial Intelligence (XAI-2025).

The paper deals with simulated social robotics navigation problem that involves a fleet of mobile agents moving in a Cross scenario, governed by a human-
like behavior. With the purpose of avoiding negative events, as collisions or deadlocks, we show how to topological features can improve the accuracy and effectiveness of safe and explainable AI (XAI) methods being an useful tool to know and adjust whether a simulation will be safe or not (free of collisions and deadlocks).

# Repository structure
- ExpReproduccionSara: Reproduce the experiments of https://doi.org/10.1007/978-3-031-63803-9_22 in order to know how to calculate safety regions.

- ExpTopologicalCollision: Experiments for avoid collisions, using safety regions and topological features.

- ExpTopologicalDeadlock: Experiments for avoid deadlocks, using safety regions and topological features.

- ExpTopologicalSafeNavigation: Experiments for avoid both collisions and deadlocks, using safety regions and topological features.

- ExpTopologicalAdvanced: Extension experiments using more topological features for build safety regions (not included in the paper).

# Usage

1) Clone this repository and create a virtual enviromment:
```bash
python3 -m venv entorno python=3.11
```
2) Install the necessary dependencies: pip install jupyter notebook
```bash
pip install navground[all] pandas==2.2.3 seaborn==0.13.2 scikit-learn==1.3.0 skope-rules==1.0.1 numpy==1.25.1 qpsolvers[open_source_solvers] cvxopt anchor-exp
```
3) **Simulation and dataset collection (including simulations and topological features)**: run the `getdataset_TopologicalFeatures.py` script with the YAML settings contained in `configTopological.yaml` file. Dataset used in further experiments.

4) **Native rule generation**: run `SkopeRules.ipynb` for training skope-rules model, and `NativeXAI_performance.ipynb` for its evaluation.

5) **Scalable Classifiers for Probabilistic/Conformal safety regions**: run `ConfidenceRegions_SVM.ipynb`.

6) **Local Rule Extraction from PSR/CSR**: run `Anchor_PSR.ipynb`, `Anchor_CSR.ipynb` for Anchors extraction, and `AnchorAnalysis_PSR.ipynb`, `AnchorAnalysis_CSR.ipynb`for their evaluation.

# Citation and reference

If you want to use our code for your experiments, please cite our paper.

For further information, please contact us at: vtoscano@us.es, sara.narteni@cnr.it

