# Quantum Computing Meets Higgs Boson
Summary of models and work done during my Seniour Honour's Project at the University of Edinburgh. During the project we aim to classify di-photon events as signal 
(events coming from a Higgs Boson) or background (those coming from other sources). We will do this via Dense Neural Networks (DNN) and later with Variational Quantum Classifiers (VQC) and compare them. For more detail, the Project Report is included in the repository.

## Monte-Carlo Dataset
In this project, we utilize data with di-photon events from the signal $H \rightarrow \gamma \gamma$ events
and background events with no Higgs boson. The signal processes were generated using
Powheg Box v2 and interfaced with Pythia 8.2. Background simulated
samples were generated using MadGraph5 aMC@NLO and interfaced with Pythia
8.2. ATLAS detector response for both signal and background events was simulated using
Geant4. The simulated data was produced by the ATLAS experiment and provided
by the project supervisor Dr. Liza Mijovic. 


## Feature Engineering
Data contains the transverse momenta $p_T$, energy $E$, pseudorapidity $\eta$, azimuthal angle $\phi$ of the 2 leading photons and 4 leading particle jets; and the total number of jets for a given event.
Using these features, we engineered new ones based on [arXiv:2207.00348 [hep-ex]](https://arxiv.org/abs/2207.00348). All the features used can be found in the project report.
To manipulate the data and make new features, we utilized heavily numpy arrays and pylorentz Momentum4 objects to represent each particle. 
The code used to engineer the features can be found in engineer.py.


