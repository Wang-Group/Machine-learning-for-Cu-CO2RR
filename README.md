# Machine-learning-for-Cu-CO2RR
This is the Python code and original data of "Machine-Learning Guided Discovery and Optimization of Additives in Preparing Cu Catalyst for Selective Electrochemical CO2 Reduction" from XMU Wang-group.
The original data lies in the files ist,2nd and 3rd,respectively.Represented for the ML discovery and optimization procedure.
in the 2nd round, besides the functional group-based featurization method, we used molecular fragment featurization (MFF) to extract matrix of substructure of a molecule . This MFF method was modified from extended-connectivity fingerprint (ECFP) method in Deepchem(https://deepchem.io/) by skipping hash function calculation step to avoid information loss.Before we formally pubulish this method, this page must be cited if you use it. Named ECFP6.0 here.
LRIF means Random intersection tree part.find which combination is inportant first and generate new.
Other program are basic python code from scikit learn, nothing special.
The prediction of 24 molecules was done by the code named Smiles2RdkitSmiles.py

This project finally published at Guo, Y. et al. Machine learning part was finished by Yuming Su and Yiheng Dai. 
The infomation of this published paper is below:
Machine-Learning-Guided Discovery and Optimization of Additives in Preparing Cu Catalysts for CO2 Reduction. J. Am. Chem. Soc. 143, 5755-5762, doi:10.1021/jacs.1c00339 (2021).
If used, please cite.
https://pubs.acs.org/doi/10.1021/jacs.1c00339?ref=pdf
