#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import rdkit
from rdkit import Chem


# In[ ]:


smiles = np.loadtxt('Predict_Smiles.csv', dtype=str, comments='!')
s_l = []
for smi in smiles:
    s_l.append(Chem.MolToSmiles(Chem.MolFromSmiles(smi)))
s_l = np.array(s_l).reshape(len(s_l), 1)
np.savetxt('Predict888_Smiles_24.csv', s_l, fmt='%s')


# In[ ]:


t = np.loadtxt('Title_137.csv', dtype=str, comments='!')
m_out = np.zeros((24, 137))
for i in range(s_l.shape[0]):
    smi = s_l[i, 0]
    m = Chem.MolFromSmiles(smi)
    for j in range(137):
        f = t[j, ]
        patt = Chem.MolFromSmarts(f)
        m_out[i, j] = len(m.GetSubstructMatches(patt))
np.savetxt('Predict888_24_137.csv', m_out, fmt='%d', delimiter=',')


# In[ ]:


from syba.syba import SybaClassifier
syba = SybaClassifier()
syba.fitDefaultScore()
syba_out = np.zeros((24, 1))
for i in range(s_l.shape[0]):
    syba_out[i, 0] = syba.predict(s_l[i, 0])
np.savetxt('Predict888_SYBA_24.csv', syba_out, fmt='%f', delimiter=',')

