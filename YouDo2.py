#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from reco import UserBased
from rich.console import Console


# In[ ]:


cons = Console()

df = pd.read_csv('https://files.grouplens.org/datasets/movielens/ml-100k/u.data', delimiter=r'\t',
                 names=['user_id', 'item_id', 'rating', 'timestamp'])

r = df.pivot(index='user_id', columns='item_id', values='rating').values


# In[ ]:


irow, jcol = np.where(~np.isnan(r))


# In[ ]:


idx = np.random.choice(np.arange(100_000), 1000, replace=False)
test_irow = irow[idx]
test_jcol = jcol[idx]


# In[ ]:


for i in test_irow:
    for j in test_jcol:
        r_copy[i][j] = np.nan


# In[ ]:


user = UserBased(beta=3, idf=True)

sim = user.fit(r_copy)


# In[ ]:


def gdes(x,y,0.6, alpha=0.001):
    beta = np.random.random(2) 

    print("starting sgd")
    for i in range(100):

        y_pred: np.ndarray = beta[0] + beta[1] * x

        g_b0 = -2 * (y - y_pred).sum()
        g_b1 = -2 * (x * (y - y_pred)).sum()

        beta_prev = np.copy(beta)

        beta[0] = beta[0] - alpha * g_b0
        beta[1] = beta[1] - alpha * g_b1

        err = 0

        for _y, _y_pred in zip(y,y_pred):

            if abs(_y-_y_pred) < theta*_y:
                err += (_y-_y_pred)**2
            
            else:
                err += theta*_y

        print(f"({i}) beta: {beta}, gradient: {g_b0} {g_b1}, error: {err}")
        if np.linalg.norm(beta - beta_prev) < 0.001:
            print(f"I do early stoping at iteration {i}")
            break

    return beta


# In[ ]:


err = []
for u, j in zip(test_irow, test_jcol):
    y_pred = user.predict1(r_copy, u, j)
    y = r[u, j]

    err.append((y_pred - y) ** 2)
    


# In[ ]:


err = err/2

