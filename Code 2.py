#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore")
import random

import numpy as np
def impliment_eba(ratings, importances, cutoffs):
    
    indices = np.random.permutation(len(importances))
    importance_shuffled = importances[indices]
    ratings = ratings[:,indices]
    cutoffs_shuffled = cutoffs[indices]
                                     
    ranks_importance = np.flip(np.argsort(importance_shuffled))
    
    ratings = ratings[:,ranks_importance]
    
    cutoffs_shuffled_v2 = cutoffs_shuffled[ranks_importance]
    
    threshold_failure = []
    
    for i in range(ratings.shape[0]):
        threshold_failure.append(np.where((ratings [i,:] >= cutoffs_shuffled_v2) == False))
    
    earliest_failure = []
    
    for i in range(len(threshold_failure)):
        
        
        if len(threshold_failure[i][0])>0:
            earliest_failure.append(threshold_failure[i][0][0])
        else:
            earliest_failure.append(len(importance)-1)
    
    
    products_to_choose = np.where(earliest_failure == max(earliest_failure),1,0)
    
    tied = []
    
    for i in range(len(products_to_choose)):
        
        if products_to_choose[i] == 1:
            tied.append(i+1)
        else:
            pass
        
    return random.choice(tied)


# In[ ]:




