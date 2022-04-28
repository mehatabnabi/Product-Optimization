#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import copy
import warnings
warnings.filterwarnings("ignore")


prod = pd.read_csv('lexic_order.csv', header=None)
prod.columns = ['prod_id','price','time_insulated','capacity','cleaning','containment']
prod.head()

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)

categorical = pd.DataFrame(encoder.fit_transform(prod[['price','time_insulated','capacity','cleaning','containment']]), columns=encoder.get_feature_names(['price','time_insulated','capacity','cleaning','containment']))
categorical.head()

prod_id = pd.DataFrame(prod['prod_id'])
prod_id.columns = ['prod_id']

prod_new = pd.concat([prod_id,categorical], axis=1)


prod_new.loc[:,'Brand_A'] = 0
prod_new.loc[:,'Brand_B'] = 0
prod_new.loc[:,'Brand_C'] = 1

prod_new.loc[243,:] = [244, 0,1,0, 0,0,1, 0,1,0, 0,1,0, 1,0,0, 1,0,0]
prod_new.loc[244,:] = [245, 1,0,0, 0,1,0, 0,1,0, 0,0,1, 0,0,1, 0,1,0]

prod_new.tail()

prod_array = np.array(prod_new)


preference = pd.read_csv('mugs-preference-parameters-full.csv')

preference = preference[['Cust',  'pPr10', 'pPr30', 'pPr05', 'pIn0.5', 'pIn1', 'pIn3', 'pCp12',
       'pCp20', 'pCp32','pClD', 'pClE', 'pClF', 'pCnLk', 'pCnSl', 'pCnSp', 
                        'pBrA', 'pBrB', 'pBrC', 'IPr', 'Iin', 'ICp', 'ICl', 'Icn', 'IBr']]

preference.tail()

preference_array = np.array(preference)

utility_price_0 = np.dot(preference_array[:,1:4], np.transpose(prod_array[:,1:4]))
utility_price = np.multiply(utility_price_0, preference_array[:,19].reshape(311,1))

utility_in_0 = np.dot(preference_array[:,4:7], np.transpose(prod_array[:,4:7]))
utility_in = np.multiply(utility_in_0, preference_array[:,20].reshape(311,1))

utility_cp_0 = np.dot(preference_array[:,7:10], np.transpose(prod_array[:,7:10]))
utility_cp = np.multiply(utility_cp_0, preference_array[:,21].reshape(311,1))

utility_cl_0 = np.dot(preference_array[:,10:13], np.transpose(prod_array[:,10:13]))
utility_cl = np.multiply(utility_cl_0, preference_array[:,22].reshape(311,1))

utility_cn_0 = np.dot(preference_array[:,13:16], np.transpose(prod_array[:,13:16]))
utility_cn = np.multiply(utility_cn_0, preference_array[:,23].reshape(311,1))

utility_br_0 = np.dot(preference_array[:,16:19], np.transpose(prod_array[:,16:19]))
utility_br = np.multiply(utility_br_0, preference_array[:,24].reshape(311,1))

utility = utility_price + utility_in + utility_cp + utility_cl + utility_cn + utility_br
utility = utility * 0.0139
utility = np.exp(utility)

utility_temp = copy.deepcopy(utility)
incumbent_utility = utility[:,243] + utility[:,244]

utility_temp = np.add(utility_temp, incumbent_utility.reshape(311,1))

prod_share = np.divide(utility, utility_temp)

prod_share = np.mean(prod_share, axis=0)
prod_share = prod_share.reshape(245,1)

prod['market_share'] = prod_share[0:243]

price_mapping = {' $30':30, ' $10':10, ' $5':5}
prod['price ($)'] = prod['price'].map(price_mapping)

in_mapping = {' 0.5 hrs':0.5, ' 1 hrs':1, ' 3 hrs':3}
prod['cost_in'] = prod['time_insulated'].map(in_mapping)

cap_mapping = {' 12 oz':1, ' 20 oz':2.6, ' 32 oz':2.8}
prod['cost_cap'] = prod['capacity'].map(cap_mapping)

cl_mapping = {' Difficult':1, ' Fair':2.2, ' Easy':3}
prod['cost_cl'] = prod['cleaning'].map(cl_mapping)

co_mapping = {' Slosh resistant':0.5, ' Spill resistant':0.8, ' Leak resistant':1}
prod['cost_co'] = prod['containment'].map(co_mapping)

prod['cost'] = prod['cost_in'] + prod['cost_cap'] + prod['cost_cl'] + prod['cost_co']

prod['margin'] = prod['price ($)'] - prod['cost']

prod['expected profit per person'] = prod['margin'] * prod['market_share']

prod.drop(['cost_in','cost_cap','cost_cl','cost_co'],axis=1, inplace=True)

print(f"Question 1")
print(f"Product Id : {prod[prod['prod_id'] == 45]['prod_id'].iloc[0]}")
print(f"Market Share : {prod[prod['prod_id'] == 45]['market_share'].iloc[0] : .3f}")
print(f"Price : {prod[prod['prod_id'] == 45]['price'].iloc[0]}")
print(f"Expected Profit per Person : {prod[prod['prod_id'] == 45]['expected profit per person'].iloc[0] : .3f}")


prod_final = prod[['prod_id','price','time_insulated','capacity','cleaning','containment','market_share','price ($)', 'margin', 'expected profit per person']]

print('---------------------------------------------------------------')

print(f"Question 2")
print(f"The product combinations with respective summaries is loaded into - 'product_combinations.csv'")

prod_final.to_csv('product_combinations.csv')
highest_profit = prod[prod['expected profit per person'] == max(prod['expected profit per person'])]
highest_share  = prod[prod['market_share'] == max(prod['market_share'])]
highest_margin = prod[prod['margin'] == max(prod['margin'])]
lowest_cost    = prod[prod['cost'] == min(prod['cost'])]
print('---------------------------------------------------------------')
print(f"Question 3")
print(f"Product Id : {highest_profit['prod_id'].iloc[0]}")
print(f"Price : {highest_profit['price'].iloc[0]}")
print(f"Time Insulated : {highest_profit['time_insulated'].iloc[0]}")
print(f"Capacity : {highest_profit['capacity'].iloc[0]}")
print(f"Cleaning : {highest_profit['cleaning'].iloc[0]}")
print(f"Containment : {highest_profit['containment'].iloc[0]}")
print(f"Market Share : {highest_profit['market_share'].iloc[0]}")
print(f"Margin : {highest_profit['margin'].iloc[0]}")
print(f"Expected profit per person : {highest_profit['expected profit per person'].iloc[0]}")
print('---------------------------------------------------------------')

print(f"Question 4.1")
print(f"Product Id : {highest_share['prod_id'].iloc[0]}")
print(f"Price : {highest_share['price'].iloc[0]}")
print(f"Time Insulated : {highest_share['time_insulated'].iloc[0]}")
print(f"Capacity : {highest_share['capacity'].iloc[0]}")
print(f"Cleaning : {highest_share['cleaning'].iloc[0]}")
print(f"Containment : {highest_share['containment'].iloc[0]}")
print(f"Market Share : {highest_share['market_share'].iloc[0]}")
print(f"Margin : {highest_share['margin'].iloc[0]}")
print(f"Expected profit per person : {highest_share['expected profit per person'].iloc[0]}")

print('\n')
print(f"Question 4.2")
print(f"Product Id : {highest_margin['prod_id'].iloc[0]}")
print(f"Price : {highest_margin['price'].iloc[0]}")
print(f"Time Insulated : {highest_margin['time_insulated'].iloc[0]}")
print(f"Capacity : {highest_margin['capacity'].iloc[0]}")
print(f"Cleaning : {highest_margin['cleaning'].iloc[0]}")
print(f"Containment : {highest_margin['containment'].iloc[0]}")
print(f"Market Share : {highest_margin['market_share'].iloc[0]}")
print(f"Margin : {highest_margin['margin'].iloc[0]}")
print(f"Expected profit per person : {highest_margin['expected profit per person'].iloc[0]}")

print('\n')
print(f"Question 4.3")
print(f"Product Id : {lowest_cost['prod_id'].iloc[0]}")
print(f"Price : {lowest_cost['price'].iloc[0]}")
print(f"Time Insulated : {lowest_cost['time_insulated'].iloc[0]}")
print(f"Capacity : {lowest_cost['capacity'].iloc[0]}")
print(f"Cleaning : {lowest_cost['cleaning'].iloc[0]}")
print(f"Containment : {lowest_cost['containment'].iloc[0]}")
print(f"Market Share : {lowest_cost['market_share'].iloc[0]}")
print(f"Margin : {lowest_cost['margin'].iloc[0]}")
print(f"Expected profit per person : {lowest_cost['expected profit per person'].iloc[0]}")


# In[ ]:




