#!/usr/bin/env python
# coding: utf-8

# In[169]:


import pandas as pd
import numpy as np


# In[170]:


mainpath='/Users/ALITO/Documents'


# In[171]:


data=pd.read_csv(mainpath+'/'+'Set/Transporte1.csv')


# In[172]:


data.head()


# In[173]:


data.corr()


# In[174]:


data.iloc[:,[0,1]].head()


# In[175]:


x=2.6 + 7 * data


# In[176]:


x.head()


# In[177]:


import matplotlib.pyplot as mpl


# In[178]:


get_ipython().run_line_magic('matplotlib', 'inline')
mpl.plot(data.corr())


# In[179]:


data=pd.read_csv(mainpath+'/'+'Set/Transporte-Estrato.csv')


# In[180]:


data.head()


# In[202]:


x=2.6 + 7 *data


# In[203]:


res = 0 + 0.9 *x


# In[204]:


y_act = 5 + 0.3 * x + res


# In[205]:


y_pred = 5 + 0.3 *x


# In[206]:


data['x']=x
data['y_prediccion']=y_pred
data['y_actual']=y_act


# In[207]:


data.head()


# In[208]:


data.corr()


# In[209]:


import matplotlib.pyplot as mpl


# In[210]:


y_mean = [np.mean(y_act) for i in range(1, len(x) + 1)]


# In[211]:


get_ipython().run_line_magic('matplotlib', 'inline')
mpl.plot(data["x"],data["y_prediccion"])
mpl.plot(data["x"], data["y_actual"], "ro")
mpl.title("Valor Actual vs Predicción")


# In[212]:


y_m = np.mean(y_act)
data["SSR"]=(data["y_prediccion"]-y_m)**2
data["SSD"]=(data["y_prediccion"]-data["y_actual"])**2
data["SST"]=(data["y_actual"]-y_m)**2


# In[213]:


data.head()


# In[214]:


data.corr()


# In[215]:


SSR = sum(data["SSR"])
SSD = sum(data["SSD"])
SST = sum(data["SST"])


# In[216]:


SSR


# In[217]:


SSD


# In[218]:


SST


# In[219]:


R2=(SSR/SST)*100


# In[220]:


R2


# In[221]:


x_mean = np.mean(data["x"])
y_mean = np.mean(data["y_actual"])
x_mean, y_mean


# In[222]:


data["beta_n"] = (data["x"]-x_mean)*(data["y_actual"]-y_mean)
data["beta_d"] = (data["x"]-x_mean)**2


# In[223]:


beta = sum(data["beta_n"])/sum(data["beta_d"])


# In[224]:


alpha = y_mean - beta * x_mean


# In[225]:


alpha, beta


# In[226]:


data["y_model"] = alpha + beta * data["x"]


# In[227]:


data.head()


# In[228]:


data.corr()


# In[229]:


SSR = sum((data["y_model"]-y_mean)**2)
SSD = sum((data["y_model"]-data["y_actual"])**2)
SST = sum((data["y_actual"]-y_mean)**2)


# In[230]:


SSR,SSD,SST


# In[231]:


R2=(SSR/SST)*100
R2


# In[235]:


y_mean = [np.mean(y_act) for i in range(1, len(x) + 1)]

get_ipython().run_line_magic('matplotlib', 'inline')
mpl.plot(data["x"], data["y_actual"], "ro")
mpl.plot(data["x"],y_mean, "g")
mpl.plot(data["x"], data["y_model"])
mpl.title("Valor Actual vs Predicción")


# In[ ]:




