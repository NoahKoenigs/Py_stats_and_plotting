#!/usr/bin/env python
# coding: utf-8

# # Intro to stats in python

# ### For a tutorial on these stats visit:
# https://scipy-lectures.org/packages/statistics/index.html

# In[51]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.plotting as pd_plotting
import scipy
from scipy import stats
import seaborn
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import os
import urllib


# ## Import .csv file

# In[2]:


data = pd.read_csv("brain_size.csv", sep=';' , na_values = ".")
print(data)
data1 = pd.read_csv("iris.csv", sep=',' , na_values = ".")


# ### Numpy Array

# In[3]:


t = np.linspace(-6, 6,20)
sin_t = np.sin(t)
cos_t = np.cos(t)


# ### Numpy Array Exposed in Pandas

# In[4]:


pd.DataFrame({'t': t, 'sin': sin_t, "cos" : cos_t})


# ## Manuipulating data

# #### It has 40 rows and 8 columns

# In[5]:


data.shape


# In[6]:


data.columns 
data.columns = pd.Index([u'Unnamed: 0', u'Gender', u'FSIQ', u'VIQ', u'PIQ', u'Weight', u'Height', u'MRI_Count'], dtype='object')
print(data.columns)


# #### Columns can be shown by name

# In[7]:


print(data['Gender'])


# # Simpler selector

# In[8]:


data[data['Gender'] == 'Female'] ['VIQ'].mean()


# In[9]:


groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))


# In[10]:


groupby_gender.mean()


# ## Making Box and Whisker Plots

# In[11]:


plt.figure(figsize=(4, 3))
data.boxplot(column= ['FSIQ', 'PIQ'])


# #### Boxplotting differences

# In[12]:


plt.figure(figsize=(4, 3))
plt.boxplot(data['FSIQ'] - data['PIQ'])
plt.xticks((1, ), ('FSIQ-PIQ',))
plt.show()


# # Plotting data

# In[13]:


pd.plotting.scatter_matrix(data[['Weight','Height', 'MRI_Count']])
plt.show()


# In[14]:


pd.plotting.scatter_matrix(data[['VIQ','PIQ', 'FSIQ']])
plt.show()


# In[15]:


pd.plotting.scatter_matrix(data[['PIQ','VIQ','FSIQ']])
plt.show()


# ### Boxplots of columns by gender

# In[41]:


groupby_gender = data.groupby('Gender')
groupby_gender.boxplot(column=['FSIQ', 'VIQ', 'PIQ'])


# # Hypothesis Testing and Comparing Two Groups

# In[16]:


stats.ttest_1samp(data['VIQ'], 0)


# In[17]:


female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)


# In[18]:


stats.ttest_ind(data['FSIQ'], data['PIQ'])


# In[19]:


stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0)


# ## Wilcoxon signed-rank test

# In[20]:


stats.wilcoxon(data['FSIQ'], data ['PIQ'])


# ## Linear Models

# #### Generate simulated data according to the model

# In[21]:


x = np.linspace(-5, 5, 20)
np.random.seed(1)
# normal distributed noise
y = -5 +3*x +4 * np.random.normal (size=x.shape)
# Create a data frame conatining all relavent variables
data = pd.DataFrame({'x' : x, 'y': y})


# ##### Specify OLS model and fit it to the graph

# In[22]:


from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit()


# ##### Inspect Stats derived from the OLS model

# In[23]:


from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit()
print(model.summary())


# ### Ex. Copmarison of male and female IQ based on brain size

# In[24]:


## rerun data definitions to make work
import pandas as pd
data = pd.read_csv ("brain_size.csv",sep=';',na_values='.')
data.columns 
data.columns = pd.Index([u'Unnamed: 0', u'Gender', u'FSIQ', u'VIQ', u'PIQ', u'Weight', u'Height', u'MRI_Count'], dtype='object')
model = ols("VIQ ~ Gender", data).fit()
print(model.summary())


# ### Link to t-tests between different FSIQ and PIQ

# In[25]:


data_fisq = pd.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pd.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pd.concat((data_fisq, data_piq))
print(data_long)


# In[26]:


[34,35,]
model = ols ("iq ~ type", data_long).fit()
print(model.summary())
stats.ttest_ind(data['FSIQ'], data['PIQ'])


# ## Multiple Regression

# ## Opening iris.csv from import

# In[27]:


print(data1)
data1.shape #150 rows and 1? column
data1.columns = pd.Index([u'sepal_length', u'sepal_width', u'petal_length', u'petal_width', u'name'], dtype='object')
print(data1.shape)
print(data1.columns)


# In[28]:


model = ols('sepal_width ~ petal_length', data1).fit()
print(model.summary())


# ## Analysis of petal sizes

# In[49]:


categories = pd.Categorical(data1['name'])
pd.plotting.scatter_matrix(data1, c=categories.codes, marker='o')
fig = plt.gcf()
fig.suptitle("blue: setosa, green: versicolor, red: virginica", size=13)


# # ANOVA
# ### Post-hoc hypothesis testing: analysis of varience

# ##### Write a vector of contrast

# In[29]:


test_input = [0, 1, -1, 0]
test_input_array = sm.add_constant(test_input)
result = model.f_test(test_input_array)
print (result)


# # More Visualization Using Seaborn

# ### Importing "wages.txt" from the web

# In[30]:


engine ='python'
if not os.path.exists('wages.txt'):
    urllib.request.urlretrieve('http://lib.stat.cmu.edu/datasets/CPS_85_Wages', 'wages.txt')
names = ['EDUCATION: Number of years of education','SOUTH: 1=Person lives in South, 0=Person lives elsewhere','SEX: 1=Female, 0=Male','EXPERIENCE: Number of years of work experience','UNION: 1=Union member, 0=Not union member','WAGE: Wage (dollars per hour)','AGE: years','RACE: 1=Other, 2=Hispanic, 3=White','OCCUPATION: 1=Management, 2=Sales, 3=Clerical, 4=Service, 5=Professional, 6=Other','SECTOR: 0=Other, 1=Manufacturing, 2=Construction','MARR: 0=Unmarried,  1=Married']
short_names = [n.split(':')[0] for n in names]
data3 = pd.read_csv('wages.txt', skiprows=27, skipfooter=6, sep=None, header=None, names=short_names)
data3.columns =short_names


# ### mulplicative factors

# In[31]:


data3['WAGE'] = np.log10(data3['WAGE'])


# In[32]:


seaborn.pairplot(data3, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg')
seaborn.pairplot(data3, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg', hue='SEX')
plt.suptitle('Effect of gender: 1=Female, 0=Male')
seaborn.pairplot(data3, vars=['WAGE','AGE','EDUCATION'], kind='reg', hue='RACE')
plt.suptitle('Effect of race: 1=Other, 2=Hispanic, 3=White')
seaborn.pairplot(data3, vars=['WAGE','AGE', 'EDUCATION'], kind='reg', hue='UNION')
plt.suptitle('Effect of union: 1=Union member, 0=Not union member')


# ### Plotting a simple regression

# In[33]:


seaborn.lmplot(y='WAGE', x='EDUCATION', data=data3)
plt.show()


# ## Viewing wages.txt

# In[34]:


print(data3)


# In[35]:


seaborn.pairplot(data3, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg')


# In[36]:


seaborn.pairplot(data3, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg', hue='SEX')


# ## lmplot for plotting a univariate regression

# In[37]:


seaborn.lmplot(y='WAGE', x='EDUCATION', data=data3)


# ### Correlation testing

# In[38]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
result = ols(formula='WAGE ~ EDUCATION + GENDER - EDUCATION * GENDER', data=data3).fit()
print(result.summary())


# ## Correlation Regression

# In[50]:


engine='python'
if not os.path.exists('wages.txt'):
    urllib.request.urlretrieve('http://lib.stat.cmu.edu/datasets/CPS_85_Wages', 'wages.txt')
names = ['EDUCATION: Number of years of education','SOUTH: 1=Person lives in South, 0=Person lives elsewhere','SEX: 1=Female, 0=Male','EXPERIENCE: Number of years of work experience','UNION: 1=Union member, 0=Not union member','WAGE: Wage (dollars per hour)','AGE: years','RACE: 1=Other, 2=Hispanic, 3=White','OCCUPATION: 1=Management, 2=Sales, 3=Clerical, 4=Service, 5=Professional, 6=Other','SECTOR: 0=Other, 1=Manufacturing, 2=Construction','MARR: 0=Unmarried,  1=Married']
short_names = [n.split(':')[0] for n in names]
data3 = pd.read_csv('wages.txt', skiprows=27, skipfooter=6, sep=None, header=None, names=short_names)
data3.columns =short_names
print(data3.columns)
seaborn.lmplot(y='WAGE', x='EDUCATION', hue='SEX', data=data3)
plt.show()


# ## Multivariant regression

# In[54]:


x= np.linspace(-5, 5, 21)
X, Y = np.meshgrid(x, x)
np.random.seed(1)
Z = -5 + 3*X - 0.5*Y + 8 * np.random.normal(size=X.shape)
## Plotting the data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface (X, Y, Z, cmap=plt.cm.coolwarm, rstride=1, cstride=1)
ax.view_init(elev=20, azim=-120)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


# In[ ]:




