#!/usr/bin/env python
# coding: utf-8

# # Intro to stats in python
# ### For a tutorial on these stats visit:
# https://scipy-lectures.org/packages/statistics/index.html
print("For a tutorial on these stats visit: https://scipy-lectures.org/packages/statistics/index.html")
print("Physical graphs will populate as .png files")

# Import Statements
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
print()

print("Running Python Stats Toutorial Examples")

# ## Import .csv file
data = pd.read_csv("brain_size.csv", sep=';' , na_values = ".")
print(data)
data1 = pd.read_csv("iris.csv", sep=',' , na_values = ".")

print("Running numpy Array")
      
# ### Numpy Array
t = np.linspace(-6, 6,20)
sin_t = np.sin(t)
cos_t = np.cos(t)
pd.DataFrame({'t': t, 'sin': sin_t, "cos" : cos_t})
      
# ## Manuipulating data
print("Section: manipulating data")
data.shape
print(data.shape)
print("Formatting Data Shape for brain_size.csv")
data.columns 
data.columns = pd.Index([u'Unnamed: 0', u'Gender', u'FSIQ', u'VIQ', u'PIQ', u'Weight', u'Height', u'MRI_Count'], dtype='object')
print(data.columns)

# #### Columns can be shown by name
print ("Example of Column Display by Name")
print(data['Gender'])

# # Simpler selector
print("Showing Means")
data[data['Gender'] == 'Female'] ['VIQ'].mean()
groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print("Gender Value Mean")
    print((gender, value.mean()))
print("Group by Gender Mean") 
print(groupby_gender.mean())
      
# ## Making Box and Whisker Plots
print("How to Show Box and Whisker Plots")
plt.figure(figsize=(4, 3))
data.boxplot(column= ['FSIQ', 'PIQ'])
plt.show()
plt.savefig("box_and_whisker_plot.png")

# #### Boxplotting differences
plt.figure(figsize=(4, 3))
plt.boxplot(data['FSIQ'] - data['PIQ'])
plt.xticks((1, ), ('FSIQ-PIQ',))
plt.show()
plt.savefig("comparative_boxplot.png")

# # Plotting data via scatter matirces
print("Running Example Scatter Matrices...")
pd.plotting.scatter_matrix(data[['Weight','Height', 'MRI_Count']])
plt.show()
plt.savefig("Weight_Height_MRI_Count_scatter_matrix.png")

pd.plotting.scatter_matrix(data[['VIQ','PIQ', 'FSIQ']])
plt.show()
plt.savefig("VIQ_PIQ_FSIQ_scatter_matix")

# ### Boxplots of columns by gender
print("Running Boxplots of Columns by Gender")
groupby_gender = data.groupby('Gender')
groupby_gender.boxplot(column=['FSIQ', 'VIQ', 'PIQ'])
plt.savefig("Boxplot_by_gender_columns.png")

# # Hypothesis Testing and Comparing Two Groups
print("Running Group comparisons and hypothesis testing")
stats.ttest_1samp(data['VIQ'], 0)
            
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)
print("T-test independent")
stats.ttest_ind(data['FSIQ'], data['PIQ'])
print("T-test 1 Sample")
stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0)

# ## Wilcoxon signed-rank test
print("Wilcoxon signed-rank test")
stats.wilcoxon(data['FSIQ'], data ['PIQ'])

# ## Linear Models
print("Linear Modeling Section")
# #### Generate simulated data according to the model
x = np.linspace(-5, 5, 20)
np.random.seed(1)
# normal distributed noise
y = -5 +3*x +4 * np.random.normal (size=x.shape)
# Create a data frame conatining all relavent variables
data = pd.DataFrame({'x' : x, 'y': y})

# ##### Specify OLS model and fit it to the graph

from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit()

# ##### Inspect Stats derived from the OLS model

from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit()
print(model.summary())
            
# ### Ex. Copmarison of male and female IQ based on brain size
## rerun data definitions to make work
import pandas as pd
data = pd.read_csv ("brain_size.csv",sep=';',na_values='.')
data.columns 
data.columns = pd.Index([u'Unnamed: 0', u'Gender', u'FSIQ', u'VIQ', u'PIQ', u'Weight', u'Height', u'MRI_Count'], dtype='object')
model = ols("VIQ ~ Gender", data).fit()
print(model.summary())

# ### Link to t-tests between different FSIQ and PIQ
print("Link to t-tests between different FSIQ and PIQ")
data_fisq = pd.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pd.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pd.concat((data_fisq, data_piq))
print(data_long)

[9,10,11,12,13,14,15,16,17,18,19,20,21]
model = ols ("iq ~ type", data_long).fit()
print(model.summary())
stats.ttest_ind(data['FSIQ'], data['PIQ'])

# ## Multiple Regression
print("Multiple Regression Analysis")
# ## Opening iris.csv from import
print("Opening iris.csv from import")
print(data1)
data1.shape #150 rows and 1? column
data1.columns = pd.Index([u'sepal_length', u'sepal_width', u'petal_length', u'petal_width', u'name'], dtype='object')
print(data1.shape)
print(data1.columns)
model = ols('sepal_width ~ petal_length', data1).fit()
print(model.summary())

# ## Analysis of petal sizes

categories = pd.Categorical(data1['name'])
pd.plotting.scatter_matrix(data1, c=categories.codes, marker='o')
fig = plt.gcf()
fig.suptitle("blue: setosa, green: versicolor, red: virginica", size=13)
plt.savefig("analysis_of_petal_sizes")

# # ANOVA
# ### Post-hoc hypothesis testing: analysis of varience
# ##### Write a vector of contrast
print("ANOVA Test Formatting")
test_input = [0, 1, -1, 0]
test_input_array = sm.add_constant(test_input)
result = model.f_test(test_input_array)
print (result)
      
# # More Visualization Using Seaborn
# ### Importing "wages.txt" from the web
print("Seaborn Visualization")
print("Importing 'wages.txt' from the web")
engine ='python'
if not os.path.exists('wages.txt'):
    urllib.request.urlretrieve('http://lib.stat.cmu.edu/datasets/CPS_85_Wages', 'wages.txt')
names = ['EDUCATION: Number of years of education','SOUTH: 1=Person lives in South, 0=Person lives elsewhere','SEX: 1=Female, 0=Male','EXPERIENCE: Number of years of work experience','UNION: 1=Union member, 0=Not union member','WAGE: Wage (dollars per hour)','AGE: years','RACE: 1=Other, 2=Hispanic, 3=White','OCCUPATION: 1=Management, 2=Sales, 3=Clerical, 4=Service, 5=Professional, 6=Other','SECTOR: 0=Other, 1=Manufacturing, 2=Construction','MARR: 0=Unmarried,  1=Married']
short_names = [n.split(':')[0] for n in names]
data3 = pd.read_csv('wages.txt', skiprows=27, skipfooter=6, sep=None, header=None, names=short_names)
data3.columns =short_names

# ### mulplicative factors
print("Running mulplicative factors analysis and printing graphs")
data3['WAGE'] = np.log10(data3['WAGE'])
seaborn.pairplot(data3, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg')
plt.savefig("Wage_age_education1.png")      
seaborn.pairplot(data3, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg', hue='SEX')
plt.suptitle('Effect of gender: 1=Female, 0=Male')
plt.savefig("Wage_age_education2.png")       
seaborn.pairplot(data3, vars=['WAGE','AGE','EDUCATION'], kind='reg', hue='RACE')
plt.suptitle('Effect of race: 1=Other, 2=Hispanic, 3=White')
plt.savefig("Wage_age_education3.png")       
seaborn.pairplot(data3, vars=['WAGE','AGE', 'EDUCATION'], kind='reg', hue='UNION')
plt.suptitle('Effect of union: 1=Union member, 0=Not union member')
plt.savefig("Wage_age_education4.png") 

# ### Plotting a simple regression
print("Plotting_simple_regression")
seaborn.lmplot(y='WAGE', x='EDUCATION', data=data3)
plt.show()
plt.savefig("simple_regression.png")

# ## Viewing wages.txt

print("Viewing wages from wages.txt")
print(data3)
print("running pairplot Wage vs. Age vs. Education")
seaborn.pairplot(data3, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg', hue='SEX')
plt.savefig("pairplot_Wage_Age_Education.png")

# ## lmplot for plotting a univariate regression
print("running univariate regression")
seaborn.lmplot(y='WAGE', x='EDUCATION', data=data3)
plt.savefig("univariate_regression.png")

# ### Correlation testing
            
import statsmodels.api as sm
from statsmodels.formula.api import ols
result = ols(formula='WAGE ~ EDUCATION + GENDER - EDUCATION * GENDER', data=data3).fit()
print(result.summary())

# ## Correlation Regression

print("running correlation regression")
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
plt.savefig("correlation_regression")

# ## Multivariant regression

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
plt.savefig("multivariant_regression")



