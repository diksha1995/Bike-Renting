
# coding: utf-8

# In[1]:


#import libraries
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from statistics import stdev
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import tree 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import metrics


# In[2]:


os.chdir("F:/Eddwisor/Task Program/Projects/Second Project Works")


# In[3]:


os.getcwd()


# In[4]:


df_data=pd.read_csv("day.csv",encoding = 'ISO-8859-1')


# In[5]:


#Check Dimensions
df_data.shape                   #rows = 731 and columns = 16


# In[6]:


#Get Names of the Columns
col_names = pd.DataFrame(df_data.columns)
#col_names


# In[7]:


df_data.head(4)


# In[8]:


# No need of instant for bike renting
#get the index of column instant and remove it from the dataset
instant_index = df_data.columns.get_loc("instant")
instant_index
df_data= df_data.drop(df_data.columns[instant_index],axis=1)


# In[9]:


instant_index


# In[10]:


df_data.info()
#Original dataset haave 4 float variables,10 int variables and 1 float vaiables


# In[11]:


num_data=df_data._get_numeric_data()
num_data.columns


# In[12]:


cat_data=df_data.select_dtypes(include=['object'])
cat_data.columns


# In[13]:


#Now we have two subset of dataset
# 1. num_data which contains only numerical variables data
# 2. cat_data which contains only categorical variables data


# In[14]:


df_data.shape    , num_data.shape  , cat_data.shape


# In[15]:


train, test = sklearn.model_selection.train_test_split(df_data, train_size = 0.7)


# In[16]:


train.shape ,test.shape          #train : rows = 511 & column =15 
                                 #test : rows = 220 & column =15


# # <b><i>Check For Missing Value Column Wise </b></i>

# In[17]:


pd.isna(df_data).sum()

#No missing value is available on sample set


# # <b>Analyses</b>

# In[18]:


# 1.How do the temperatures change across the seasons? What are the mean and median temperatures?

#First we converted the temperature, because the data of temperature was divided to 41. 
#Secondly we calculated the mean, the median and the standard deviation of all seasons.


# In[19]:


# Converting the nomalized temperature:
x = df_data.iloc[:, [8]]                     #temp is a column here
df_data['raw.temp']= x * 41                  #raw.data is a new column here 
df_data.head()


# In[20]:


#Calculating Median, Mean and Standard deviation of spring
rawtemp_col = df_data.iloc[:, [15]]                   #location of raw.data column
spring=rawtemp_col[df_data.season == 1]               
mean_sp = spring.mean()
median_sp = spring.median()
sd_sp=np.std(spring)


#Calculating Median, Mean and Standard deviation of summer.
rawtemp_col = df_data.iloc[:,[15]]
summer = rawtemp_col[df_data.season == 2]
mean_su = summer.mean()
median_su = summer.median()
sd_su = np.std(summer)


#Calculating Median, Mean and Standard deviation of fall.
rawtemp_col = df_data.iloc[:,[15]]
fall = rawtemp_col[df_data.season == 3]
mean_fa = fall.mean()
median_fa = fall.median()
sd_fa = np.std(fall)


#Calculating Median, Mean and Standard deviation of winter.
rawtemp_col = df_data.iloc[:,[15]]
winter = rawtemp_col[df_data.season == 4]
mean_wi = winter.mean()
median_wi = winter.median()
sd_wi = np.std(winter)


# In[21]:


#Spring
mean_sp , median_sp , sd_sp


# In[22]:


#Summer
mean_su, median_su, sd_su


# In[23]:


#Fall
mean_fa ,median_fa,sd_fa


# In[24]:


#Winter
mean_wi,median_wi,sd_wi


# In[25]:


#Spring:
#The mean temperature of spring was 12.21.
#The median temperature of spring was 11.72.
#The standard deviation of the temperature in spring was 4.21.

#Summer:
#The mean temperature of summer was 22.32.
#The median temperature of summer was 23.05.
#The standard deviation of the temperature in summer was 5.03.

#Fall:
#The mean temperature of fall was 28.96.
#The median temperature of fall was 29.3.
#The standard deviation of the temperature in fall was 2.9.

#Winter:
#The mean temperature of winter was 17.34.
#The median temperature of winter was 16.78.
#The standard deviation of the temperature in winter was 4.42.


# In[26]:


#Secondly we created a histogram displaying the temperatures of each season including lines for the mean and median temperatures


# In[27]:


#create histogram for the distribution of temperatures in spring
  
plt.hist(x = spring ,  bins = 10,
        histtype = 'bar', rwidth = 0.8)
plt.title("Temperatures in Spring")
plt.xlabel("Temperature in Celcius")
plt.ylabel("Number of Days") 


# In[28]:


#create histogram for the distribution of temperatures in summer
  
plt.hist(x = summer ,  bins = 10,
        histtype = 'bar', rwidth = 0.8)
plt.title("Temperatures in Summer")
plt.xlabel("Temperature in Celcius")
plt.ylabel("Number of Days") 


# In[29]:


#create histogram for the distribution of temperatures in fall
  
plt.hist(x = fall ,  bins = 10,
        histtype = 'bar', rwidth = 0.8)
plt.title("Temperatures in Fall")
plt.xlabel("Temperature in Celcius")
plt.ylabel("Number of Days") 


# In[30]:


#create histogram for the distribution of temperatures in winter
  
plt.hist(x = winter ,  bins = 10,
        histtype = 'bar', rwidth = 0.8)
plt.title("Temperatures in Winter")
plt.xlabel("Temperature in Celcius")
plt.ylabel("Number of Days") 


# In[31]:


#2. Is there a correlation between the temp/atemp/mean.temp.atemp and the total count of bike rentals?

#First we checked the dataset of integers, NA or NULL and duplicates.
#Because the dataset was already recoded and correct we created a new coloumn. Afterwards we did a correlation test.


# In[32]:


#Checking dataset 
#Tests if values in a vector are integers
df_data.dtypes


# In[33]:


#Tests if values in a vector are NA or NULL
#pd.isna(df_data) we tested it but due to the huge output we deleted it. There was no "NA".

pd.isna(df_data).sum()


# In[34]:


#Tests for duplicates
#There were no duplicates:df_data.duplicated() 

df_data.duplicated() 


# # <b>Creating a new Column</b>

# In[35]:


#The Dataset is already recoded and correct.
#For this question we converted "atemp" because it was devided of 50.
y = df_data.iloc[:,[9]]
df_data['raw.atemp'] = y * 50
df_data.head()


#Create a new column of the mean of raw.temp and raw.atemp.
df_data['raw.mean.temp.atemp'] =(df_data['raw.temp'] + df_data['raw.atemp'])/2
df_data.head()


# # <b>Correlation Test</b>

# In[36]:


# Correlation between raw.temp and the total count of bike rentals.

corr_temp = df_data['raw.temp'].corr(df_data['cnt'])
corr_temp            #The Correlation was 0.63

#Temprature <- df_data['raw.temp']
#Amount.Rentals <- df_data['cnt']


# In[37]:


#Correlation between atemp and the total count of bike rentals.

corr_atemp = df_data['raw.atemp'].corr(df_data['cnt'])
corr_atemp           #The Correlation was 0.63

#feeled temprature = df_data['raw.temp']
#annual rentals = df_data['cnt']


# In[38]:


#Correlation between mean.temp.atemp and the total count of bike rentals.

corr_mean_temp_atemp = df_data['raw.mean.temp.atemp'].corr(df_data['cnt'])
corr_mean_temp_atemp     #The Correlation was 0.63

#feeled raw temprature = df_data['mean.temp.atemp']
#annual rentals = df_data['cnt']


# # <b>Plotting the Correlations</b>

# In[39]:


x= df_data['raw.temp']
y= df_data['cnt']
plt.plot(x,y,color="red")


# In[40]:


x=df_data['raw.atemp']
y=df_data['cnt']
plt.plot(x,y,color="green")


# In[41]:


x=df_data['raw.mean.temp.atemp']
y=df_data['cnt']
plt.plot(x,y,color="blue")


# In[42]:


#3.Is temperature associated with bike rentals (registered vs. casual)?


# In[43]:


# Plotting the association:
x = df_data['raw.temp']
y = df_data['cnt']
plt.plot(x,y,color="orange")
plt.xlabel("Temperature in Celcius") 
plt.ylabel("Bike rentals")
type = "n"
plt.title("Association between temperature and bike rentals")


#Calculating min and max for the x-axis and y-axis:
min(df_data['raw.temp'])     #For raw.temp 


# In[44]:


max(df_data['raw.temp'])


# In[45]:


min(df_data['casual']) , min(df_data['registered'])


# In[46]:


max(df_data['casual']) ,max(df_data['registered'])


# In[47]:


# Calculating the correlation between raw.temp and registered users and between raw.temp and causal users
corr_reg = df_data['raw.temp'].corr(df_data['registered'])
corr_reg


# In[48]:


corr_cas = df_data['raw.temp'].corr(df_data['casual'])
corr_cas


# In[49]:


x = df_data['casual']
y = df_data['registered']
z = df_data['raw.temp']

plt.plot(x,z,color="blue")               #Blue represent the casual users
plt.plot(y,z,color="orange")             #Orange represent the registered users


# In[50]:


# 4.Can the number of total bike rentals be predicted by holiday and weather?


# In[51]:


lookup = {'numbers':["1","2","3","4"],
        'weather':["nice","cloudy", "wet", "lousy"]}
df=pd.DataFrame(lookup)

df_data['weather']= np.where(df_data['weathersit'] == 1,'nice','cloudy') 


# In[52]:


df_data.head()


# In[53]:


# 5. What are the mean temperature, humidity, windspeed and total rentals per months?


# In[54]:


# Months is coded as 1 to 12
# Converting month with "merge"

lookup_month ={'mnth' :["1","2","3","4","5","6","7","8","9","10","11","12"],
                          'mnth.name':["01Jan", "02Feb", "03March", "04April", "05May", "06June", "07July", "08Aug", "09Sept", "10Oct", "11Nov", "12Dec"]}

df=pd.DataFrame(lookup_month)

df_data['mnth.name'] = np.where(df_data['mnth']==1,'01Jan','02Feb')


# Convert the nomalized windspeed and humidity
df_windspeed=df_data.iloc[:,[11]]
df_data['raw.windspeed']=df_windspeed * 67

df_hum = df_data.iloc[:,[10]]
df_data['raw.hum']= df_hum * 100

df_data


# In[55]:


x = df_data['mnth.name']
z = df_data['raw.hum']

plt.bar(x,z,color="red")


# In[56]:


x = df_data['mnth.name']
y = df_data['raw.windspeed']

plt.bar(x,y,color="blue")


# # Decision  Tree

# In[57]:


#Convert String into int
import random
random.seed(1234)
df_data['dteday'],_ = pd.factorize(df_data['dteday'])
df_data['mnth.name'],_ = pd.factorize(df_data['mnth.name'])
df_data['weather'],_ = pd.factorize(df_data['weather'])

#Select the predictor feature and target variable
X = df_data.iloc[:,:-1]
y = df_data.iloc[:,-1]

#split data randomly into 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[58]:


#Create a Model for Decision Tree
deci_tree_model=tree.DecisionTreeRegressor(criterion="mse",min_samples_leaf=5).fit(X_train,y_train)
deci_tree_model


# In[59]:


#Prediction on test dataset
score = deci_tree_model.score(X_test, y_test)
print(score)


# In[60]:


#Compute and plot the RMSE
RMSE = np.sqrt(np.sum(((y_train-score)**2)/len(y_test)))
RMSE


# In[61]:


#Because of large dataset,we cannot built a decision tree here.It contain messy data.
#Create Dot file to Visualize Tree                           #http://webgraphviz.com
dotfile=open("pt.dot","w")     
df=tree.export_graphviz(deci_tree_model,out_file=dotfile,feature_names = X.columns)


# In[62]:


#Decision Tree:
#RMSE = 95.68%


# # Random Forest

# In[63]:


#Create a model for Random Forest
regressor = RandomForestRegressor(n_estimators=1000, random_state=0)  
regressor.fit(X_train, y_train)  


# In[64]:


#Prediction on test dataset
y_pred = regressor.predict(X_test) 


# In[65]:


#Calculate MAE
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


# In[66]:


#Random Forest :
#MAE : 0.30


# # Linear Regression

# In[67]:


#Linear Regression is apply on sample_set dataset.
#Create A model For Linear Regression
df_data['weather'],_ = pd.factorize(df_data['weather'])


# In[68]:


linearReg = linear_model.LinearRegression().fit(X_train,y_train)
linearReg


# In[69]:


#To Make Predictions On Test Dataset
logit_predictions = linearReg.predict(X_test)
#logit_predictions


# In[70]:


#Calculate Confusion Matrix for the model
CM = pd.crosstab(y_test,logit_predictions)
CM


# In[71]:


#Once we get Confusion Matrix then we calculate the term accordingly.
TN = CM.iloc[0,0]             #True Negative : 1
FN = CM.iloc[1,0]             #False Negative : 0
TP = CM.iloc[1,1]             #True Positive : 1
FP = CM.iloc[0,1]             #False Positive : 0
TN,FN,TP,FP


# In[72]:


#Calculate The Accuracy for the model
((TP+TN)*100)/(TN+FN+TP+FP)      #Accuracy = 100%


# In[73]:


#Calculate The False Negative Rate
(FN*100)/(FN+TP)                #FNR = 0.0%


# In[74]:


#Logistic Regression :
#Accuracy = 100%
#FNR = 0.0%

