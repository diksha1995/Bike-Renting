#Remove The Variable & Values In Environment
rm(list=ls())

#Set The Directory
setwd("F:/Eddwisor/Task Program/Projects/Second Project Works")

#Get The Directory
getwd()

#------Load the dataset-------#
df_data=read.csv("day.csv",encoding = 'ISO-8859-1')

#-----Check Dimension-------#
dim(df_data)              #rows=731 and column=16

#Get the Column Names
names(df_data)

#Get the structure of the dataset
str(df_data)
head(df_data,4)

#The DataSet contain 12 independent variables and 3 dependent variables
#Split the dataset into train and test dataset
library(caret)
intrain<-createDataPartition(y=df_data$cnt,p=0.7,list=FALSE)
train<-df_data[intrain,]            
test<-df_data[-intrain,]

#------Check Dimension of Train & Test Dataset ----------#
dim(train)              #rows=515  and column=16

dim(test)               #rows=216  and column=16

#Get The Column Names of Train & Test Dataset
names(train)
names(test)

#------------------------------------------------------------------
#No need of instant for Bike Renting
#get the index of column instant and remove it from the train dataset

instant_index=match("instant",names(train))
instant_index
train=train[-instant_index]

#After Removing The columns Check The train Dataset
str(train)
#Train dataset have 4 num variables, 10 int and 1 factor
#------------------------------------------------------------------

#------------------------------------------------------------------
#No need of instant for Bike Renting
#get the index of column instant and remove it from the test dataset

instant_index=match("instant",names(test))
instant_index
test=test[-instant_index]

#In test dataset no need of dependent variables
#So we get the index of columns and remove it from the test dataset

casual_index=match("casual",names(test))
casual_index
test=test[-casual_index]

registered_index=match("registered",names(test))
registered_index
test=test[-registered_index]

cnt_index=match("cnt",names(test))
cnt_index
test=test[-cnt_index]

#After Removing The columns Check The test Dataset
str(test)
#Train dataset have 4 num variables, 7 int and 1 factor
#------------------------------------------------------------------

#Combine both Train and Test Data set(to understand the distribution of independent variable together)

test$registered=0
test$casual=0
test$cnt=0
data=rbind(train,test)

#Get the Structure Of Dataset
str(data)
#Combine dataset have 7 num variables,7 int and 1 factor
#------------------------------------------------------------------

#Check for the Missing Value
sum(is.na(data))                      #no missing value is avaliable

#Plot the histogram of each numerical variables and analyze the distribution
par(mfrow=c(2,2))
par(mar = rep(2, 4))
hist(data$season)
hist(data$weather)
hist(data$hum)
hist(data$holiday)
hist(data$workingday)
hist(data$temp)
hist(data$atemp)
hist(data$windspeed)

prop.table(table(data$weathersit))

#Convert discrete variables into factor (season, weather, holiday, workingday)

data$season=as.factor(data$season)
data$weathersit=as.factor(data$weathersit)
data$holiday=as.factor(data$holiday)
data$workingday=as.factor(data$workingday)

#Outliers Detection using Multivariate Analysis
#Here I have added some additional hypothesis from the dataset. Let's test them one by one:

##############################################################################
# 1. Hourly Trend:
#We don't have variable 'hour' with us right now.But we can extract it using datetime(dteday)column.

data$hour=substr(data$dteday,12,13)
data$hour=as.factor(data$hour)

#Let's plot the hourly trend of count over hours and check if our hypothesis is correct or not. 
#We will separate train and test data set from combined one.

train=data[as.integer(substr(data$dteday,9,10))<20,]      #training dataset is for the first 19 days of each month
test=data[as.integer(substr(data$dteday,9,10))>19,]       #Test dataset is from 20th days to month ends.

boxplot(train$cnt ~ train$hour,xlab="hour",ylab="count of users",main="count of users Vs.hour")

#Distribution Of Registered and Casual Users Seperately
boxplot(train$registered ~ train$hour,xlab="hour",ylab="Registered users",main="Registered Users Vs.hour")
boxplot(train$casual ~ train$hour,xlab="hour",ylab="Casual users",main="Casual Users Vs.hour")
#Above you can see that registered users have similar trend as count. Whereas, casual users have different trend. 
#Thus, we can say that 'hour' is significant variable and our hypothesis is 'true'.

#############################################################################
# 2. Daily Trend:
# Like Hour,we will generate a variable for day from datetime(dteday)variable and after that we'll plot it.

date=substr(data$dteday,1,10)
days<-weekdays(as.Date(date))
data$day=days

#Plot the Registered and casual Users seperately
boxplot(data$registered ~ data$day,xlab="day",ylab="registered users",main="Registered users vs. day")
boxplot(data$casual ~ data$day,xlab="day",ylab="casual users",main="Causal users vs.day")

#While looking at the plot, I can say that the demand of causal users increases over weekend.
############################################################################
# 3.Rain :
#We don't have the 'rain' variable with us but have 'weathersit' which is sufficient to test our hypothesis. 

data$rain=substr(data$weathersit,12,13)
data$rain=as.factor(data$rain)

#train=data[as.integer(substr(data$weathersit,9,10))<20,]
#test=data[as.integer(substr(data$weathersit,9,10))>19,]

boxplot(data$registered ~ data$rain,xlab="rain",ylab="registered users",main="Registered Users Vs.Rain")
boxplot(data$casual ~ data$rain,xlab="rain",ylab="casual users",main="Casual Users Vs. Rain")

#It is clearly satisfying our hypothesis.
###########################################################################
# 4.Temperature, Windspeed and Humidity :
# These are continuous variables so we can look at the correlation factor to validate hypothesis.

sub=data.frame(train$registered,train$casual,train$cnt,train$temp,train$hum,train$atemp,train$windspeed)
cor(sub)

#Variable temp is positively correlated with dependent variables (casual is more compare to registered)
#Variable atemp is highly correlated with temp.
#Windspeed has lower correlation as compared to temp and humidity.
##########################################################################
# 5.Year : 
#We have a yr variable to hypothesis the varaibles.
boxplot(data$cnt ~ data$yr,xlab="year",ylab="count",main="Count Vs.Year")

#Here 0 represent 2011 and 1 represent 2012
#You can see that 2012 has higher bike demand as compared to 2011.
##########################################################################
# 6.Pollution & Traffic :
#We don't have the variable related with these metrics in our data set so we cannot test this hypothesis.

##########################Feature Engineering#############################
train$hour = as.integer(train$hour)           #Convert hour to integer
test$hour = as.integer(test$hour)             #Modifying in both train and test data set

#We use the library rpart for decision tree algorithm.
library(rpart)
library(rattle)                 #These libraries will be used to get a good visual plot for the decision model
library(rpart.plot)
library(RColorBrewer)

hour_reg=rpart(registered~hour,data=train)
summary(hour_reg)
fancyRpartPlot(hour_reg)

hour_cas=rpart(casual~hour,data=train)
summary(hour_cas)
fancyRpartPlot(hour_cas)

#We have created bins for temperature for both registered and casual users

temp_reg=rpart(registered~temp,data=train)
summary(temp_reg)
fancyRpartPlot(temp_reg)

temp_cas=rpart(casual~temp,data=train)
summary(temp_cas)
fancyRpartPlot(temp_cas)

#We have created bins for year for both registered and casual users

year_reg=rpart(registered~yr,data = train)
summary(year_reg)
fancyRpartPlot(year_reg)

year_cas=rpart(casual~yr,data = train)
summary(year_cas)
fancyRpartPlot(year_cas)

#We have created bins for day for both registered and casual users
day_reg=rpart(registered~day,data = data)
summary(day_reg)
fancyRpartPlot(day_reg)

day_cas=rpart(casual~day,data = data)
summary(day_cas)
fancyRpartPlot(day_cas)

##########################################################################
#Before executing random forest,execute following steps
#Convert discrete variables into factor(weathersit,season,hour,holiday,workingday,mnth)

train$hour = as.factor(train$hour)
test$hour = as.factor(test$hour)

train$weathersit = as.factor(train$weathersit)
test$weathersit  = as.factor(test$weathersit)

train$season = as.factor(train$season)
test$season = as.factor(test$season)

train$holiday = as.factor(train$holiday)
test$holiday = as.factor(test$holiday)

train$workingday = as.factor(train$workingday)
test$workingday = as.factor(test$workingday)

train$mnth = as.factor(train$mnth)
test$mnth = as.factor(test$mnth)


#log transformation for some skewed variables, which can be seen from their distribution.
train$reg1=train$registered+1
train$cas1=train$casual+1
train$logcas=log(train$cas1)
train$logreg=log(train$reg1)
test$logreg=0
test$logcas=0


#predicting the log of registered users.
library(randomForest)
set.seed(415)
fit1 <- randomForest(logreg ~ hour +workingday+holiday+hum+atemp+windspeed+season+weathersit+yr, data=train,importance=TRUE, ntree=250)
pred1=predict(fit1,test)
test$logreg=pred1
print(fit1)
plot(fit1,main="Random Forest For Registered Users")


#predicting the log of casual users.
set.seed(415)
fit2 <- randomForest(logcas ~hour +hum+atemp+windspeed+season+weathersit+holiday+workingday+yr, data=train,importance=TRUE, ntree=250)
pred2=predict(fit2,test)
test$logcas=pred2
print(fit2)
plot(fit2,main="Random Forest For Casual Users")

#Re-transforming the predicted variables and then writing the output of count to the file submit.csv
#creating the final submission file
test$registered=exp(test$logreg)-1
test$casual=exp(test$logcas)-1
test$cnt=test$casual+test$registered
s<-data.frame(dteday=test$dteday,count=test$cnt)
write.csv(s,file="submit.csv",row.names=FALSE)

##########################Logistic Regression###########################
#Perform Logistic Regression for Registered Users
logit_model_1 <-glm(logreg ~ season+mnth+holiday+workingday+weathersit, data=train)
pred_model_1 <-predict(logit_model_1,test)
test$logreg=pred_model_1
print(logit_model_1)
plot(logit_model_1,main="Logistic Regression for Registered Users")

#Perform Logistic Regression For Casual Users
logit_model_2 <-glm(logcas ~season+mnth+holiday+workingday+weathersit,data=train)
pred_model_2 <-predict(logit_model_2,test)
test$logcas=pred_model_2
print(logit_model_2)
plot(logit_model_2,main="Logistic Regression For Casual Users")

y_pred <- ifelse(pred_model_1 > 0.5, 1, 0)
y_act <- test$logreg

#######################Evaluation Of Model############################
###Computing a simple ROC curve (x-axis: fpr, y-axis: tpr)
# 1.Logistic Regression
# Performance Evaluation Of Logistic Regression for Registered Users
library(ROCR)
pred_logit_reg<- prediction(predict(logit_model_1), train$holiday)
perf_logit_reg <- performance(pred_logit_reg,"tpr","fpr")
plot(perf_logit_reg,main="Performance Of Logistic Regression For Registered Users")

#Performance Evaluation Of Logistic Regression for Casual Users
pred_logit_cas<- prediction(predict(logit_model_2), train$holiday)
perf_logit_cas <- performance(pred_logit_cas,"tpr","fpr")
plot(perf_logit_cas,main="Performance Of Logistic Regression For Casual Users")


# 2.Random Forest
#Performance Evaluation Of Random Forest for Registered Users
pred_randomforest_reg <-prediction(predict(fit1),train$holiday)
perf_randomforest_reg <-performance(pred_randomforest_reg,"tpr","fpr")
plot(perf_randomforest_reg,main="Performance of Random Forest For Registered Users")

#Performance Evaluation Of Random Forest for Casual Users
pred_randomforest_cas <-prediction(predict(fit2),train$holiday)
perf_randomforest_cas <-performance(pred_randomforest_cas,"tpr","fpr")
plot(perf_randomforest_cas,main="Performance of Random Forest For Casual Users")

#we have here is a line that traces the probability cutoff from 1 at the bottom-left to 0 in the top right.
#This is a way of analyzing how the sensitivity and specificity perform for the full range of probability cutoffs,
#that is from 0 to 1.

#----------------------------------------------------------------------------------------------

#Evaluation Using Concordance & Discordance
install.packages("InformationValue")  # For stable CRAN version

#Performance Evaluation of Registered Users Using Concordance $ Discordance
InformationValue::Concordance(test$logreg, test$holiday)

##Performance Evaluation of Casual Users Using Concordance $ Discordance
InformationValue::Concordance(test$logcas,test$holiday)

#we take all possible combinations of true events and non-events. 
#Concordance is the percentage of pairs, 
#where true event's probability scores are greater than the scores of true non-events.
#Here no pairing is given for calculating the concordance.

#--------------------------------------------------------------------------------------

#Evaluation Using ks-plot of Registered Users
InformationValue::ks_plot(test$logreg,test$holiday)

#Evaluation Using ks-plot of Casual Users
InformationValue::ks_plot(test$logcas,test$holiday)

#The KS chart and statistic that is widely used in credit scoring scenarios 
#and for selecting the optimal population size of target users for marketing campaigns.

#--------------------------------------------------------------------------------------

#Evaluation Using ks-stat of Registered Users
InformationValue::ks_stat(test$logreg,test$holiday)

InformationValue::ks_stat(test$logreg,test$holiday,returnKSTable = T)

#Evaluation Using ks-stat of Casual Users
InformationValue::ks_stat(test$logcas,test$holiday)

InformationValue::ks_stat(test$logcas,test$holiday,returnKSTable = T)

#The significance of KS statistic is, it helps to understand, 
#what portion of the population should be targeted to get the highest response rate (1's).

#--------------------------------------------------------------------------------------

#For Recall,Precision and fscore we have to convert variables into factor
test$workingday  =  as.factor(test$workingday)
test$holiday = as.factor(test$holiday)

#Evaluation Using Recall,Precision and F-score
recall(test$workingday,test$holiday)                 #recall = 29.21
precision(test$workingday,test$holiday)              #precision = 90.69

#F1 Score = (2 * Precision * Recall) / (Precision + Recall)
f1_score = (2 * 90.69 * 29.21)/(90.69 + 29.21)       #f1_score = 44.18
f1_score

#You have an F1 Score of 44.18 percent.That's not so good.
#A good model should have a good precision as well as a high recall. 
#So ideally, I want to have a measure that combines both these aspects in one single metric - the F1 Score.

#-------------------------------------------------------------------------------------------

#Evalution Performance Using Confusion Matrix
library(caret)
caret::confusionMatrix(test$workingday, test$holiday, positive="1", mode="everything")

#The rows in the confusion matrix are the count of predicted 0's and 1's (from y_pred), 
#while, the columns are the actuals (from y_act).

#-------------------------------------------------------------------------------------------
