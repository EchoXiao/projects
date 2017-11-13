
data=read.csv("train.csv",header=T,na.strings = "NA")

# remove ID
data=data[,-c(1)]

summary(data)
str(data)
# optional: data$MSSubClass=as.factor(data$MSSubClass)

# EDA
library(ggplot2)
ggplot(data = data) +
  geom_bar(mapping = aes(x = MSZoning )) + # bar for categorical
  ggtitle("Histogram for Varaible MSZoning")

summary(data$LotArea) # to determine binwidth
ggplot(data = data) +
  geom_histogram(mapping = aes(x = LotArea), binwidth =5000) + # histogram for continuous
  ggtitle("Histogram for Varaible LotArea")

library(dplyr)
data %>% count(MSZoning)

data %>% count(cut_width(LotArea, 5000))

# data manipulation use filter then visualize
outlier = data %>% filter(LotArea>50000)
ggplot(data = outlier) +
  geom_histogram(mapping = aes(x = LotArea), binwidth =5000)

# visualize a categorical and a continuous variable
ggplot(data = data, mapping = aes(x = LotArea, colour = KitchenQual)) +
  geom_freqpoly(binwidth = 5000)

library(lattice)
densityplot(data$SalePrice)

ggplot(data = data, mapping = aes(x = BsmtCond , y = SalePrice)) +
  geom_boxplot()

# Percentage of NA
MissingPercentage <- function(x){sum(is.na(x))/length(x)*100}
sort(apply(data,2,MissingPercentage),decreasing=TRUE)

# check # of NA
sort(sapply(data, function(x) sum(is.na(x))),decreasing=TRUE)

# Visualize missing data
library(VIM)
aggr_plot <- aggr(data, 
                  col=c('navyblue','red'), 
                  numbers=TRUE, 
                  sortVars=TRUE, 
                  labels=names(data), 
                  cex.axis=.7, 
                  gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))

# Delete columns with more than 5% missing data 
library(dplyr)
data=select(data,-c(PoolQC,MiscFeature,Alley,Fence,FireplaceQu,LotFrontage))


# CART: classification and regression trees
library(mice)
imp_data<- mice(data, m=1, method='cart', printFlag=FALSE)

# Test Original and Imputed
table(data$GarageType)
table(imp_data$imp$GarageType)

# vasualize density blue:actual; red:imputed
densityplot(imp_data, ~GarageType)

# Merge to Original Data
data_complete <- complete(imp_data)

#Confirm no NAs
sum(sapply(data_complete, function(x) { sum(is.na(x)) }))

write.csv(data_complete, file = "data_complete.csv")
data_complete=read.csv("data_complete.csv",header=TRUE)

################### Multiple Linear Regression #####################

set.seed(1) # for reproducibility
train = sample(1:nrow(data_complete),nrow(data_complete)/2)
#train = sample(nrow(data_complete), 1000)
test = -train
traindata = data_complete[train,]
testdata = data_complete[test,]

# This is not always true due to imputation
# testdata=filter(testdata,Condition1!="RRNe")
##traindata=rbind(traindata,condition1fromtest)

distinct(traindata,Condition1)
distinct(testdata,Condition1)

model0=lm(SalePrice~.,data=traindata) # unused levels have been dropped
distinct(data_complete,RoofStyle)
distinct(traindata,RoofStyle)
summary(model0)

# Calculate training set Root Mean Squared Error 18K
RMSE <- sqrt(mean(model0$residuals^2)) 
RMSE

# RMSE <- sqrt(mean((y-y_pred)^2))

#NA as a coefficient in a regression indicates that the variable in question is linearly related to the other variables. 
#a.k.a collinearity
model1=lm(SalePrice~
            LotArea+OverallQual+OverallCond+YearBuilt+BsmtQual+BsmtFinSF1+
            BsmtFinSF2+BsmtUnfSF+X1stFlrSF+X2ndFlrSF+BedroomAbvGr+
            KitchenAbvGr+KitchenQual+GarageCars+PoolArea,
          data=traindata)
summary(model1)

# make prediction based on test set
predict_model= predict(model1,testdata)
head(predict_model) #prediction results
head(testdata$SalePrice) # vs. actual Saleprice

# calculate the value of R-squared for the prediction model on the test data set as follows:
SSE <- sum((testdata$SalePrice - predict_model) ^ 2)
SST <- sum((testdata$SalePrice - mean(testdata$SalePrice)) ^ 2)
1 - SSE/SST

# testset RMSE compare to traindata
testRMSE <- sqrt(mean((predict_model - testdata$SalePrice)^2))
testRMSE #43K
trainRMSE <- sqrt(mean(model1$residuals^2)) 
trainRMSE #26K

# Diagnostic Plots
par(mfrow=c(2,2))
plot(model1)

# multicollinearity
library(car)
vif(model1)

# Linear Regression Assumptions-Independence
library(lmtest)
dwtest(model1)

# Basic Scatterplot Matrix
# pairs(~SalePrice+YearBuilt+
# BsmtFinSF1+BsmtFinSF2+BsmtUnfSF+X1stFlrSF+X2ndFlrSF,
# data=traindata,main="Simple Scatterplot Matrix")

# this correlation test: round(cor(traindata,method="pearson"),2) won't work due to categorical variables

# dropped the "bad" independent variable that cause multicolinearity and not significant variables
model2=lm(SalePrice~
            LotArea+OverallQual+OverallCond+YearBuilt+BsmtQual+
            X1stFlrSF+X2ndFlrSF+BedroomAbvGr+
            KitchenAbvGr+KitchenQual+GarageCars+PoolArea,data=traindata)
summary(model2)

trainRMSE <- sqrt(mean(model2$residuals^2)) 
trainRMSE #28k
predict_model= predict(model2,testdata)
testRMSE <- sqrt(mean((predict_model - testdata$SalePrice)^2))
testRMSE #41k

library(car)
vif(model2) #result looks good now:)
par(mfrow=c(2,2))
plot(model2)

# Interaction Term
model_int=lm(SalePrice~
               LotArea+OverallQual+OverallCond+YearBuilt+BsmtQual+
               BsmtFinSF1*X1stFlrSF+X2ndFlrSF+BedroomAbvGr+
               KitchenAbvGr+KitchenQual+GarageCars+PoolArea,data=traindata)
summary(model_int)

trainRMSE <- sqrt(mean(model_int$residuals^2)) 
trainRMSE
predict_model= predict(model_int,testdata)
testRMSE <- sqrt(mean((predict_model - testdata$SalePrice)^2))
testRMSE

# test the hypothesis that the 2nd model adds explanatory value over the 1st model
anova(model2,model_int,test = "F")


# Log Transformation
modellog=lm(log(SalePrice)~
              LotArea+OverallQual+OverallCond+YearBuilt+BsmtQual+
              X1stFlrSF+X2ndFlrSF+BedroomAbvGr+
              KitchenAbvGr+KitchenQual+GarageCars,data=traindata)
summary(modellog)

par(mfrow=c(2,2))
plot(modellog)

modellog2=lm(log(SalePrice)~
               poly(LotArea,2)+OverallQual+OverallCond+YearBuilt+BsmtQual+
               X1stFlrSF+X2ndFlrSF+BedroomAbvGr+
               KitchenAbvGr+KitchenQual+GarageCars,data=traindata)

summary(modellog2)
anova(modellog,modellog2)

trainRMSE <- sqrt(mean((exp(modellog2$fitted.values)-traindata$SalePrice)^2)) 
trainRMSE #25K
predict_model= exp(predict(modellog2,testdata))
testRMSE <- sqrt(mean((predict_model - testdata$SalePrice)^2))
testRMSE #85K

#### Stepwise
library(leaps)
regfit.full=regsubsets(SalePrice~
                         LotArea+OverallQual+OverallCond+YearBuilt+BsmtQual+BsmtFinSF1+
                         BsmtFinSF2+BsmtUnfSF+X1stFlrSF+X2ndFlrSF+BedroomAbvGr+
                         KitchenAbvGr+KitchenQual+GarageCars+PoolArea,data=traindata,nvmax=19)
#nvmax default max is 8
summary_reg = summary(regfit.full)
names(summary_reg)
summary_reg$rsq # r square
summary_reg$adjr2
par(mfrow =c(2,2))
plot(summary_reg$rss ,xlab=" Number of Variables ",ylab=" RSS",
     type="l") # type l connect dot with line 
plot(summary_reg$adjr2 ,xlab =" Number of Variables ",
     ylab=" Adjusted R-Square",type="l")
which.max(summary_reg$adjr2)
plot(regfit.full,scale="adjr2") #visualize model selection
which.min(summary_reg$rss)
which.min(summary_reg$bic)

regfit.bwd=regsubsets (SalePrice~.,nvmax =50,really.big = TRUE,
                       method="backward",data=traindata)
summary(regfit.fwd)
summary_reg2= summary(regfit.bwd)
which.min(summary_reg2$rss)

par(mfrow=c(2,2))
plot(summary_reg2$adjr2,xlab="Number of variables", ylab="Adjusted R-Square", type="l")
points(1:50,summary_reg2$rsq[1:50], col="red",cex=2,pch=20)
plot(summary_reg2$rss,xlab="Number of variables", ylab="RSS", type="l")
points(1:50,summary_reg2$rss[1:50], col="blue",cex=2,pch=20)

coef(regfit.bwd,20)

# since no predict function for regsubsets
predict.regsubsets =function (object ,newdata ,id ,...){
  form=as.formula (object$call[[2]])
  mat=model.matrix (form,newdata)
  coefi =coef(object,id=id)
  xvars =names (coefi)
  mat[,xvars ]%*% coefi
}

best_subset_pred = predict.regsubsets (regfit.bwd,testdata,20)

testRMSE <- sqrt(mean((best_subset_pred- testdata$SalePrice)^2))
testRMSE

########### Ridge and Lasso ##########

# convert any qualitative variables to dummy variables
x=model.matrix(SalePrice~.,data_complete)[,-1] # get rid of intercept column
head(x)
y=data_complete$SalePrice

# split the dataset into training and testing
set.seed(1)
train = sample(1:nrow(x), nrow(x)/2)
test = -train

training_x = x[train,]
testing_x = x[test,]

training_y = y[train]
testing_y = y[test]

# glmnet package to perform ridge and lasso
library(glmnet)
#lambda 10^10 to 10^-2
grid = 10^seq (10,-2,length =100)
ridge_model = glmnet(training_x,training_y,alpha = 0,lambda = grid, standardize = FALSE)
dim(coef(ridge_model)) # 233 for predictor, 100 for lambda

plot(ridge_model, xvar = "lambda", label = TRUE)

### choose the best value of lambda that would minimize the error. Run cross validation
set.seed(2)
cv_error = cv.glmnet(training_x, 
                     training_y, 
                     alpha = 0) #default 10 fold cv
plot(cv_error) # cross-validation curve (red dotted line), upper and lower standard deviation curves 
# Two selected ??'s:vertical dotted lines

best_lambda = cv_error$lambda.min
best_lambda

# model with best lambda
model_coef = predict(ridge_model, 
                     type = "coefficients",
                     s= best_lambda)

### test the model 
predicted_y = predict(ridge_model, 
                      s= best_lambda,
                      newx = testing_x)
### RMSE
sqrt(mean((predicted_y - testing_y)^2))


##### LASSO 
lasso_model = glmnet(training_x, 
                     training_y, 
                     alpha =1,
                     lambda=grid,
                     standardize=FALSE)

plot(lasso_model, xvar = "lambda",label = TRUE)

set.seed(2)
cv_error = cv.glmnet(training_x, 
                     training_y, 
                     alpha = 1)
best_lambda = cv_error$lambda.min
best_lambda

plot(cv_error)

### OUR FINAL LASSO 
model_coef = predict(lasso_model, 
                     type = "coefficients",
                     s= best_lambda)

### test the model 
predicted_y = predict(lasso_model, 
                      s= best_lambda,
                      newx = testing_x)
### RMSE
sqrt(mean((predicted_y - testing_y)^2))


### if imputation was completed in "the real" test set, make the prediction and submit the answer
### BestPred <- data.frame(Id = data2$Id, SalePrice= predict(lasso_model, data2))
### write_csv(BestPred, "BestPred.csv")


############ Binary Logistic Regression #############

# create a binary variable "sale_above_avg", delete SalePrice
data_complete$sale_above_avg =ifelse(data_complete$SalePrice>=mean(data_complete$SalePrice),1,0)
head(data_complete)
library(dplyr)
data_complete_l=select(data_complete,-SalePrice)

set.seed(1) # for reproducibility
train = sample(1:nrow(data_complete),nrow(data_complete)/2)
#train = sample(nrow(data_complete), 1000)
test = -train
traindata = data_complete_l[train,]
testdata = data_complete_l[test,]

# fit the model
glm.fit=glm(sale_above_avg~
              LotArea+OverallQual+OverallCond+YearBuilt+BsmtQual+
              X1stFlrSF+X2ndFlrSF+BedroomAbvGr+
              KitchenAbvGr+KitchenQual+GarageCars,
            data=traindata,family =binomial)
summary(glm.fit) # but have a problem of perfect separation
anova(glm.fit, test="Chisq")

# Prediction and accuracy test

fitted.results <- predict(glm.fit,newdata=testdata,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != testdata$sale_above_avg)
print(paste('Accuracy',1-misClasificError))


table=table(fitted.results,testdata$sale_above_avg)
table
error_rate <- (table[1,2]+table[2,1])/sum(table)
error_rate

#ROC Curve
library(ROCR)
rocpred <- prediction(fitted.results,testdata$sale_above_avg)
rocperf <- performance(rocpred,'tpr','fpr')
plot(rocperf,colorize=TRUE, text.adj=c(-0.2,1.7))
