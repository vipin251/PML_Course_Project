---
title: "Practical Machine Learning Project"
author: "Vipin"
date: "October 20, 2015"
output: html_document
---
## Objective
People use devices to collect data about their personal activity.These type of devices are part of the quantified self movement.The objective is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website  here: http://groupware.les.inf.puc-rio.br/har.   
The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. Use the other available variables to predict the way a participant did the project.

## Reading and Preprocessing of the data  
Read the training files. Some of the columns are NAs. Remove those columns. Also checked the columns wiht no or near zero variability.


```r
# Read files
library(caret)
raw_file <- read.csv("pml-training.csv",na.strings = "NA",stringsAsFactors = F)

#get number of NAs in each cols
nas <- (sapply((lapply(raw_file, is.na)), sum))
# Remove columns with all NAs
na_removed <- raw_file[,nas<19000]

#Remove cols contains No data
raw_cleaned <- na_removed[,-c(grep("^kurtosis|^skewness|^max_yaw|^min_yaw|^amplitude|new_window|cvtd_timestamp", names(na_removed)))]

#nearZeroVar(raw_cleaned,saveMetrics = T)
```

##Create a train and test set  
The data is then partitioned into a test and a train set. 75% of the data is used to train the model and rest is used for testing.

```r
set.seed(122)
Intrain <- createDataPartition(raw_cleaned$classe, p = 0.75, list = F)
train <- raw_cleaned[Intrain,]
test <- raw_cleaned[-Intrain,]
```
## Fit a random forest model with 5 Fold cross validation
Different models like rpart and gbm tried for prediction and randomforest gives highly accurate metrics with 100 trees. Model is then validated with 5 fold cross validation.

```r
library(caret)
rf_fit <- train(as.factor(classe) ~ ., 
                method = "rf",
                data = train[,-c(1,2)],
                ntree = 100,
                na.action = na.omit,
                importance = TRUE,
                keep.inbag=T,tuneLength=1,
                trControl = trainControl(method = "cv",
                                         number = 5))

  
print(rf_fit)
```

```
Random Forest 

14718 samples
   55 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 11775, 11774, 11775, 11773, 11775 
Resampling results

  Accuracy   Kappa     Accuracy SD  Kappa SD    
  0.9988449  0.998539  0.000568666  0.0007193053

Tuning parameter 'mtry' was held constant at a value of 18
 
```

```r
rf_fit$finalModel
```

```

Call:
 randomForest(x = x, y = y, ntree = 100, mtry = param$mtry, importance = TRUE,      keep.inbag = ..3) 
               Type of random forest: classification
                     Number of trees: 100
No. of variables tried at each split: 18

        OOB estimate of  error rate: 0.12%
Confusion matrix:
     A    B    C    D    E class.error
A 4185    0    0    0    0 0.000000000
B    2 2845    1    0    0 0.001053371
C    0    4 2563    0    0 0.001558239
D    0    0    6 2405    1 0.002902156
E    0    0    0    3 2703 0.001108647
```
# Validate the model using the test data set  
Final model has a **OOB error rate less than 0.12%**after 5 fold cross validation and is highly accurate with Accuracy .999. The model is then tested with the test set and following results obtained.

```r
test$predict <- predict(rf_fit,test[,-c(1,2)],type = "raw")
confusionMatrix(test$predict, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    0  949    2    0    0
##          C    0    0  853    2    0
##          D    0    0    0  801    0
##          E    0    0    0    1  901
## 
## Overall Statistics
##                                           
##                Accuracy : 0.999           
##                  95% CI : (0.9976, 0.9997)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9987          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9977   0.9963   1.0000
## Specificity            1.0000   0.9995   0.9995   1.0000   0.9998
## Pos Pred Value         1.0000   0.9979   0.9977   1.0000   0.9989
## Neg Pred Value         1.0000   1.0000   0.9995   0.9993   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1935   0.1739   0.1633   0.1837
## Detection Prevalence   0.2845   0.1939   0.1743   0.1633   0.1839
## Balanced Accuracy      1.0000   0.9997   0.9986   0.9981   0.9999
```
