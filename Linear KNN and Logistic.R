#### Load the library ####
library("MASS")
library("class")
library("tidyverse")

#### Search for the help documents about this dataset ####
?biopsy

#### Get data ####
data(biopsy)
biopsy[1:5, ]

#### Delete the missing value ####
biopsy2 <- biopsy[!is.na(biopsy$V6), ]

#### Set labels, benign for 0 and malignant for 1 #####
len_bio <- nrow(biopsy2) ## Row number of the dataset
label <- rep(1, len_bio) ## Initial label

biopsy2 <- cbind(biopsy2, label) ## Add labels to biopsy2
biopsy2[which(biopsy2$class == "benign"), ]$label <- 0  ## Set label 0 for benign

############################################################################################
############# Randomly split the data into a training set and a test set####################  
############# containing about 2/3 and 1/3 of total observations respectively.##############
############################################################################################

#### Split data into a training set and a test set ####
set.seed(1234) ## Initial seeds
len_train <- round(len_bio / 3 * 2)
len_test <- round(len_bio / 3)
train_num <- sample(len_bio, len_train) ## Get the 2/3 dataset's indexes randomly.
test_num <- sample(len_bio, len_test) ## Get the 1/3 dataset's indexes randomly.

biopsy_train <- biopsy2[train_num, 2:12] ## Get training set
y_train <- biopsy2[train_num, 12]

biopsy_test <- biopsy2[test_num, 2:12] ## Get test set
y_test <- biopsy2[test_num, 12]



##################################### Linear Regression ######################################
fit_LR <- lm(label ~V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = biopsy_train)
summary(fit_LR)

#### Training Error ####
ypred <- predict(fit_LR)  ## Use coeffient to calculate y -- label

lmtrain <- 
  cbind(biopsy_train, ypred) %>% 
  select(label, ypred) %>% 
  mutate(diff = abs(ypred - label))  ## Calculate differences between predicted value and actural value

lmerrTr <- sum(lmtrain$diff > 0.5) / len_train ## Calculate training data error ratio


#### Testing Error ####
test_lmnew <- biopsy_test[, 1:9] ## Get predictors in dataset
ytpred <- predict(fit_LR, newdata = test_lmnew) ## Use coeffient to calculate y -- label

lmtest <- 
  cbind(biopsy_test, ytpred) %>% 
  select(label, ytpred) %>% 
  mutate(diff = abs(ytpred - label))  ## Calculate differences between predicted value and actural value

lmerrTs <- sum(lmtest$diff > 0.5) / len_test ## Calculate test data error ratio


#### LOOCV ####
CV <- c()
for (i in 1:len_train) {
  ## Using training data to get parameters
  fit_LR <- lm(label ~V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = biopsy_train[-i, ])
  ## Predict label with tunning data
  ytunpred <- predict(fit_LR, newdata = biopsy_train[i, ])
  ## Calculate differences between predicted value and actural value 
  gap <- abs(ytunpred - biopsy_train[i, ]$label)
  ## Get the one LOOCV value
  CV[i] <-  sum(gap > 0.5)
}

LOOCV_LR <- mean(CV)



########################################### KNN ##############################################
train_rv <- biopsy_train[, 1:9] ## Get predictors in dataset
test_rv <- biopsy_test[, 1:9]

ks <- seq(1, 228, by = 3) ## Initial K value
TRerr <- TSerr <- CVerr <- rep(0, length(ks)) ## Initial error ratio

for(i in 1:length(ks)){
  ## training error ratio
  tr_res <- knn(train_rv, train_rv, k=ks[i], cl = biopsy_train[,11])
  
  TRerr[i] = sum(tr_res != biopsy_train$label)/ len_train
  
  ## test error ratio
  ts_res <- knn(train_rv, test_rv, k=ks[i], cl = biopsy_train[,11])
  
  TSerr[i] = sum(ts_res != biopsy_test$label)/ len_test
  
  ## LOOCV
  cv_res <- knn.cv(train_rv, k=ks[i], cl = biopsy_train[,11])
  
  CVerr[i] = sum(cv_res != biopsy_train$label)/ len_train
  
}


## summarizing the results in a plot
plot(ks, TRerr, xlab="k", type="l", main="Performance of kNN",
     ylab="Error rate", ylim=c(0,0.1), col="black")
lines(ks, TSerr, type="l", col="red") 
lines(ks, CVerr, type="l", col="blue")
legend("topleft", legend = c("TRerr", "TSerr", "CVerr"), col=c("black", "red", "blue"), 
       border = "black",  lty = c(1, 1, 1))

##################################### Logistic Regression ####################################

fit_lgt <- glm(label ~V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = biopsy_train, family = binomial)
summary(fit_lgt)

#### Training Error ####
p <- predict(fit_lgt, type="response")

logisttrain <- 
  cbind(biopsy_train, p) %>% 
  select(label, p) %>% 
  mutate(diff = abs(p - label))  ## Calculate differences between predicted value and actural value

logisterrTr <- sum(logisttrain$diff > 0.5) / len_train ## Calculate training data error ratio


#### Testing Error ####
## Use coeffient to calculate y -- label
expsum <- fit_lgt$coefficients[1]
for (k in 1:9) {
  expsum <- expsum + fit_lgt$coefficients[k+1] * biopsy_test[,k]
}
pTs <- 1/(1 + exp(-expsum))

logisttest <- 
  cbind(biopsy_test, pTs) %>% 
  select(label, pTs) %>% 
  mutate(diff = abs(pTs - label))  ## Calculate differences between predicted value and actural value

logisterrTs <- sum(logisttest$diff > 0.5) / len_test ## Calculate test data error ratio


#### LOOCV ####
CV <- c()
for (i in 1:len_train) {
  ## Using training data to get parameters
  fit_lgt <- glm(label ~V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = biopsy_train[-i, ], family = binomial)
  ## Predict label with tunning data
  tunning <- biopsy_train[i, ]
  expsum <- fit_lgt$coefficients[1]
  for (k in 1:9) {
    expsum <- expsum + fit_lgt$coefficients[k+1] * tunning[,k]
  }
  pTu <- 1/(1 + exp(-expsum))
  ## Calculate differences between predicted value and actural value 
  gap <- abs(pTu - tunning$label)
  ## Get the one LOOCV value
  CV[i] <-  sum(gap > 0.5)
}

LOOCV_logist <- mean(CV)







############################################################################################
############# Randomly split the data into a training set and a test set####################  
############# containing about 1/3 and 2/3 of total observations respectively.##############
############################################################################################

#### Split data into a training set and a test set ####
set.seed(2345) ## Initial seeds
len_train <- round(len_bio / 3)
len_test <- round(len_bio / 3 * 2)
train_num <- sample(len_bio, len_train) ## Get the 1/3 dataset's indexes randomly.
test_num <- sample(len_bio, len_test) ## Get the 2/3 dataset's indexes randomly.

biopsy_train <- biopsy2[train_num, 2:12] ## Get training set
y_train <- biopsy2[train_num, 12]

biopsy_test <- biopsy2[test_num, 2:12] ## Get test set
y_test <- biopsy2[test_num, 12]



##################################### Linear Regression ######################################
fit_LR <- lm(label ~V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = biopsy_train)
summary(fit_LR)

#### Training Error ####
ypred <- predict(fit_LR)  ## Use coeffient to calculate y -- label

lmtrain <- 
  cbind(biopsy_train, ypred) %>% 
  select(label, ypred) %>% 
  mutate(diff = abs(ypred - label))  ## Calculate differences between predicted value and actural value

lmerrTr2 <- sum(lmtrain$diff > 0.5) / len_train ## Calculate training data error ratio


#### Testing Error ####
test_lmnew <- biopsy_test[, 1:9] ## Get predictors in dataset
ytpred <- predict(fit_LR, newdata = test_lmnew) ## Use coeffient to calculate y -- label

lmtest <- 
  cbind(biopsy_test, ytpred) %>% 
  select(label, ytpred) %>% 
  mutate(diff = abs(ytpred - label))  ## Calculate differences between predicted value and actural value

lmerrTs2 <- sum(lmtest$diff > 0.5) / len_test ## Calculate test data error ratio


#### LOOCV ####
CV <- c()
for (i in 1:len_train) {
  ## Using training data to get parameters
  fit_LR <- lm(label ~V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = biopsy_train[-i, ])
  ## Predict label with tunning data
  ytunpred <- predict(fit_LR, newdata = biopsy_train[i, ])
  ## Calculate differences between predicted value and actural value 
  gap <- abs(ytunpred - biopsy_train[i, ]$label)
  ## Get the one LOOCV value
  CV[i] <-  sum(gap > 0.5)
}

LOOCV_LR2 <- mean(CV)



########################################### KNN ##############################################
train_rv <- biopsy_train[, 1:9] ## Get predictors in dataset
test_rv <- biopsy_test[, 1:9]

ks <- seq(1, 228, by = 3) ## Initial K value
TRerr2 <- TSerr2 <- CVerr2 <- rep(0, length(ks)) ## Initial error ratio

for(i in 1:length(ks)){
  ## training error ratio
  tr_res <- knn(train_rv, train_rv, k=ks[i], cl = biopsy_train[,11])
  
  TRerr2[i] = sum(tr_res != biopsy_train$label)/ len_train
  
  ## test error ratio
  ts_res <- knn(train_rv, test_rv, k=ks[i], cl = biopsy_train[,11])
  
  TSerr2[i] = sum(ts_res != biopsy_test$label)/ len_test
  
  ## LOOCV
  cv_res <- knn.cv(train_rv, k=ks[i], cl = biopsy_train[,11])
  
  CVerr2[i] = sum(cv_res != biopsy_train$label)/ len_train
  
}


## summarizing the results in a plot
plot(ks, TRerr2, xlab="k", type="l", main="Performance of kNN",
     ylab="Error rate", ylim=c(0,0.4), col="black")
lines(ks, TSerr2, type="l", col="red") 
lines(ks, CVerr2, type="l", col="blue")
legend("topleft", legend = c("TRerr", "TSerr", "CVerr"), col=c("black", "red", "blue"), 
       border = "black",  lty = c(1, 1, 1))


##################################### Logistic Regression ####################################

fit_lgt <- glm(label ~V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = biopsy_train, family = binomial)
summary(fit_lgt)

#### Training Error ####
p <- predict(fit_lgt, type="response")

logisttrain <- 
  cbind(biopsy_train, p) %>% 
  select(label, p) %>% 
  mutate(diff = abs(p - label))  ## Calculate differences between predicted value and actural value

logisterrTr2 <- sum(logisttrain$diff > 0.5) / len_train ## Calculate training data error ratio


#### Testing Error ####
## Use coeffient to calculate y -- label
expsum <- fit_lgt$coefficients[1]
for (k in 1:9) {
  expsum <- expsum + fit_lgt$coefficients[k+1] * biopsy_test[,k]
}
pTs <- 1/(1 + exp(-expsum))

logisttest <- 
  cbind(biopsy_test, pTs) %>% 
  select(label, pTs) %>% 
  mutate(diff = abs(pTs - label))  ## Calculate differences between predicted value and actural value

logisterrTs2 <- sum(logisttest$diff > 0.5) / len_test ## Calculate test data error ratio


#### LOOCV ####
CV <- c()
for (i in 1:len_train) {
  ## Using training data to get parameters
  fit_lgt <- glm(label ~V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9, data = biopsy_train[-i, ], family = binomial)
  ## Predict label with tunning data
  tunning <- biopsy_train[i, ]
  expsum <- fit_lgt$coefficients[1]
  for (k in 1:9) {
    expsum <- expsum + fit_lgt$coefficients[k+1] * tunning[,k]
  }
  pTu <- 1/(1 + exp(-expsum))
  ## Calculate differences between predicted value and actural value 
  gap <- abs(pTu - tunning$label)
  ## Get the one LOOCV value
  CV[i] <-  sum(gap > 0.5)
}

LOOCV_logist2 <- mean(CV)

















