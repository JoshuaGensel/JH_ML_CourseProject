if(!dir.exists("./data")){dir.create("./data")}
if(!file.exists("./data/training.csv")&!exists("./data/testing.csv")){
    url_training = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    download.file(url_training, "./data/training.csv")
    url_testing = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download.file(url_testing, "./data/testing.csv")
}
training_full <- read.csv("./data/training.csv")
testing <- read.csv("./data/testing.csv")

set.seed(12031998)

library(tidyverse)
library(ggplot2)
library(caret)

inTrain <- createDataPartition(y=training_full$classe, p=.6, list = FALSE)
training <- training_full[inTrain,]
validation <- training_full[-inTrain,]

unique(training$classe)
nearZeroVar(training, saveMetrics = TRUE)
sapply(training, function(x) sum(is.na(x))/nrow(training))
sapply(training, function(x) (sum(is.na(x))/nrow(training)<0.1))
sapply(training, class)


training <- training %>%
    select(-nearZeroVar(training,names = TRUE)) %>%
    select(-c(cvtd_timestamp, X, raw_timestamp_part_1, raw_timestamp_part_2, num_window)) %>%
    mutate(classe = as.factor(classe))
training <- training[,sapply(training, function(x) (sum(is.na(x))/nrow(training)<0.1))]

training_pca <- prcomp(training[,2:52], scale. = TRUE)
library(factoextra)
fviz_eig(training_pca)
library(ggfortify)
p1 <- autoplot(training_pca, data = training, col = "classe")
p2 <- autoplot(training_pca, data = training, col = "user_name")
library(patchwork)
p1 + p2

folds <- createFolds(training$classe, k = 10, returnTrain = TRUE)


train_control <- trainControl(method = "cv",
                     number = 10,
                     search = 'grid',
                     classProbs = TRUE,
                     savePredictions = "final",
                     index = folds)

#random forest

tuneGrid = expand.grid(.mtry = c(1:50))
rf_model <- train(classe~.,
                      data = training,
                      method = "rf",
                      importance=TRUE,
                      tuneGrid = tuneGrid,
                      trControl = train_control,
                      ntree = 50)
rf_model$bestTune
#mtry = 8 optimal

mtryValue = expand.grid(.mtry = c(8))
rf_fit <- train(classe~.,
                  data = training,
                  method = "rf",
                  importance=TRUE,
                  tuneGrid = mtryValue,
                  trControl = train_control,
                  ntree = 200)
rf_fit$results
#almost max accuracy, maybe overfit


#gradient boosting

gbmGrid <- expand.grid(interaction.depth = 1:5,
                       n.trees = 100,
                       shrinkage = .1,
                       n.minobsinnode = c(10,20))
gbm_model <- train(classe ~ .,
                   data = training,
                   method = "gbm",
                   tuneGrid = gbmGrid,
                   verbose = FALSE,
                   trControl = train_control)
gbm_model$results[which.max(gbm_model$results$Accuracy),]
gbm_model$bestTune

#> gbm_model$bestTune
#n.trees interaction.depth shrinkage n.minobsinnode
#10     100                 5       0.1             20
gbmGrid2 <- expand.grid(interaction.depth = 5,
                       n.trees = 400,
                       shrinkage = .1,
                       n.minobsinnode = 20)
gbm_fit <- train(classe ~ .,
                   data = training,
                   method = "gbm",
                   tuneGrid = gbmGrid2,
                   verbose = FALSE,
                   trControl = train_control)
gbm_fit$results

confusionMatrix(predict(rf_fit, validation), as.factor(validation$classe))
confusionMatrix(predict(gbm_fit, validation), as.factor(validation$classe))
#random forest a little better, both >0.99 acc