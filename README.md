# Practical Machine Learning Prediction Assignment Week 4
**Name : M. Fauzan Al Zamzami**

Purpose: Classify the measurement from Accelerometer into 5 Different Class

## Data Source
http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har

Training Data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

Testing Data: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

### Install the Required Package
``` 
install.packages("caret")
install.packages("rpart")
install.packages("randomForest") 
```

### Load the Required Package
``` 
library(caret)
library(rpart)
library(randomForest)
```

### Load the training and testing datasets
```
train <- read.csv("/content/pml-training.csv", na.strings = c("NA", "#DIV/0!", ""))
test <- read.csv("/content/pml-testing.csv", na.strings = c("NA", "#DIV/0!", ""))
```

### Remove columns with missing values and the first 7 columns that are not needed
```
test_clean <- names(test[, colSums(is.na(test)) == 0])[8:59]
clean_train <- train[, c(test_clean, "classe")]
clean_test <- test[, c(test_clean, "problem_id")]
```

### Split the dataset into training and testing sets (70% training, 30% testing)
```
set.seed(69)
inTrain <- createDataPartition(clean_train$classe, p = 0.7, list = FALSE)
training <- clean_train[inTrain, ]
testing <- clean_train[-inTrain, ]
```

### Linear Discriminant Analysis (LDA) model
```
lda_model <- train(classe ~ ., data = training, method = "lda")
set.seed(69)
predict_lda <- predict(lda_model, testing)
confusionMatrix(predict_lda, as.factor(testing$classe))
```

### LDA Result
**Accuracy = 69.02%**

**Kappa = 0.6081**

**Confusion Matrix and Statistics**
```
          Reference
Prediction    A    B    C    D    E
         A 1358  171  106   61   45
         B   35  708   94   46  184
         C  133  142  649  122   90
         D  142   55  146  693  109
         E    6   63   31   42  654
```

### Decision Tree model
```
decision_tree_model <- rpart(classe ~ ., data = training, method = "class")
set.seed(69)
predict_tree <- predict(decision_tree_model, testing, type = "class")
confusionMatrix(predict_tree, as.factor(testing$classe))
```

### Decision Tree Result
**Accuracy = 75.87%**

**Kappa = 0.6932**

**Confusion Matrix and Statistics**
```
          Reference
Prediction    A    B    C    D    E
         A 1557  253   17  106   38
         B   44  572   43   26   54
         C   41  206  892   97  132
         D   22   75   74  659   73
         E   10   33    0   76  785

```

### Random Forest Model
```
random_forest_model <- randomForest(as.factor(classe) ~ ., data = training, ntree = 500)
set.seed(69)
predict_rf <- predict(random_forest_model, testing, type = "class")
confusionMatrix(predict_rf, as.factor(testing$classe))
```

**Accuracy = 99.52%**

**Kappa = 0.994**

**Confusion Matrix and Statistics**
```
          Reference
Prediction    A    B    C    D    E
         A 1674    3    0    0    0
         B    0 1133    4    0    0
         C    0    3 1021   14    0
         D    0    0    1  948    1
         E    0    0    0    2 1081
```

# Predictions for 20 participants
```
predict_rf <- predict(random_forest_model, clean_test, type = "class")
predict_rf
```
# Final Result
```
B, A, B, A, A, E, D, B, A, A, B, C, B, A, E, E, A, B, B, B
```

# Conclusion
Based on result, Random Forest model has the highest accuracy among all the model, which is 99.52%.

Random Forest > Deciion Tree > LDA

