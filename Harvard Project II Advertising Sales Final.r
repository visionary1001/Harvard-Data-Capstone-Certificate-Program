required_packages <- c("tidyverse", "caret", "data.table", "ggplot2", "readr", "dplyr")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

#Installing & preparing for coded project Predictiive Modeling For Future Revenue [Business Understanding Phase]
#The business problem that is investigated or solved is to reach the business status of hyper-growth 
#through advertising from accurately predicting future revenue from past advertising performance. 
library(readr)

library(tidyverse)

#Putting dataset into notebook for coding [Data Understanding Phase]
Adv_data <- read_csv("C:/Users/Mtize/Downloads/Advertising_Budget_and_Sales.csv")




#Examinating Dataset
Adv_data



head(Adv_data)



str(Adv_data)

#Summary Stats of Advertising Variables
summary(Adv_data)



#[Data Explore/Prep phase]
pairs(Adv_data, main = "Scatterplot Matrix of Advertising and Sales")



#Corrleations Overview within the Dataset
cor(Adv_data)



library(ggplot2)

#Column Name Changes 
colnames(Adv_data)

colnames(Adv_data) <- c("Tv_Ad_Budget", "Radio_Ad_Budget", "Newspaper_Ad_Budget", "Sales")

colnames(Adv_data)

library(reshape2)



#Reshaping dataset for easier visuals to be created
long_data <- reshape2::melt(Adv_data, id.vars="Sales", 
                            measure.vars=c("Tv_Ad_Budget", "Radio_Ad_Budget", "Newspaper_Ad_Budget"),
                            variable.name="Advertising Type", value.name="Ad Budget")

#scatter plot to visualize the relationship between advertising budgets and sales revenue, 
#with points colored by the type of advertising
ggplot(long_data, aes(x=`Ad Budget`, y=Sales, color=`Advertising Type`)) +
  geom_point() +
  labs(title="Advertising Budgets vs Sales Revenue", 
       x="Advertising Budget", y="Sales Revenue") +
  theme_minimal() +
  scale_color_manual(values=c("blue", "red", "green"))



install.packages("corrplot")

library(corrplot)

numeric_data <- Adv_data[, c("Tv_Ad_Budget", "Radio_Ad_Budget", "Newspaper_Ad_Budget", "Sales")]

cor_matrix <- cor(numeric_data)

#Visual of Correlation matrix
corrplot(cor_matrix, method = "color", type = "upper", addCoef.col = "black", 
         tl.col = "black", tl.srt = 45, number.cex = 0.8)



#Test/Train Split 70/30 
#Set a seed for reproducibility
set.seed(123)

# Get the number of rows
n <- nrow(Adv_data)

# Choose 70% for training
train_indices <- sample(1:n, size = 0.7 * n)

# Spliting the data
train_data <- Adv_data[train_indices, ]
test_data <- Adv_data[-train_indices, ]

# Checking the split
nrow(train_data)  # Should be ~70% of data
nrow(test_data)   # Should be ~30% of data



#[Modeling Phase] ; Model 1 is Multiple Linear Regression & Its Evaluation 

model <- lm(Sales ~ Tv_Ad_Budget + Radio_Ad_Budget + Newspaper_Ad_Budget, data = train_data)

summary(model)



predictions <- predict(model, newdata = test_data)

# Now calculate RMSE
rmse_value <- sqrt(mean((test_data$Sales - predictions)^2))

# Print the RMSE
print(rmse_value)

mean(test_data$Sales)

#The RMSE value of 19.4 is quite large relative to the average sales of 32.13, 
#indicating that the model’s predictions are off by a significant margin. Specifically, the RMSE represents 
#about 60.4% of the average sales, which suggests a high error rate. This means the model’s predictions
#are not very accurate, as the errors are almost as large as the typical sales value. 
#To improve this, im consider going to add more relevant features, explore non-linear models like decision trees
#or random forests, or possibly some feature engineering techniques to capture more complex relationships in the data.



#Model 2 Random Forest & Its Evaluation 

install.packages("randomForest")

library(randomForest)



rf_model <- randomForest(Sales ~ Tv_Ad_Budget + Radio_Ad_Budget + Newspaper_Ad_Budget, 
                         data = train_data, ntree = 500)

print(rf_model)



rf_predictions <- predict(rf_model, newdata = test_data)

head(rf_predictions)



# Calculate RMSE for Random Forest model
rf_rmse <- sqrt(mean((test_data$Sales - rf_predictions)^2))

print(rf_rmse)



#Cross Validation 

install.packages("caret")

library(caret)

train_control <- trainControl(method = "cv", number = 5)  # 5-fold CV

rf_cv_model <- train(Sales ~ Tv_Ad_Budget + Radio_Ad_Budget + Newspaper_Ad_Budget, 
                     data = train_data, 
                     method = "rf", 
                     trControl = train_control, 
                     ntree = 1000)

print(rf_cv_model)




rf_cv_rmse <- rf_cv_model$results$RMSE
print(rf_cv_rmse)

print(rf_cv_model$results)



#Model 3 xgboost & its Evaluation



install.packages("xgboost")

library(xgboost)

train_matrix <- as.matrix(train_data[, c("Tv_Ad_Budget", "Radio_Ad_Budget", "Newspaper_Ad_Budget")])
test_matrix <- as.matrix(test_data[, c("Tv_Ad_Budget", "Radio_Ad_Budget", "Newspaper_Ad_Budget")])

train_label <- train_data$Sales
test_label <- test_data$Sales

library(xgboost)

xgb_model <- xgboost(
  data = train_matrix, 
  label = train_label, 
  nrounds = 100,              # number of trees
  objective = "reg:squarederror",  # regression task
  eta = 0.1,                  # learning rate
  max_depth = 6,              # depth of each tree
  verbose = 1                 # show training output
)

xgb_predictions <- predict(xgb_model, newdata = test_matrix)

xgb_rmse <- sqrt(mean((test_label - xgb_predictions)^2))
print(paste("XGBoost RMSE:", xgb_rmse))

# R-squared
xgb_r_squared <- 1 - (sum((test_label - xgb_predictions)^2) / sum((test_label - mean(test_label))^2))
print(paste("XGBoost R-squared:", xgb_r_squared))



#Model 4 Linear Model (TV & Radio Advertising Budgets only) & Its Evaluation



model_simple <- lm(Sales ~ Tv_Ad_Budget + Radio_Ad_Budget, data = train_data)

summary(model_simple)

simple_predictions <- predict(model_simple, newdata = test_data)



#Model 5 (Tv Ad Budget Only) & Its evaluation

model_tv_only <- lm(Sales ~ Tv_Ad_Budget, data = train_data)

summary(model_tv_only)

tv_predictions <- predict(model_tv_only, newdata = test_data)

tv_rmse <- sqrt(mean((test_data$Sales - tv_predictions)^2))
print(paste("TV Only Model RMSE:", tv_rmse))

tv_r2 <- 1 - (sum((test_data$Sales - tv_predictions)^2) / sum((test_data$Sales - mean(test_data$Sales))^2))
print(paste("TV Only Model R-squared:", tv_r2))

#[Deployment Phase] Deployment would come after making major changes to where the evaluation would show 
# a model performance high enough for it to be used to solve the business problem






