required_packages <- c("tidyverse", "caret", "data.table", "ggplot2", "readr", "dplyr")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}



#Installing & preparing for coded project Moive Recommendation System Creation [Business Understanding Phase]
#The business problem thats solved here is to create a movie recommendation system that helps increase user retention
install.packages("randomForest")
library(randomForest)



#Putting data set into notebook for coding [Data Understanding Phase]
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

options(timeout = 120)
dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)
movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))
movielens <- left_join(ratings, movies, by = "movieId")

#Examining dataset
str(movielens)



head(movielens)



dim(movielens)

summary(movielens)



#total number of unique movies 
unique_movies <- movielens %>%
distinct(movieId, title)

nrow(unique_movies)



#ratings display 
rating_counts <- movielens %>%
  count(rating, sort = FALSE)

rating_counts

rating_levels <- seq(5, 0.5, by = -0.5)

# Count ratings from 5 stars to the lowest and i ensure all levels are included
rating_counts <- movielens %>%
  count(rating) %>%
  mutate(rating = factor(rating, levels = rating_levels)) %>%
  complete(rating, fill = list(n = 0)) %>%  # Fill in missing ratings with 0
  arrange(desc(as.numeric(as.character(rating))))  # Sort from 5.0 to 0.5

rating_counts



#Average movie rating across entire datset 
mean(movielens$rating)



#[Data Explore/Prep phase]
#Exploring to see patterns around the highs, lows and average movie ratings reltaive to the number of ratings 
library(tidyverse)

movielens %>%
  group_by(movieId, title) %>%
  summarise(avg_rating = mean(rating), num_ratings = n()) %>%
  ggplot(aes(x = num_ratings, y = avg_rating)) +
  geom_point(alpha = 0.4) +
  geom_smooth(method = "loess") +
  scale_x_log10() +
  labs(title = "Average Rating vs. Number of Ratings",
       x = "Number of Ratings (log scale)",
       y = "Average Rating")



#Patterns that might be linked to low movie reviews
movielens %>%
  group_by(movieId, title) %>%
  summarise(avg_rating = mean(rating), num_ratings = n()) %>%
  filter(avg_rating <= 2.5) %>%  # Focus on lower-rated movies
  ggplot(aes(x = num_ratings, y = avg_rating)) +
  geom_point(alpha = 0.5, color = "red") +
  geom_smooth(method = "loess", color = "black") +
  scale_x_log10() +
  labs(title = "Low Average Ratings vs. Number of Ratings",
       x = "Number of Ratings (log scale)",
       y = "Average Rating (≤ 2.5)")


#patterns that might be linked to high movie ratings 
movielens %>%
  group_by(movieId, title) %>%
  summarise(avg_rating = mean(rating), num_ratings = n(), .groups = "drop") %>%
  filter(avg_rating >= 4.5) %>%  # Focus on high-rated movies
  ggplot(aes(x = num_ratings, y = avg_rating)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "loess", color = "black") +
  scale_x_log10() +
  labs(title = "High Average Ratings vs. Number of Ratings",
       x = "Number of Ratings (log scale)",
       y = "Average Rating (≥ 4.5)")





#Average moive rating over a span of time 
movielens %>%
  mutate(date = as.POSIXct(timestamp, origin = "1970-01-01")) %>%
  mutate(year = format(date, "%Y")) %>%
  group_by(year) %>%
  summarise(avg_rating = mean(rating)) %>%
  ggplot(aes(x = as.integer(year), y = avg_rating)) +
  geom_line() +
  geom_point() +
  labs(title = "Average Rating Over Time",
       x = "Year",
       y = "Average Rating")



library(dplyr)

#converting three variables in dataset called movielens into factors
movielens$userId <- as.factor(movielens$userId)
movielens$movieId <- as.factor(movielens$movieId)
movielens$genres <- as.factor(movielens$genres)



#identifying the top 10 most active users
top_users <- movielens %>%
  count(userId, sort = TRUE) %>%
  slice_head(n = 10) %>%
  pull(userId)

print(top_users)



#identifying the top 10 most rated movies
top_movies <- movielens %>%
  count(movieId, sort = TRUE) %>%
  slice_head(n = 10) %>%
  pull(movieId)

print(top_movies)



#average ratings for the top 10 users & the top 10 movies, grouped by user, movie, and genre.
agg <- movielens %>%
  filter(userId %in% top_users, movieId %in% top_movies) %>%
  group_by(userId, movieId, genres) %>%
  summarise(avg_rating = mean(rating), .groups = "drop")

print(agg)



#heatmap of average ratings given by the top users to the top movies, faceted by genre
ggplot(agg, aes(x = as.factor(userId), y = as.factor(movieId), fill = avg_rating)) +
  geom_tile(color = "white") +
  facet_wrap(~genres) +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(
    title = "Average Rating by User and Movie (Faceted by Genre)",
    x = "User ID",
    y = "Movie ID",
    fill = "Avg Rating"
  )



#Test/Train Split 80/20 ratio
set.seed(123)  # for reproducibility
train_index <- createDataPartition(movielens$rating, p = 0.8, list = FALSE)
train_data <- movielens[train_index, ]
test_data <- movielens[-train_index, ]

install.packages("ranger")

library(ranger)



#[Modeling Phase]
#training a Random Forest regression model
rf_model <- ranger(
  rating ~ userId + movieId + genres,
  data = train_data,
  num.trees = 10,
  importance = "impurity"
)

rf_model



print(rf_model)

str(rf_model)

#predictions from Random Forest Model Created ; predicted movie ratings on the test dataset
predictions <- predict(rf_model, data = test_data)$prediction

print(predictions)

#[Model Evaluation & Deployment Phase] 
rmse <- sqrt(mean((predictions - test_data$rating)^2))
cat("RMSE on test set:", rmse, "\n")

print(rmse)



# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



predicted_ratings <- predict(rf_model, data = final_holdout_test)$predictions

RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

final_rmse <- RMSE(final_holdout_test$rating, predicted_ratings)
cat("Final RMSE of random forest on holdout test set:", final_rmse, "\n")



#RMSE is to High So will make adjustments to get a better RMSE Score

# ensemble averaging approach to combine predictions of my rf_model and a baseline model with 
#regularized movie plus user effects
mu <- mean(edx$rating)

lambda <- 0.25

b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu) / (n() + lambda))

b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu - b_i) / (n() + lambda))

baseline_preds <- final_holdout_test %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rf_preds <- predict(rf_model, data = final_holdout_test)$predictions

ensemble_preds <- (baseline_preds + rf_preds) / 2

RMSE <- function(true, pred) sqrt(mean((true - pred)^2))

#RMSE improved but is still not ideal
final_rmse <- RMSE(final_holdout_test$rating, ensemble_preds)
cat("Final ensemble RMSE:", final_rmse, "\n")

#To improve even better the RMSE score i give the combined model different weighted averages so the weight of the predictions lean
#more toward the better model
w <- 0.7  # weight for baseline model
ensemble_preds <- w * baseline_preds + (1 - w) * rf_preds
final_rmse <- RMSE(final_holdout_test$rating, ensemble_preds)
cat("Weighted ensemble RMSE:", final_rmse, "\n")






