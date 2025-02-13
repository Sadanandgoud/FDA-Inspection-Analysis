install.packages("fastDummies")
install.packages("randomForest")
install.packages("klaR")
install.packages("caret")
install.packages("nnet")
install.packages("smotefamily")
install.packages("ranger")

library(readxl)
library(readr)
library(plotly)
library(dplyr) 
library(tidyverse)  
library(ggplot2)
library(fastDummies)
library(randomForest)
library(nnet)
library(caret)  
library(smotefamily)
library(ranger)
library(ggplot2)

# K-means Libraries
library(cluster)
library(ggplot2)    
library(dplyr)      


options(scipen = 999)
FDA_Inspections = read_csv("~/Desktop/Capstone_Project/Dataset/FDA Inspections.csv")

# Data Cleaning

# Rename columns by replacing spaces with underscores
colnames(FDA_Inspections) = gsub(" ", "_", colnames(FDA_Inspections))
colnames(FDA_Inspections) = gsub("/", "_", colnames(FDA_Inspections))

colSums(is.na(FDA_Inspections))

# Converting Date_value types

FDA_Inspections$Inspection_End_Date = as.Date(FDA_Inspections$Inspection_End_Date, format = "%m/%d/%Y")

FDA_Inspections$Zip = as.factor(FDA_Inspections$Zip)

str(FDA_Inspections)

print(head(FDA_Inspections))




##---------------- Random Forest -----------------##

# Ensure categorical columns are factors
fda_data = FDA_Inspections


fda_data$Classification <- as.factor(fda_data$Classification)
fda_data$State <- as.factor(fda_data$State)
fda_data$Country_Area <- as.factor(fda_data$Country_Area)
fda_data$Posted_Citations <- as.factor(fda_data$Posted_Citations)
fda_data$Project_Area <- as.factor(fda_data$Project_Area)
fda_data$Product_Type <- as.factor(fda_data$Product_Type)

str(fda_data)


predictors <- c("State", "Country_Area", "Posted_Citations", "Project_Area", "Product_Type")
response <- "Classification"

# Split data into training and test sets
set.seed(42)
trainIndex <- createDataPartition(fda_data$Classification, p = 0.7, list = FALSE)
train_data <- fda_data[trainIndex, ]
test_data <- fda_data[-trainIndex, ]


# Create dummy variables for categorical columns for the entire training dataset
train_data_encoded <- model.matrix(~ State + Country_Area + Posted_Citations + Project_Area + Product_Type - 1, data = train_data)

# Combine the numeric data with the encoded categorical data
train_data_numeric <- data.frame(train_data_encoded, train_data[, sapply(train_data, is.numeric)])

# Include the target variable `Classification`
train_data_numeric$Classification <- train_data$Classification


# Subset the "OAI" class and the rest of the classes separately from the encoded data
oai_data <- train_data_numeric[train_data_numeric$Classification == "Official Action Indicated (OAI)", ]
other_data <- train_data_numeric[train_data_numeric$Classification != "Official Action Indicated (OAI)", ]

# Convert all columns to numeric after encoding
oai_data_numeric <- as.data.frame(lapply(oai_data, as.numeric))

# Increase SMOTE intensity for "OAI"
oai_smote <- SMOTE(X = oai_data_numeric[, -which(names(oai_data_numeric) == "Classification")], 
                   target = oai_data_numeric$Classification, 
                   K = 10, dup_size = 45)  # dup_size to generate more synthetic samples

# Extract the balanced data for "OAI"
balanced_oai_data <- oai_smote$data

# Ensure that the "Classification" column is included
if (!"Classification" %in% colnames(balanced_oai_data)) {
  balanced_oai_data$Classification <- as.factor(rep("Official Action Indicated (OAI)", nrow(balanced_oai_data)))
}

# Convert target column back to a factor
balanced_oai_data$Classification <- as.factor(balanced_oai_data$Classification)

# Ensure column names match between datasets
balanced_oai_data <- balanced_oai_data[, colnames(other_data)]

# Combine the datasets after ensuring consistency
balanced_train_data <- rbind(other_data, balanced_oai_data)

# Convert categorical columns in the test dataset to dummy variables
test_data_encoded <- model.matrix(~ State + Country_Area + Posted_Citations + Project_Area + Product_Type - 1, data = test_data)

# Combine the numeric data with the encoded categorical data
test_data_numeric <- data.frame(test_data_encoded, test_data[, sapply(test_data, is.numeric)])

# Include the target variable for evaluation purposes
test_data_numeric$Classification <- test_data$Classification

# Reorder columns to match the training data
test_data_numeric <- test_data_numeric[, colnames(balanced_train_data)]


# Tune Random Forest hyperparameters
best_mtry <- tuneRF(x = balanced_train_data[, -which(names(balanced_train_data) == "Classification")], 
                    y = balanced_train_data$Classification, 
                    stepFactor = 1.5, improve = 0.01, ntreeTry = 300)
best_mtry <- 103


# Using random forset ranger class for more quick predictions
rf_model_final <- ranger(Classification ~ ., data = balanced_train_data, num.trees = 300, best_mtry,
                         importance = "impurity", num.threads = parallel::detectCores())



# Predict on the test data using ranger
rf_predictions <- predict(rf_model_final, data = test_data_numeric)


# Extract the predicted classes (since `ranger` returns a list)
rf_predictions <- rf_predictions$predictions

# Evaluate using confusion matrix
rf_conf_matrix <- confusionMatrix(as.factor(rf_predictions), test_data_numeric$Classification)
print(rf_conf_matrix)


rf_predictions_saved <- predict(rf_model_loaded, data = test_data_numeric)

conf_matrix <- confusionMatrix(as.factor(rf_predictions), as.factor(test_data_numeric$Classification))
print(conf_matrix)


# Saving the model named 'rf_model_final'
saveRDS(rf_model_final, file = "~/Users/anishgkaushik/rf_model_final.rds")

# --- K-modes Clustering ---#

# Load dataset
df <- read.csv("~/Desktop/Capstone_Project/Dataset/FDA Inspections.csv", header = TRUE)

colnames(df) <- gsub("[ /]", "_", colnames(df))

print(head(df))

# Data Cleaning and Preparation
# Convert "Zip" to numeric, removing any NA values if introduced
df$Zip <- as.numeric(df$Zip)
df <- df[!is.na(df$Zip), ]
sum(is.na(df))

# Convert categorical columns to numeric using encoding
df$Country.Area <- as.numeric(as.factor(df$Country.Area))
df$State <- as.numeric(as.factor(df$State))
df$Posted.Citations <- as.numeric(as.factor(df$Posted.Citations))
df$Classification <- NULL  # Removing the target variable for unsupervised learning

str(df)

# Select relevant columns for clustering
df_cluster <- df %>% select(FEI.Number, Zip, Country.Area, State, Fiscal.Year)

# Check variance of each column
variances <- sapply(df_cluster, function(x) var(x, na.rm = TRUE))
print(variances)  # Print the variances to see if any column has zero variance

df_cluster <- df_cluster[, variances > 0]


# Scaling the data to normalize features
df_scaled <- scale(df_cluster)

sum(is.na(df_scaled))

wss <- sapply(1:10, function(k) {
  kmeans(df_scaled, centers = k, nstart = 10)$tot.withinss
})


# Plot to visualize the Elbow Method
plot(1:10, wss, type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters K",
     ylab = "Total within-clusters sum of squares")


# Based on the plot, let's say we choose k = 3
k <- 4


# Apply K-Means Clustering
set.seed(123)
kmeans_model <- kmeans(df_scaled, centers = k, nstart = 25)

# Add cluster results to the original dataset
df$Cluster <- kmeans_model$cluster

# Summary of the clusters
print(table(df$Cluster))

# Visualize Clustering Results
# Using Principal Component Analysis (PCA) to reduce dimensions for visualization
pca <- prcomp(df_scaled, center = TRUE, scale. = TRUE)
pca_data <- data.frame(pca$x[, 1:2], Cluster = factor(df$Cluster))

# Create an interactive scatter plot using plotly
plotly_plot <- plot_ly(
  pca_data, 
  x = ~PC1, 
  y = ~PC2, 
  type = 'scatter', 
  mode = 'markers', 
  color = ~Cluster, 
  marker = list(size = 6, opacity = 0.6)
) %>%
  layout(
    title = "K-Means Clustering of FDA Inspections (PCA Reduced)",
    xaxis = list(title = "Principal Component 1"),
    yaxis = list(title = "Principal Component 2"),
    showlegend = TRUE
  )

# Display the interactive plot
plotly_plot


# Run K-means clustering
kmeans_result <- kmeans(df_scaled, centers = 4, nstart = 10)
df_cluster$Cluster <- as.factor(kmeans_result$cluster)

# Scatter plot for clusters (using first two features for simplicity)
ggplot(df_cluster, aes(x = df_scaled[, 1], y = df_scaled[, 2], color = Cluster)) +
  geom_point(size = 3, alpha = 0.7) +
  theme_minimal() +
  labs(title = "K-means Clustering Results",
       x = "Feature 1",
       y = "Feature 2",
       color = "Cluster")

# Line plot for Elbow Method
ggplot(data.frame(K = 1:10, WSS = wss), aes(x = K, y = WSS)) +
  geom_line(color = "steelblue", size = 1.2) +
  geom_point(color = "darkred", size = 3) +
  theme_minimal() +
  labs(title = "Elbow Method for Optimal Clusters",
       x = "Number of Clusters (K)",
       y = "Within-Cluster Sum of Squares (WSS)")







