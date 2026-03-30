# ICT583 Assignment 2 - Data Science Application Project
# Dementia Risk Analysis
# Author: Kabita Joshi

# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(ggplot2)
library(readxl)
library(RColorBrewer)
library(reshape2)

# Set random seed
set.seed(12)


# 1. Load Dataset
data <- read_excel("dementia.xlsx")

# View structure
str(data)
colnames(data)

# 2. Data Preprocessing
colSums(is.na(data))  # Check missing values
data_clean <- na.omit(data)  # Remove rows with missing values

# Convert categorical variables
data_clean$Education_ID <- as.factor(data_clean$Education_ID)
data_clean$Mobility <- as.factor(data_clean$Mobility)
data_clean$Hyperlipidaemia <- as.factor(data_clean$Hyperlipidaemia)
data_clean$MMSE_class <- as.factor(data_clean$MMSE_class)

# 3. Exploratory Data Analysis (EDA)
summary(data_clean)

# --- Enhanced Correlation Matrix using ggplot2 ---
numeric_vars <- data_clean %>% select(where(is.numeric))
corr_matrix <- cor(numeric_vars)
melted_corr <- melt(corr_matrix)

ggplot(melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "red", high = "blue", mid = "white", midpoint = 0,
                       limit = c(-1, 1), space = "Lab", name = "Correlation") +
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  labs(title = "Correlation Matrix of Numeric Features", x = "", y = "")

# MMSE_class distribution
ggplot(data_clean, aes(x = MMSE_class)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of MMSE Class", x = "MMSE Class", y = "Count") +
  theme_minimal(base_size = 14)

# Age distribution
ggplot(data_clean, aes(x = Age)) +
  geom_histogram(binwidth = 2, fill = "lightgreen", color = "black") +
  labs(title = "Age Distribution", x = "Age", y = "Count") +
  theme_minimal(base_size = 14)

# 4. Data Splitting
trainIndex <- createDataPartition(data_clean$MMSE_class, p = 0.7, list = FALSE)
train_data <- data_clean[trainIndex, ]
test_data <- data_clean[-trainIndex, ]

# 5. Prediction Models

## Logistic Regression
log_model <- glm(MMSE_class ~ ., data = train_data, family = binomial)
log_pred_prob <- predict(log_model, test_data, type = "response")
log_pred <- ifelse(log_pred_prob > 0.5, "1", "0") %>% as.factor()

# Evaluation for Logistic Regression
log_cm <- confusionMatrix(log_pred, test_data$MMSE_class, positive = "1")
log_precision <- posPredValue(log_pred, test_data$MMSE_class, positive = "1")
log_recall <- sensitivity(log_pred, test_data$MMSE_class, positive = "1")
log_f1 <- 2 * log_precision * log_recall / (log_precision + log_recall)
log_roc <- roc(test_data$MMSE_class, log_pred_prob)



# 6. ROC Curves (Corrected: x = 1 - Specificity)

## ROC for Logistic Regression
ggroc(log_roc, legacy.axes = TRUE) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(title = "ROC Curve - Logistic Regression",
       x = "1 - Specificity (False Positive Rate)",
       y = "Sensitivity (True Positive Rate)") +
  xlim(0, 1) + ylim(0, 1) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5))




# 7. Performance Summary Table
results <- data.frame(
  Model = c("Logistic Regression"),
  Accuracy = c(log_cm$overall["Accuracy"]),
  Precision = c(log_precision),
  Recall = c(log_recall),
  F1_Score = c(log_f1),
  AUC = c(auc(log_roc))
)

print(results)
