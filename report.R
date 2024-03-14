install.packages("tidyverse")
install.packages("readr")
install.packages("gridExtra")
install.packages('mlr3')
install.packages('mlr3learners')
install.packages('mlr3tuning')
install.packages("ranger")
install.packages("kernlab")
install.packages("mlr3viz")
install.packages('ggplot2')
install.packages('paradox')
install.packages("knitr")
library(knitr)
library(paradox)
library(ggplot2)
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(kernlab)
library(mlr3viz)
library(tidyverse)
library(readr)
library(gridExtra)

set.seed(1)

data <- read_csv("./bank_personal_loan.csv", show_col_types = FALSE)
summary(data)  

p1 <- ggplot(data, aes(x = factor(Education), fill = factor(Personal.Loan))) + 
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(x = "Education Level", y = "Percentage", fill = "Personal Loan", 
       title = "Loan Acceptance by Education Level")

p2 <- ggplot(data, aes(x = factor(CreditCard), fill = factor(Personal.Loan))) + 
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(x = "Has Credit Card", y = "Percentage", fill = "Personal Loan", 
       title = "Loan Acceptance by Credit Card Ownership")

p3 <- ggplot(data, aes(x = factor(Online), fill = factor(Personal.Loan))) + 
  geom_bar(position = "fill") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(x = "Uses Online Banking", y = "Percentage", fill = "Personal Loan", 
       title = "Loan Acceptance by Online Banking Usage")

grid.arrange(p1, p2, p3, ncol = 3) 




data <- read_csv("./bank_personal_loan.csv", show_col_types = FALSE)
data$Personal.Loan <- as.factor(data$Personal.Loan)

task <- TaskClassif$new(id = "PersonalLoan", backend = data, target = "Personal.Loan")


cv <- rsmp("cv", folds = 5)
cv$instantiate(task)


learner_logreg <- lrn("classif.log_reg", predict_type = "prob")
resampling_logreg <- resample(task, learner_logreg, cv)
predictions_logreg <- resampling_logreg$prediction()


learner_rf <- lrn("classif.ranger", predict_type = "prob")
resampling_rf <- resample(task, learner_rf, cv)
predictions_rf <- resampling_rf$prediction()


learner_svm <- lrn("classif.svm", predict_type = "prob")
resampling_svm <- resample(task, learner_svm, cv)
predictions_svm <- resampling_svm$prediction()


autoplot(predictions_logreg, type = "roc") + labs(title = "ROC Curve for Logistic Regression")
autoplot(predictions_rf, type = "roc") + labs(title = "ROC Curve for Random Forest")
autoplot(predictions_svm, type = "roc") + labs(title = "ROC Curve for SVM")




param_set <- ParamSet$new(list(
  ParamInt$new("num.trees", lower = 100, upper = 1000),
  ParamInt$new("max.depth", lower = 5, upper = 30),
  ParamInt$new("min.node.size", lower = 1, upper = 10)
))

auto_tuner <- AutoTuner$new(
  learner = lrn("classif.ranger", predict_type = "response"),
  resampling = rsmp("cv", folds = 5), 
  measure = msr("classif.acc"), 
  search_space = param_set,
  terminator = trm("evals", n_evals = 20), 
  tuner = tnr("random_search") 
)

auto_tuner$train(task)

best_model <- auto_tuner$model
best_params <- auto_tuner$tuning_result$learner_param_vals

cat("best parameters：\n")

if (!is.null(best_params[[1]])) {
  params <- best_params[[1]]
  for (param_name in names(params)) {
    cat(sprintf("- %s: %s\n", param_name, params[[param_name]]))
  }
} else {
  cat("can't found best parameters.\n")
}



resampling_default <- resample(task, learner_rf, cv)
accuracy_default <- resampling_default$aggregate(msr("classif.acc"))[[1]]

learner_rf_best <- lrn("classif.ranger", predict_type = "prob",
                       num.trees = best_params[[1]]$num.trees,
                       num.threads = best_params[[1]]$num.threads,
                       max.depth = best_params[[1]]$max.depth,
                       min.node.size = best_params[[1]]$min.node.size)

resampling_best <- resample(task, learner_rf_best, cv)
accuracy_best <- resampling_best$aggregate(msr("classif.acc"))[[1]]

accuracy_data <- data.frame(
  Model = c("Default Parameters", "Best Parameters"),
  Accuracy = c(accuracy_default, accuracy_best)
)

kable(accuracy_data, format = "markdown", col.names = c("Model", "Accuracy"), caption = "Classification Accuracy Comparison")

