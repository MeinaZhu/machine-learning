library(tidyverse)
library(tidymodels)
library(here)
library(vip) # a new package we're adding for variable importance measures
d <- read_csv("lab-4/data/ngsschat-processed-data.csv")

train_test_split <- initial_split(d, prop = .80)
data_train <- training(train_test_split)
kfcv <- vfold_cv(data_train, v=5) # again, we will use resampling

my_rec <- recipe(code ~ ., data = data_train) %>% 
    step_normalize(all_numeric_predictors()) %>%
    step_nzv(all_predictors())


# specify model
my_mod <-
    rand_forest(mtry = tune(), # this specifies that we'll take steps later to tune the model
                min_n = tune(),
                trees = tune()) %>%
    set_engine("ranger", importance = "impurity") %>%
    set_mode("classification")

# specify workflow
my_wf <-
    workflow() %>%
    add_model(my_mod) %>% 
    add_recipe(my_rec)

my_wf


# specify tuning grid
finalize(mtry(), data_train)
finalize(min_n(), data_train)
finalize(trees(), data_train)


tree_grid <- grid_max_entropy(mtry(range(1, 8)),
                              min_n(range(2, 40)),
                              trees(range(1, 600)), # how do you know how many trees you will use; you can try it out
                              size = 7)

tree_grid


# fit model with tune_grid
fitted_model <- my_wf %>% 
    tune_grid(
        resamples = kfcv,
        grid = tree_grid,
        metrics = metric_set(roc_auc, accuracy, kap, sensitivity, specificity, precision)
        )
        
        
# examine best set of tuning parameters; repeat?
show_best(fitted_model, n = 1000, metric = "accuracy")

# select best set of tuning parameters
best_tree <- fitted_model %>% select_best(metric = "accuracy")

# finalize workflow with best set of tuning parameters
final_wf <- my_wf %>% 
        finalize_workflow(best_tree)
        final_fit <- final_wf %>% 
            last_fit(train_test_split, metrics = metric_set(roc_auc, accuracy, kap, sensitivity, specificity, precision))
        

# fit stats
final_fit %>%
    collect_metrics()

# variable importance plot
final_fit %>% 
    pluck(".workflow", 1) %>%   
    extract_fit_parsnip() %>% 
    vip(num_features = 10)
        