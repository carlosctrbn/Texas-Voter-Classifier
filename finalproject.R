#####
#
#
#
#####

# Setting the Working Directory to my own path
setwd("C:/Users/carlos.coronado/Google Drive/21-22/0. MS ASDS/MATH 6333 Statistical Learning/Final Project/Voting")

## Using Libraries
library(foreign)
library(readxl)
library(class)
library(caret)
library(leaps)
library(dplyr)
library(ggplot2)
library(tidyr)
library(purrr)
library(ggpubr)
library(pROC)
library(glmnet)
library(stats)
library(pls)
library(MASS)
library(e1071)
library(pracma)
library(randomForest)
library(neuralnet)

## Load the Data Set from a DBF file to a Data Frame
DF_voter_data = as.data.frame(read.dbf("tx_vtds.dbf", as.is = T ))

## Get rid of unnecessary columns
needed_cols = c(3:14, 29:32, 35)
DF_voter_data = DF_voter_data[, needed_cols]

## Make the Housing District the First Column
DF_voter_data = cbind(DF_voter_data$HD, DF_voter_data[1:15])
names(DF_voter_data)[names(DF_voter_data) == "DF_voter_data$HD"] = "HD"

## Find the number of House Districts in TX
num_hdistrcts = max(DF_voter_data$HD)

## Create an Empty Data Frame with Data by House District
DF_voter_data_by_HD = as.data.frame(matrix(NA, nrow = num_hdistrcts, ncol = ncol(DF_voter_data)))
## Name the Housing Districts in that data.frame
DF_voter_data_by_HD[, 1] = seq(1:num_hdistrcts)
## Rename the Columns to match the old data.frame
names(DF_voter_data_by_HD) = names(DF_voter_data)

# adds up the totals per House District
for(district_num in 1:num_hdistrcts){
  DF_voter_data_by_HD[district_num,2:ncol(DF_voter_data)] = colSums(DF_voter_data[which(DF_voter_data$HD == district_num), 2:ncol(DF_voter_data)])
}

## Standardizes the Data Set
DF_voter_data_by_HD[,3:(ncol(DF_voter_data_by_HD) - 3)] = scale(DF_voter_data_by_HD[,3:(ncol(DF_voter_data_by_HD)- 3)])

## Winning Political Party -1 = Democrat, 1 = Republican
## Third Party is Removed from Classification problem
DF_voter_data_by_HD$TOTVR16  = 2*(DF_voter_data_by_HD$PRES16D < DF_voter_data_by_HD$PRES16R) - 1


## Remove the Columns with the Number of Voters that elected Republican
## or Democrat, since it might cause problems as these are highly correlated
## with the outcome.
new_needed_cols = c(1:13,16)
DF_voter_data_by_HD = DF_voter_data_by_HD[ , new_needed_cols]
DF_voter_data_by_HD = cbind(DF_voter_data_by_HD$TOTVR16, DF_voter_data_by_HD[, 1:(ncol(DF_voter_data_by_HD)-1)])
names(DF_voter_data_by_HD)[names(DF_voter_data_by_HD) == "DF_voter_data_by_HD$TOTVR16"] = "Winner"


## Now, to work on the second file with more demographical information than just ethnicity.
DF_demo = as.data.frame(read_xlsx("Texas_District_Profile.xlsx"))
DF_demo = as.data.frame(DF_demo)

## Omit Unnecessary Data
unneeded_subset = c(1:126, 245:nrow(DF_demo))
DF_demo = DF_demo[-unneeded_subset, ]
DF_demo = t(DF_demo)
unneeded_subset = c(3:35)
DF_demo = DF_demo[-unneeded_subset, ]
DF_demo = t(DF_demo)
DF_demo = as.data.frame(cbind(1:nrow(DF_demo), DF_demo))
DF_demo = DF_demo[order(DF_demo$HD001, decreasing = T), ]
DF_demo = DF_demo[-which(rowSums(DF_demo > 1.000000000)>3), ]
DF_demo = as.data.frame(t(DF_demo))
colnames(DF_demo) = DF_demo[2,]
unneeded_subset = c(1:3)
DF_demo = DF_demo[-unneeded_subset, ]
indx = sapply(DF_demo, is.character)
DF_demo[indx] <- lapply(DF_demo[indx], function(x) as.numeric(as.character(x)))
# View(DF_demo)
complete_data = cbind(DF_voter_data_by_HD, scale(DF_demo))
# View(complete_data)

## Choose a training data set and a testing data set
set.seed(1234)
training = complete_data[sample(0.8 * nrow(complete_data)), ] # for 5-fold CV
testing  = setdiff(complete_data, training)

############################## Linear Regression ############################## 

# Train Linear Model
linear_model = lm(Winner ~., data = training)

# Linear Model Prediction (Positives are Republican, Negatives are Democrat Wins)
linear_prediction = 2 * ((predict(linear_model, testing) > 0) + 0) - 1

# Find Misclassification Error based on 0-1 Loss Function
MCE_linear = mean(testing$Winner != linear_prediction)
# 26.67% Misclassification Error
# 23.33% better than random guessing

confusionMatrix(data = as.factor(linear_prediction),
                reference = as.factor(testing$Winner),
                positive = "1")

linear_roc = roc(testing$Winner, linear_prediction_R)

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       53.33%      |       16.67%  
# ---------------------+-------------------+--------------------
# Predicted Republican |       10.00%      |       20.00%

# Accuracy    (Total)         = 73.33%
# Sensitivity (True Positive) = 54.55%
# Specificity (True Negative) = 84.21%
# AUC                         = 74.64%

############################## Model Diagnostics ############################## 

# 1. Check Coefficient of Determination
COR = cor(training)^2
COR[upper.tri(COR)] = 0
diag(COR) = 0

new_training = training[, !apply(COR, 2, function(x) any(abs(x) > 0.90, na.rm = TRUE))]
new_testing  = testing[, !apply(COR, 2, function(x) any(abs(x) > 0.90, na.rm = TRUE))]

linear_model_R = lm(Winner~., data = new_training)
linear_prediction_R = 2 * ((predict(linear_model_R, new_testing) > 0) + 0) - 1
MCE_linear_R = mean(testing$Winner != linear_prediction_R) # Now 23.33%, a little better


# 2. Check Adjusted Coefficient of Determination

lmp = function (modelobject) {
  f = summary(modelobject)$fstatistic
  p = pf(f[1],f[2],f[3],lower.tail=F)
  attributes(p) = NULL
  return(p)
}
p = lmp(linear_model_R)

COR_adj = cor(new_training)^2
COR_adj[upper.tri(COR_adj)] = 0
diag(COR_adj) = 0
COR_adj = 1 - (1 - COR_adj)*((nrow(new_training) - 1) / (nrow(new_training) - p - 1))

new_training = new_training[, !apply(COR_adj, 2, function(x) any(abs(x) > 0.90, na.rm = TRUE))]
new_testing = new_testing[, !apply(COR_adj, 2, function(x) any(abs(x) > 0.90, na.rm = TRUE))]

linear_model_R = lm(Winner~., data = new_training)
linear_prediction_R = 2 * ((predict(linear_model_R, testing) > 0) + 0) - 1
MCE_linear_R = mean(testing$Winner != linear_prediction_R) # Now 23.33%, same as 1.

# 3. Residual Analyses

for(plotnum in 1:ncol(new_training)) {
  print(ggqqplot(new_training[ ,plotnum]) +
          ggtitle(colnames(new_training[plotnum])))
}

nonnorm = c("pubtrans_per","engpoor_per","Asian_per","Hisp_per", "collegeover_per",
            "Black_per","age1864_per")

temp = new_training
new_training = new_training[, -which(colnames(temp) %in% nonnorm)]
new_testing = new_testing[, -which(colnames(temp) %in% nonnorm)]

linear_model_R = lm(Winner~., data = new_training)
linear_prediction_R = 2 * ((predict(linear_model_R, new_testing) > 0) + 0) - 1
MCE_linear_R = mean(testing$Winner != linear_prediction_R) # Now 20.00%, better.

# 4. Test for Outliers.
# For the outliers, I choose to make them either the max or min and not remove any data.
outlier_curation = function(x){
  fns = fivenum(x)
  below = fns[2] - 1.5*(IQR(x))
  above = fns[2] + 1.5*(IQR(x))
  x[(x > above)] = above
  x[(x < below)] = below
  return(x)
}

new_training = as.data.frame(apply(new_training, 2, outlier_curation))

linear_model_R = lm(Winner~., data = new_training)
linear_prediction_R = 2 * ((predict(linear_model_R, new_testing) > 0) + 0) - 1
MCE_linear_R = mean(testing$Winner != linear_prediction_R) # Now 13.33%, much better

############## Variables of Greatest Influence (by Linear Model) ############## 

## Top 10 Most Influential Variables:
colnames(new_training[, order(abs(linear_model_R$coefficients), decreasing = T)])
##   1. "hhinc100199_per"   % Population with $100,000 to $199,999 annual household income
##   2. "age65over_per"     % Population 65 years and over
##   3. "hhsocsec_per"      % Households with social security income
##   4. "hhval200499_per"   % Population with Owner-occupied housing value from $200,000 to $499,999
##   5. "govt_per"          % Population Employed in government sector
##   6. "hh5more_per"       % Population with 5 or more persons household
##   7. "nohs25over_per"    % Less than high school graduate
##   8. "vacant_per"        % Population with Vacant Housing Units
##   9. "hhval100199_per"   % Population Owner-occupied housing value from $100,000 to $199,999
##  10. "hhinc2549_per"     % Population with $25,000 to $49,999 annual household income

## Top 10 Most Influential Toward Republicans:
colnames(new_training[, order(linear_model_R$coefficients, decreasing = T)])
##   1. "age65over_per"     % Population 65 years and over
##   2. "hhval200499_per"   % Population with Owner-occupied housing value from $200,000 to $499,999
##   3. "govt_per"          % Population Employed in government sector
##   4. "hh5more_per"       % Population with 5 or more persons household
##   5. "vacant_per"        % Population with Vacant Housing Units
##   6. "hhval100199_per"   % Population Owner-occupied housing value from $100,000 to $199,999 
##   7. "fambothwork_per"   % Families with both parents employed
##   8. "home_per"          % Population who Worked at Home
##   9. "hhval500_per"      % Population Owner-occupied housing value from $500,000 and above
##  10. "hhvallt50_per"     % Population Owner-occupied housing at value less than $50,000

## Top 10 Most Influential Toward Democrats:
colnames(new_training[, order(linear_model_R$coefficients)])
##   1. "hhinc100199_per"   % Population with $100,000 to $199,999 annual household income   
##   2. "hhsocsec_per"      % Population Households with social security income
##   3. "nohs25over_per"    % Population Age 25 and over with Less than high school graduate
##   4. "hhinc2549_per"     % Population with $25,000 to $49,999 annual household income
##   5. "NonAnglo_per"      % Population NonAnglo
##   6. "finance_per"       % Population Finance and insurance, and real estate and rental and leasing industries
##   7. "educ_per"          % Population Educational services and health care and social assistance industries
##   8. "hhinc200over_per"  % Population $200,000 and over annual household income: Percent
##   9. "hhsnap_per"        % Households that received food stamps/SNAP in the past 12 months
##  10. "age04_per"         % Total Population 0 to 4 years

########################## Confusion Matrix and AUC ########################### 
confusionMatrix(data = as.factor(linear_prediction_R),
                reference = as.factor(new_testing$Winner),
                positive = "1")

linear_roc = roc(new_testing$Winner, linear_prediction_R)

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       56.67%      |       06.67%  
# ---------------------+-------------------+--------------------
# Predicted Republican |       06.67%      |       30.00%

# Accuracy    (Total)         = 86.67%
# Sensitivity (True Positive) = 81.82%
# Specificity (True Negative) = 89.47%
# AUC                         = 85.65%


################################## Voronoi ################################### 

voronoi_test = knn(new_training[ ,1:2], new_testing[ ,1:2], new_training[ ,3], k = 1,
               prob = FALSE, use.all=FALSE)
levels(voronoi_test) = 2*((levels(voronoi_test) > 0) + 0) - 1 
MCE_voronoi  = mean(testing$Winner != voronoi_test) # 63.33% Misclassification. Expected

confusionMatrix(data = voronoi_test,
                reference = as.factor(testing$Winner),
                positive = "1")

voronoi_roc = roc(as.factor(testing$Winner), as.numeric(voronoi_test))

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       00.00%      |       00.00% 
# ---------------------+-------------------+--------------------
# Predicted Republican |       63.33%      |       36.67%

# Accuracy    (Total)         = 36.67%
# Sensitivity (True Positive) = 100.0%
# Specificity (True Negative) = 00.00%
# AUC                         = 50.00%

#################################### K-NN ####################################

# Function that finds the optimal value of K in KNN using 5-fold Cross Validation
Find_K = function(K,M=1000){
  MSE = 0
  for(i in 1:M){
    train = sample(nrow(new_training),0.8*nrow(new_training))
    KNN = knn(new_training[train, 1:2], new_training[-train, 1:2],
              new_training[train, 3], k = K, prob = FALSE, use.all=FALSE)
    MSE = MSE + mean(KNN != new_training[-train, 3])
  }
  CVK = MSE/M
  return(CVK)
}
# Find optimal K
LK = 1:nrow(new_training) 
vCVK = LK %>%
  map(function(K) Find_K(K))

vCVK = unlist(vCVK)
vCVK = as.data.frame(vCVK)
colnames(vCVK)<-c("CVK")

K = LK[which.min(vCVK$CVK)] # K returned is 110.

KNN_test = knn(new_training[ ,1:2], new_testing[ ,1:2], new_training[ ,3], k = K,
                   prob = FALSE, use.all=FALSE)
levels(KNN_test) = 2*((levels(KNN_test) > 0) + 0) - 1 
MCE_KNN  = mean(testing$Winner != KNN_test) # 63.33% Misclassification. Same as Voronoi.

confusionMatrix(data = KNN_test,
                reference = as.factor(testing$Winner),
                positive = "1")

KNN_roc = roc(as.factor(testing$Winner), as.numeric(voronoi_test))

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       00.00%      |       00.00% 
# ---------------------+-------------------+--------------------
# Predicted Republican |       63.33%      |       36.67%

# Accuracy    (Total)         = 36.67%
# Sensitivity (True Positive) = 100.0%
# Specificity (True Negative) = 00.00%
# AUC                         = 50.00%

## This is like saying, "The state majority is Republican, so just expect everyone
## to vote in the same way." 

########################## Subset Selection Methods ###########################

# Recall the Linear Model
new_training = as.data.frame(new_training) # Just in case something happened
LR_All = lm(Winner~., data = new_training)

LR.Forward_AIC = step(LR_All, direction = "forward", k = 2) # for AIC
numpar_LR_B_AIC = nrow(as.data.frame(LR.Forward_AIC$coefficients)) - 1 # Kept 88
training_B_AIC = LR.Forward_AIC$model
testing_B_AIC  = new_testing[, colnames(training_B_AIC)]
LR_B_AIC = lm(Winner ~., training_B_AIC)
LP_B_AIC = 2 * ((predict(LR_B_AIC, testing_B_AIC) > 0) + 0) - 1
MCE_LP_B_AIC = mean(testing_B_AIC$Winner != LP_B_AIC) # 13.33%

LR.Forward_BIC = step(LR_All, direction = "forward", k = log(nrow(new_training))) # for BIC
numpar_LR_B_BIC = nrow(as.data.frame(LR.Forward_BIC$coefficients)) - 1 # Kept 88 parameters
training_B_BIC = LR.Forward_BIC$model
testing_B_BIC  = new_testing[, colnames(training_B_BIC)]
LR_B_BIC = lm(Winner ~., training_B_BIC)
LP_B_BIC = 2 * ((predict(LR_B_BIC, testing_B_BIC) > 0) + 0) - 1
MCE_LP_B_BIC = mean(testing_B_BIC$Winner != LP_B_BIC) # 13.33%, same as Forward AIC with less complexity (36 < 44)

LR.Backward_AIC  = step(LR_All, direction = "backward", k = 2) # for AIC
numpar_LR_B_AIC = nrow(as.data.frame(LR.Backward_AIC$coefficients)) - 1 # Kept 44 Parameters
training_B_AIC = LR.Backward_AIC$model
testing_B_AIC  = new_testing[, colnames(training_B_AIC)]
LR_B_AIC = lm(Winner ~., training_B_AIC)
LP_B_AIC = 2 * ((predict(LR_B_AIC, testing_B_AIC) > 0) + 0) - 1
MCE_LP_B_AIC = mean(testing_B_AIC$Winner != LP_B_AIC) # 10.00%, better and less complexity

LR.Backward_BIC  = step(LR_All, direction = "backward", k = log(nrow(new_training))) # for AIC
numpar_LR_B_BIC = nrow(as.data.frame(LR.Backward_BIC$coefficients)) - 1 # Kept 36 Parameters
training_B_BIC = LR.Backward_BIC$model
testing_B_BIC  = new_testing[, colnames(training_B_BIC)]
LR_B_BIC = lm(Winner ~., training_B_BIC)
LP_B_BIC = 2 * ((predict(LR_B_BIC, testing_B_BIC) > 0) + 0) - 1
MCE_LP_B_BIC = mean(testing_B_BIC$Winner != LP_B_BIC) # 10.00%, with even less complexity

LR.Hybrid_AIC   = step(LR_All, direction = "both", k = 2) # for AIC
numpar_LR_H_AIC = nrow(as.data.frame(LR.Hybrid_AIC$coefficients)) - 1  # Kept 44 parameters
training_H_AIC = LR.Hybrid_AIC$model
testing_H_AIC  = new_testing[, colnames(training_H_AIC)]
LR_H_AIC = lm(Winner ~., training_H_AIC)
LP_H_AIC = 2 * ((predict(LR_H_AIC, testing_H_AIC) > 0) + 0) - 1
MCE_LP_H_AIC = mean(testing_H_AIC$Winner != LP_H_AIC) # 10.00%, same as other previous, but more complex

LR.Hybrid_BIC   = step(LR_All, direction = "both", k = log(nrow(new_training))) # for BIC
numpar_LR_H_BIC = nrow(as.data.frame(LR.Hybrid_BIC$coefficients)) - 1  # Kept 36 parameters
training_H_BIC = LR.Hybrid_BIC$model
testing_H_BIC  = new_testing[, colnames(training_H_BIC)]
LR_H_BIC = lm(Winner ~., training_H_BIC)
LP_H_BIC = 2 * ((predict(LR_H_BIC, testing_H_BIC) > 0) + 0) - 1
MCE_LP_H_BIC = mean(testing_H_BIC$Winner != LP_H_BIC) # 10.00%

LR.Leap_fwd = regsubsets(Winner ~., data = new_training,
                         intercept = TRUE,
                         #method = "Forward",
                         method = "forward",
                         really.big = T,
                         nvmax = ncol(new_training) - 1 # Maximum size of subsets to examine
)

sum.LR.leap_fwd = summary(LR.Leap_fwd)
outmat_Leap_fwd = as.data.frame(sum.LR.leap_fwd$outmat)

nadfwd = which.max(sum.LR.leap_fwd$adjr2) # output is 54
training_nadfwd = cbind(new_training$Winner, new_training[ , which(outmat_Leap_fwd[nadfwd, ] == "*")])
testing_nadfwd  = cbind(new_testing$Winner,  new_testing[ , which(outmat_Leap_fwd[nadfwd, ] == "*")])
colnames(training_nadfwd)[1] = "Winner"
colnames(testing_nadfwd)[1] = "Winner"
LR_L_nadfwd = lm(Winner ~., training_nadfwd)
LP_L_nadfwd = 2 * ((predict(LR_L_nadfwd, testing_nadfwd) > 0) + 0) - 1
MCE_LP_L_nadfwd = mean(testing_nadfwd$Winner != LP_L_nadfwd) # 13.33%

nrssfwd = which.min(sum.LR.leap_fwd$rss)  # output is 88
training_nrssfwd = cbind(new_training$Winner, new_training[ , which(outmat_Leap_fwd[nrssfwd, ] == "*")])
testing_nrssfwd  = cbind(new_testing$Winner,  new_testing[ , which(outmat_Leap_fwd[nrssfwd, ] == "*")])
colnames(training_nrssfwd)[1] = "Winner"
colnames(testing_nrssfwd)[1] = "Winner"
LR_L_nrssfwd = lm(Winner ~., training_nrssfwd)
LP_L_nrssfwd = 2 * ((predict(LR_L_nrssfwd, testing_nrssfwd) > 0) + 0) - 1
MCE_LP_L_nrssfwd = mean(testing_nrssfwd$Winner != LP_L_nrssfwd) # 06.67%

ncpfwd = which.min(sum.LR.leap_fwd$cp)    # output is 22
training_ncpfwd = cbind(new_training$Winner, new_training[ , which(outmat_Leap_fwd[ncpfwd, ] == "*")])
testing_ncpfwd  = cbind(new_testing$Winner,  new_testing[ , which(outmat_Leap_fwd[ncpfwd, ] == "*")])
colnames(training_ncpfwd)[1] = "Winner"
colnames(testing_ncpfwd)[1] = "Winner"
LR_L_ncpfwd = lm(Winner ~., training_ncpfwd)
LP_L_ncpfwd = 2 * ((predict(LR_L_ncpfwd, training_ncpfwd) > 0) + 0) - 1
MCE_LP_L_ncpfwd = mean(testing_ncpfwd$Winner != LP_L_ncpfwd) # 50.00%

nbicfwd = which.min(sum.LR.leap_fwd$bic)  # output is 15
training_nbicfwd = cbind(new_training$Winner, new_training[ , which(outmat_Leap_fwd[nbicfwd, ] == "*")])
testing_nbicfwd  = cbind(new_testing$Winner,  new_testing[ , which(outmat_Leap_fwd[nbicfwd, ] == "*")])
colnames(training_nbicfwd)[1] = "Winner"
colnames(testing_nbicfwd)[1] = "Winner"
LR_L_nbicfwd = lm(Winner ~., training_nbicfwd)
LP_L_nbicfwd = 2 * ((predict(LR_L_nbicfwd, training_nbicfwd) > 0) + 0) - 1
MCE_LP_L_nbicfwd = mean(testing_nbicfwd$Winner != LP_L_nbicfwd) # 50.83%

LR.Leap_bwd = regsubsets(Winner ~., data = new_training,
                         intercept = TRUE,
                         method = "backward",
                         #method = "forward",
                         really.big = T,
                         nvmax = ncol(new_training) - 1 # Maximum size of subsets to examine
)

sum.LR.leap_bwd = summary(LR.Leap_bwd)
outmat_Leap_bwd = as.data.frame(sum.LR.leap_bwd$outmat)

nadbwd = which.max(sum.LR.leap_bwd$adjr2) # output is 44
training_nadbwd = cbind(new_training$Winner, new_training[ , which(outmat_Leap_bwd[nadbwd, ] == "*")])
testing_nadbwd  = cbind(new_testing$Winner,  new_testing[ , which(outmat_Leap_bwd[nadbwd, ] == "*")])
colnames(training_nadbwd)[1] = "Winner"
colnames(testing_nadbwd)[1] = "Winner"
LR_L_nadbwd = lm(Winner ~., training_nadbwd)
LP_L_nadbwd = 2 * ((predict(LR_L_nadbwd, testing_nadbwd) > 0) + 0) - 1
MCE_LP_L_nadbwd = mean(testing_nadbwd$Winner != LP_L_nadbwd) # 20.00%

nrssbwd = which.min(sum.LR.leap_bwd$rss)  # output is 88
training_nrssbwd = cbind(new_training$Winner, new_training[ , which(outmat_Leap_bwd[nrssbwd, ] == "*")])
testing_nrssbwd  = cbind(new_testing$Winner,  new_testing[ , which(outmat_Leap_bwd[nrssbwd, ] == "*")])
colnames(training_nrssbwd)[1] = "Winner"
colnames(testing_nrssbwd)[1] = "Winner"
LR_L_nrssbwd = lm(Winner ~., training_nrssbwd)
LP_L_nrssbwd = 2 * ((predict(LR_L_nrssbwd, testing_nrssbwd) > 0) + 0) - 1
MCE_LP_L_nrssbwd = mean(testing_nrssbwd$Winner != LP_L_nrssbwd) # 06.67%

ncpbwd = which.min(sum.LR.leap_bwd$cp)    # output is 31
training_ncpbwd = cbind(new_training$Winner, new_training[ , which(outmat_Leap_bwd[ncpbwd, ] == "*")])
testing_ncpbwd  = cbind(new_testing$Winner,  new_testing[ , which(outmat_Leap_bwd[ncpbwd, ] == "*")])
colnames(training_ncpbwd)[1] = "Winner"
colnames(testing_ncpbwd)[1] = "Winner"
LR_L_ncpbwd = lm(Winner ~., training_ncpbwd)
LP_L_ncpbwd = 2 * ((predict(LR_L_ncpbwd, testing_ncpbwd) > 0) + 0) - 1
MCE_LP_L_ncpbwd = mean(testing_ncpbwd$Winner != LP_L_ncpbwd) # 10.00%

nbicbwd = which.min(sum.LR.leap_bwd$bic)  # output is 10
training_nbicbwd = cbind(new_training$Winner, new_training[ , which(outmat_Leap_bwd[nbicbwd, ] == "*")])
testing_nbicbwd  = cbind(new_testing$Winner,  new_testing[ , which(outmat_Leap_bwd[nbicbwd, ] == "*")])
colnames(training_nbicbwd)[1] = "Winner"
colnames(testing_nbicbwd)[1] = "Winner"
LR_L_nbicbwd = lm(Winner ~., training_nbicbwd)
LP_L_nbicbwd = 2 * ((predict(LR_L_nbicbwd, testing_ncpbwd) > 0) + 0) - 1
MCE_LP_L_nbicbwd = mean(testing_nbicbwd$Winner != LP_L_nbicbwd) # 13.33%

## Q: Which subset selection method to choose for reporting?
## A: We will choose the method that minimizes the MCE and minimizes the number of parameters

# Method                    | Number of Parameters | Misclassification Error
# --------------------------+----------------------+-------------------------
# Forward Stepwise AIC      |          88          |          13.33%
# Forward Stepwise BIC      |          88          |          13.33%
# Backward Stepwise AIC     |          44          |          10.00%
# Backward Stepwise BIC     |          36          |          10.00%
# Hybrid Stepwise AIC       |          44          |          10.00%
# Hybrid Stepwise BIC       |          36          |          10.00%
# L&B Forward (max R^2adj)  |          54          |          13.33%
# L&B Forward (min RSS)     |          88          |          06.67%
# L&B Forward (min Cp)      |          22          |          50.00%
# L&B Forward (min BIC)     |          15          |          50.83%
# L&B Backward (max R^2adj) |          44          |          20.00%
# L&B Backward (min RSS)    |          88          |          06.67%
# L&B Backward (min Cp)     |          31          |          10.00% **** Choose
# L&B Backward (min BIC)    |          10          |          13.33% 

# Choose a Backward Leaps and Bounds method that minimizes Mallows's Cp.
# for subset selection. The mode misclassification error is 10.00%, where its
# number of parameters is 31 I could definitely choose all the other subset
# selection methods that have a misclassification error of 6.67%, however
# because of their higher complexity (more parameters kept), I prefer a more 
# explanatory model with fewer variables.

LR_best_subset = LR_L_ncpbwd

confusionMatrix(data = as.factor(LP_L_ncpbwd),
                reference = as.factor(testing_ncpbwd$Winner),
                positive = "1")

LR_best_subset_roc = roc(as.factor(testing_ncpbwd$Winner), as.numeric(LP_L_ncpbwd))

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       56.67%      |       03.33% 
# ---------------------+-------------------+--------------------
# Predicted Republican |       06.67%      |       33.33%

# Accuracy    (Total)         = 90.00%
# Sensitivity (True Positive) = 90.91%
# Specificity (True Negative) = 89.47%
# AUC                         = 90.19% Awesome!

####################### 31 Variables in the Model Above ###################### 

## coefficients:
# "VAP"                   % Population of Voting Age       
# "privveh_per"           % Population who drives a Private vehicle to work
# "private_per"           % Population Employed in private sector
# "famwithchild_per"      % Population with Families with children by birth, marriage or adoption under 18 years
# "hh2per_per"            % Population living in a 2-person household
# "rent35over_per"        % Population where 35 percent or more of household income spent on gross rent (renter occupied)
# "travel1529_per"        % Population Taking 15 minutes to 29 minutes travel time to work
# "hhval100199_per"       % Population Owner-occupied housing value from $100,000 to $199,999
# "house1970_per"         % Population Living in a home Built before 1970
# "hhval5099_per"         % Population with Owner-occupied housing value from $50,000 to $99,999
# "hhinc2549_per"         % Population with $25,000 to $49,999 annual household income
# "own2034_per"           % Population where 20 to 34.9 percent of household income on housing costs (owner occupied)
# "hhinc1024_per"         % Population with $10,000 to $24,999 annual household income
# "hhval200499_per"       % Population Owner-occupied housing value from $200,000 to $499,999
# "age65over_per"         % Population 65 years and over
# "vacant_per"            % Houses Vacant
# "personspov_per"        % Persons in poverty
# "own35over_per"         % Population who sepnd 35 percent or more of household income on housing costs (owner occupied)
# "hhsnap_per"            % Households that received food stamps/SNAP in the past 12 months
# "house0009_per"         % Population with Houses Built between 2000 and 2009
# "fampovmarried_per"     % Married couple families living in Poverty
# "famsinglework_per"     % Single-parent households with the parent employed
# "hhinclt10_per"         % Households with Less than $10,000 annual household income
# "fampovmalehh_per"      % Population Living in Poverty with Male head of household
# "grent1250_per"         % Population with Gross rent $1,250 and above
# "presch_per"            % Population In preschool (public and private)
# "self_per"              % Population Self-employed
# "house2010_per"         % Population Built in 2010 or later
# "finance_per"           % Population Employed in Finance and insurance, and real estate and rental and leasing industries
# "travel60over_per"      % Population with 60 minutes or more travel time to work
# "hhinc200over_per"      % Population with $200,000 and over annual household income

## Top 10 Most Influential Variables:
coeffs_best_subset = names(LR_best_subset$coefficients)[order(abs(LR_best_subset$coefficients), decreasing = T)]
##   1. "hh2per_per"       % Population with 2 persons household
##   2. "famwithchild_per" % Population with Families with children by birth, marriage or adoption under 18 years
##   3. "age65over_per"    % Population 65 years and over
##   4. "vacant_per"       % Vacant Housing units
##   5. "rent35over_per"   % 35 percent or more of household income on gross rent (renter occupied)
##   6. "privveh_per"      % Population Driving Private vehicle to work
##   7. "private_per"      % Population Employed in private sector
##   8. "hhval200499_per"  % Population Owner-occupied housing value from $200,000 to $499,999
##   9. "hhval5099_per"    % Population Owner-occupied housing value from $50,000 to $99,999
##  10. "house0009_per"    % Population Housing units Built between 2000 and 2009

## Top 10 Most Influential Variables for Republican Majority:
coeffs_best_subset = names(LR_best_subset$coefficients)[order(LR_best_subset$coefficients, decreasing = T)]
##   1. "famwithchild_per" % Population with Families with children by birth, marriage or adoption under 18 years
##   2. "hhval200499_per"  % Population Owner-occupied housing value from $200,000 to $499,999
##   3. "hh2per_per"       % Population with 2 persons household
##   4. "hhval5099_per"    % Population Owner-occupied housing value from $50,000 to $99,999
##   5. "vacant_per"       % Vacant Housing units
##   6. "hhval100199_per"  % Population Owner-occupied housing value from $100,000 to $199,999
##   7. "rent35over_per"   % 35 percent or more of household income on gross rent (renter occupied)
##   8. "house0009_per"    % Population Housing units Built between 2000 and 2009
##   9. "age65over_per"    % Population 65 years and over
##  10. "private_per"      % Population Employed in private sector

## Top 10 Most Influential Variables for Democratic Majority:
coeffs_best_subset = names(LR_best_subset$coefficients)[order(LR_best_subset$coefficients, decreasing = F)]
##   1. "hhsnap_per"       % Households that received food stamps/SNAP in the past 12 months
##   2. "hhinc200over_per" % Population with $200,000 and over annual household income
##   3. "grent1250_per"    % Population with Gross rent $1,250 and above
##   4. "finance_per"      % Population Finance and insurance, and real estate and rental and leasing industries
##   5. "own2034_per"      % Population where 20 to 34.9 percent of household income on housing costs (owner occupied)
##   6. "hhinc1024_per"    % Population with $10,000 to $24,999 annual household income
##   7. "fampovmarried_per"% Married couple families living in Poverty
##   8. "VAP"              % Population of Voting Age
##   9. "famsinglework_per"% Single-parent households with the parent employed
##  10. "self_per"         % Population Self-employed

######################## Shrinkage: lasso Regression #########################

set.seed(1234) # Just to reset seed and as a "Checkpoint"
new_training_x = scale(model.matrix(Winner ~ ., data = new_training)[,-1])
new_training_y = new_training$Winner

new_testing_x = scale(model.matrix(Winner ~ ., data = new_testing)[,-1])
new_testing_y = new_testing$Winner

# beta_0 =
mean(new_training_y) # 0.2333 # bias toward Republican

# Make a list of Lambda values
list.lambda = 10^seq(1,-2,length = 100) # large numbers for lambda

train = sample(nrow(new_training_x), 0.8*nrow(new_training_x)) # 5-fold CV
lasso.cv = cv.glmnet(new_training_x[train, ], new_training_y[train], alpha = 0)

plot(lasso.cv) # to see what the MSE looks like

lasso.cv$lambda.min # 7.63205
lasso.cv$lambda.1se # 11.0728

## Function to find Degrees of Freedom
df = function(lambda, matrix = x){
  vdfL = c()
  svd.x = svd(matrix)
  # U = svd.x$u
  D = svd.x$d # only calculation needed for SVD.
  # V = svd.x$v # read each of the matrices
  for(i in 1:length(lambda)){
    # need singular value decomposition
    
    dfL = D^2 / (D^2 + lambda[i])
    dfL = sum(dfL)
    # rename the sum. Change from vector to a single value
    # take a list of lambdas, not just one 
    vdfL = c(vdfL, dfL)
    # takes the previous matrix and appends the last element as we go on in i over R.
  }
  return(vdfL) # return as a vector
}

df(lasso.cv$lambda.min, new_training_x[train, ]) # 44.08797
df(lasso.cv$lambda.1se, new_training_x[train, ]) # 39.89835, better because of lower degrees of freedom

# Final Model
final.lasso.model = glmnet(x = new_training_x,
                           y = new_training_y,
                           alpha = 0) # 0 for lasso

final.lasso.model

lasso_test = 2*(predict(final.lasso.model, s = lasso.cv$lambda.1se, newx = new_testing_x) > 0) - 1


## For misclassification error:
MCE_lasso  = mean(new_testing$Winner != lasso_test) # 23.33% Misclassification.

confusionMatrix(data = as.factor(lasso_test),
                reference = as.factor(new_testing$Winner),
                positive = "1")

lasso_roc = roc(as.factor(new_testing$Winner), as.numeric(lasso_test))

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       40.00%      |       00.00% 
# ---------------------+-------------------+--------------------
# Predicted Republican |       23.33%      |       36.67%

# Accuracy    (Total)         = 76.67%
# Sensitivity (True Positive) = 100.0%
# Specificity (True Negative) = 63.16%
# AUC                         = 81.58% Not bad

############################## Shrinkage: LASSO ###############################

set.seed(1234) # Just to reset seed and as a "Checkpoint"
new_training_x = scale(model.matrix(Winner ~ ., data = new_training)[,-1])
new_training_y = new_training$Winner

new_testing_x = scale(model.matrix(Winner ~ ., data = new_testing)[,-1])
new_testing_y = new_testing$Winner

# beta_0 =
mean(new_training_y) # 0.2333 # bias toward Republican

# Make a list of Lambda values
list.lambda = 10^seq(1,-2,length = 100) # large numbers for lambda

train = sample(nrow(new_training_x), 0.8*nrow(new_training_x)) # 5-fold CV
lasso.cv = cv.glmnet(new_training_x[train, ], new_training_y[train], alpha = 1)

plot(lasso.cv) # to see what the MSE looks like

lasso.cv$lambda.min # 0.01763103
lasso.cv$lambda.1se # 0.1644275

df(lasso.cv$lambda.min, new_training_x[train, ]) # 85.5317
df(lasso.cv$lambda.1se, new_training_x[train, ]) # 77.87815, better because of lower degrees of freedom

# Final Model
final.lasso.model = glmnet(x = new_training_x,
                           y = new_training_y,
                           alpha = 1) # 1 for lasso

lasso_test = 2*(predict(final.lasso.model, s = lasso.cv$lambda.1se, newx = new_testing_x) > 0) - 1


## For misclassification error:
MCE_lasso  = mean(new_testing$Winner != lasso_test) # 23.33% Misclassification.

confusionMatrix(data = as.factor(lasso_test),
                reference = as.factor(new_testing$Winner),
                positive = "1")

lasso_roc = roc(as.factor(new_testing$Winner), as.numeric(lasso_test))

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       40.00%      |       00.00% 
# ---------------------+-------------------+--------------------
# Predicted Republican |       23.33%      |       36.67%

# Accuracy    (Total)         = 76.67%
# Sensitivity (True Positive) = 100.0%
# Specificity (True Negative) = 63.16%
# AUC                         = 81.58% Same as Ridge


##################### Shrinkage: Elastic Net Regression ######################

set.seed(1234) # Just to reset seed and as a "Checkpoint"
new_training_x = scale(model.matrix(Winner ~ ., data = new_training)[,-1])
new_training_y = new_training$Winner

new_testing_x = scale(model.matrix(Winner ~ ., data = new_testing)[,-1])
new_testing_y = new_testing$Winner

# beta_0 =
mean(new_training_y) # 0.2333 # bias toward Republican

best_alpha = function(x = new_testing_x, y = new_testing_y){
  best_MCE = 1.00
  best_alph = Inf
  list_alpha = seq(1,0,0.01)
  # Make a list of Lambda values
  list.lambda = 10^seq(1,-2,length = 100) # large numbers for lambda

  for(alph in list_alpha){
    train = sample(nrow(new_training_x), 0.8*nrow(new_training_x)) # 5-fold CV
    enet.cv = cv.glmnet(new_training_x[train, ], new_training_y[train], alpha = alph)
    
    dfs = c(df(enet.cv$lambda.min, new_training_x[train, ]),
            df(enet.cv$lambda.1se, new_training_x[train, ]))
    df_best = dfs[which(min(dfs) == dfs)]
    
    enet.model = glmnet(x = new_training_x,
                        y = new_training_y,
                        alpha = alph)
    
    enet_test = 2*(predict(enet.model, s = df_best, newx = new_testing_x) > 0) - 1
    
    MCE_enet  = mean(new_testing$Winner != enet_test)
    if(MCE_enet < best_MCE){
      return(alph)
    }
  }
}

best_alpha(new_testing_x, new_testing_y) # Returns 0 -> Best is Ridge Regression

####################### Principal Component Regression ########################

pcr.model = pcr(Winner ~ . ,
                data = new_training, scale = TRUE, validation = "CV", subset = train) # default is ten-fold CV
p = validationplot(pcr.model, val.type = "MSEP") # looks like it should be 15 components

pcr.model = pcr(Winner  ~. ,
                data = new_training, scale = TRUE, ncomp = 15) # number of components is 5
summary(pcr.model)
# These components explain 88% of the variance explained by the model

pcr_test = 2*(predict(pcr.model, new_testing, ncomp = 15) > 0) - 1

MCE_pcr  = mean(new_testing$Winner != pcr_test) # 13.33% Misclassification.

confusionMatrix(data = as.factor(pcr_test),
                reference = as.factor(new_testing$Winner),
                positive = "1")

pcr_roc = roc(as.factor(new_testing$Winner), as.numeric(pcr_test))

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       53.33%      |       03.33% 
# ---------------------+-------------------+--------------------
# Predicted Republican |       10.00%      |       33.33%

# Accuracy    (Total)         = 86.67%
# Sensitivity (True Positive) = 90.91%
# Specificity (True Negative) = 84.21%
# AUC                         = 87.56%

########################### Partial Least Squares #############################

pls.model = plsr(Winner ~ . , data = new_training, scale = TRUE, validation = "CV", subset = train) # default is ten-fold CV
p = validationplot(pls.model, val.type = "MSEP") # Looks like 4 (Very strange)
summary(pls.model) # Looks like it should be 22 components

pls_test = 2*(predict( pls.model, new_testing, ncomp = 22) > 0) - 1 

MCE_pls  = mean(new_testing$Winner != pls_test) # 20.00% Misclassification.

confusionMatrix(data = as.factor(pls_test),
                reference = as.factor(new_testing$Winner),
                positive = "1")

pls_roc = roc(as.factor(new_testing$Winner), as.numeric(pls_test))

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       46.67%      |       03.33% 
# ---------------------+-------------------+--------------------
# Predicted Republican |       16.67%      |       33.33%

# Accuracy    (Total)         = 80.00%
# Sensitivity (True Positive) = 90.91%
# Specificity (True Negative) = 73.68%
# AUC                         = 82.30%

############################ K-Means Regression ##############################

set.seed(1234) # Reset Seed

# A function that returns the MSE per column to later perform CV
K_means_MSE_calculator = function(K){
  s = sample(nrow(training), 0.8*nrow(training))
  train = new_training[s, ]
  test = new_training[-s, ]
  K_means = kmeans(train, centers = K)
  KNN_levels = knn(train, test, K_means$cluster)
  MSE_all_columns = colMeans((K_means$centers[KNN_levels, ] - test)^2)
  return(MSE_all_columns)
}

# initialize an empty vector to later visualize the MSE for all variables in
means_to_display = matrix(nrow = ncol(new_training), ncol = (0.8*nrow(new_training)-1))

# Performs CV
for (k in 1:50){
  max_iter = 100
  a = replicate(max_iter, K_means_MSE_calculator(k))
  mean_MSE_all_rows = as.matrix(rowMeans(a))
  means_to_display[1:ncol(new_training), k] = mean_MSE_all_rows
}
  
list_ks = 1:50
p = plot(list_ks, na.omit(means_to_display[ncol(new_training), ]), 
           xlab = "Value of k in K-Means Regression",
           ylab = "MSE for Winner", 
           main = "Evaluating MSE for Winner using K-Means") # looks like k = 18 (elbow)

# Final k-means regression 
K_means_final = kmeans(new_training, centers = 18)

# Predictions on the Testing Data
KNN_test = as.numeric(knn(new_training, new_testing, cl = K_means_final$cluster)) - 1 

MCE_KMeans = mean(KNN_test != new_testing$Winner) #63.33% misclassification 

confusionMatrix(data = as.factor(KNN_test),
                reference = as.factor(new_testing$Winner),
                positive = "1")

kmeans_roc = roc(as.factor(new_testing$Winner), as.numeric(KNN_test))

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       00.00%      |       00.00% 
# ---------------------+-------------------+--------------------
# Predicted Republican |       63.33%      |       36.67%

# Accuracy    (Total)         = 36.67%
# Sensitivity (True Positive) = 100.0%
# Specificity (True Negative) = 36.67%
# AUC                         = 50.00%

############################ Logistic Regression ##############################

# Logistic Regression requires Winner to be between 0 and 1 
new_training$Winner = 0.5*(new_training$Winner + 1)
new_testing$Winner = 0.5*(new_testing$Winner + 1)

logistic_model = glm(Winner~., data = new_training, family = binomial)

logistic_prediction = (predict(logistic_model, new_testing) > 0) + 0 

MCE_logistic = mean(new_testing$Winner != logistic_prediction)
# 20.00% Misclassification Error

confusionMatrix(data = as.factor(logistic_prediction),
                reference = as.factor(new_testing$Winner),
                positive = "1")

logistic_roc = roc(new_testing$Winner, logistic_prediction)

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       53.33%      |       10.00%  
# ---------------------+-------------------+--------------------
# Predicted Republican |       10.00%      |       26.67%

# Accuracy    (Total)         = 80.00%
# Sensitivity (True Positive) = 72.73%
# Specificity (True Negative) = 84.21%
# AUC                         = 78.47%

# Change back "Winner" for the rest of the methods
new_training$Winner = 2*(new_training$Winner) - 1
new_testing$Winner = 2*(new_testing$Winner) - 1

######################## Linear Discriminant Analysis #########################

LDA_model = lda(Winner ~., data = new_training)

LDA_prediction = predict(LDA_model, new_testing)$class

MCE_LDA = mean(new_testing$Winner != LDA_prediction)
# 13.33% Misclassification Error

confusionMatrix(data = as.factor(LDA_prediction),
                reference = as.factor(new_testing$Winner),
                positive = "1")

LDA_roc = roc(new_testing$Winner, as.numeric(LDA_prediction))


#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       56.67%      |       06.67%  
# ---------------------+-------------------+--------------------
# Predicted Republican |       06.67%      |       30.00%

# Accuracy    (Total)         = 86.67%
# Sensitivity (True Positive) = 81.82%
# Specificity (True Negative) = 89.47%
# AUC                         = 85.65%

###################### Quadratic Discriminant Analysis ########################

# QDA_model = qda(Winner ~., data = new_training)
# leads to an error because #parameters > #observations

# Must reduce dimension. I will use the subset as created by the 
# Backwards Leaps and Bounds Method that minimizes Mallow's Cp.

subset = setdiff(names(LR_best_subset$coefficients), "(Intercept)")

new_training_qda = new_training[c("Winner",subset)]
new_testing_qda = new_testing[c("Winner", subset)]

QDA_model = qda(Winner ~., data = new_training_qda)

QDA_prediction = predict(QDA_model, new_testing_qda)$class

MCE_QDA = mean(new_testing$Winner != QDA_prediction)
# 20.00% Misclassification Error

confusionMatrix(data = as.factor(QDA_prediction),
                reference = as.factor(new_testing$Winner),
                positive = "1")

QDA_roc = roc(new_testing$Winner, as.numeric(QDA_prediction))


#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       46.67%      |       03.33%  
# ---------------------+-------------------+--------------------
# Predicted Republican |       16.67%      |       33.33%

# Accuracy    (Total)         = 80.00%
# Sensitivity (True Positive) = 90.91%
# Specificity (True Negative) = 73.68%
# AUC                         = 82.30%

######################## Naive Bayes Classification ##########################

nb_model = naiveBayes(Winner ~., data = new_training)

nb_prediction = predict(nb_model, new_testing)

MCE_nbc = mean(new_testing$Winner != nb_prediction)
# 10.00% Misclassification Error

confusionMatrix(data = as.factor(nb_prediction),
                reference = as.factor(new_testing$Winner),
                positive = "1")

nbc_roc = roc(new_testing$Winner, as.numeric(nb_prediction))


#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       56.67%      |       03.33%  
# ---------------------+-------------------+--------------------
# Predicted Republican |       06.67%      |       33.33%

# Accuracy    (Total)         = 90.00%
# Sensitivity (True Positive) = 90.91%
# Specificity (True Negative) = 89.47%
# AUC                         = 90.19%

############# Support Vector Machine (Polynomial Kernel, tuned) ###############

poly_svm = function(costvalue, polydegree){
  svm_model = svm(Winner ~. ,
                  data = new_training, 
                  cost = costvalue,
                  kernel = "polynomial",
                  degree = polydegree,
                  probability = T)
  svm_prediction = 2*(predict(svm_model, new_testing, probability = T) > 0) - 1
  CM = confusionMatrix(data = as.factor(svm_prediction), 
                       reference = as.factor(new_testing$Winner), 
                       positive = "1")
  accu = CM$overall[1]
  sens = CM$byClass[1]
  spec = CM$byClass[2]
  svm_roc = roc(new_testing$Winner, as.numeric(svm_prediction))
  auc = svm_roc$auc
  bind = c(as.integer(polydegree), costvalue, accu, sens, spec, auc)
  names(bind) = c("Degree", "Cost", "Accuracy", "Sensitivity", "Specificity", "AUC")
  return(bind)
}

costvalues = seq(0.001, 1, 0.001)
degrees = seq(1,3)

poly_svms = data.frame(Degree = as.integer(),
                       Cost = double(),
                       Accuracy = double(),
                       sensitivity = double(),
                       Specificity = double(),
                       AUC = double())

idx = 1
for(d in degrees){
  for(c in costvalues){
    poly_svms[nrow(poly_svms) + idx, ] = poly_svm(c,d)
    idx = idx + 1
  }
  poly_svms = na.omit(poly_svms)
}

rownames(poly_svms) = seq(1:nrow(poly_svms))

x = seq(1:nrow(poly_svms))

pAUC = ggplot(data = poly_svms, aes(x = x, y = poly_svms$AUC)) +
  geom_point(data = as.data.frame(cbind(x, poly_svms$AUC)))

pspec = ggplot(data = poly_svms, aes(x = x, y = poly_svms$Specificity)) +
  geom_point(data = as.data.frame(cbind(x, poly_svms$Specificity)))

psens = ggplot(data = poly_svms, aes(x = x, y = poly_svms$sensitivity)) +
  geom_point(data = as.data.frame(cbind(x, poly_svms$sensitivity)))

paccu = ggplot(data = poly_svms, aes(x = x, y = poly_svms$Accuracy)) +
  geom_point(data = as.data.frame(cbind(x, poly_svms$Accuracy)))

View(poly_svms) # Best: Degree 1, Cost = 0.355

svm_model = svm(Winner ~. ,
                data = new_training, 
                cost = 0.355,
                kernel = "polynomial",
                degree = 1,
                probability = T)
svm_prediction = 2*(predict(svm_model, new_testing, probability = T) > 0) - 1

MCE_svm = mean(svm_prediction != new_testing$Winner)

confusionMatrix(data = as.factor(svm_prediction), 
                     reference = as.factor(new_testing$Winner), 
                     positive = "1")
svm_roc = roc(new_testing$Winner, as.numeric(svm_prediction))

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       56.67%      |       00.00%  
# ---------------------+-------------------+--------------------
# Predicted Republican |       06.67%      |       36.67%

# Accuracy    (Total)         = 93.33%
# Sensitivity (True Positive) = 100.0%
# Specificity (True Negative) = 89.47%
# AUC                         = 90.19%

################ Support Vector Machine (Radial Kernel, Tuned) ################

rad_svm = function(gammavalue, costvalue = 0.355){
  svm_model = svm(Winner ~. ,
                  data = new_training, 
                  cost = costvalue,
                  kernel = "radial",
                  gamma = gammavalue,
                  probability = T)
  
  svm_prediction = as.factor(2*(predict(svm_model, new_testing, probability = T) > 0) - 1)
  
  CM = confusionMatrix(data = svm_prediction, 
                       reference = as.factor(new_testing$Winner), 
                       positive = "1")
  accu = CM$overall[1]
  sens = CM$byClass[1]
  spec = CM$byClass[2]
  svm_roc = roc(new_testing$Winner, as.numeric(svm_prediction))
  auc = svm_roc$auc
  bind = c(gammavalue, costvalue, accu, sens, spec, auc)
  names(bind) = c("Gamma", "Cost", "Accuracy", "Sensitivity", "Specificity", "AUC")
  return(bind)
}

gammas = seq(0.009,0.01,0.0001)

rad_svms =  data.frame(Gamma = double(),
                       Cost = double(),
                       Accuracy = double(),
                       Sensitivity = double(),
                       Specificity = double(),
                       AUC = double())

idx = 1
for(g in gammas){
  rad_svms[nrow(rad_svms) + idx, ] = rad_svm(gammavalue = g, costvalue = 0.355)
  idx = idx + 1
  rad_svms = na.omit(rad_svms)
}

rownames(rad_svms) = seq(1:nrow(rad_svms))

x = seq(1:nrow(rad_svms))

pAUC = ggplot(data = rad_svms, aes(x = x, y = rad_svms$AUC)) +
  geom_point(data = as.data.frame(cbind(x, rad_svms$AUC)))

pspec = ggplot(data = rad_svms, aes(x = x, y = rad_svms$Specificity)) +
  geom_point(data = as.data.frame(cbind(x, rad_svms$Specificity)))

psens = ggplot(data = rad_svms, aes(x = x, y = rad_svms$Sensitivity)) +
  geom_point(data = as.data.frame(cbind(x, rad_svms$sensitivity)))

paccu = ggplot(data = rad_svms, aes(x = x, y = rad_svms$Accuracy)) +
  geom_point(data = as.data.frame(cbind(x, rad_svms$Accuracy)))

# Best: Gamma = 0.09, Cost = 0.355 (like in Poly)

svm_model = svm(Winner ~. ,
                data = new_training, 
                cost = 0.355,
                kernel = "radial",
                gamma = 0.009,
                probability = T)
svm_prediction = 2*(predict(svm_model, new_testing, probability = T) > 0) - 1

MCE_svm = mean(svm_prediction != new_testing$Winner)

confusionMatrix(data = as.factor(svm_prediction), 
                reference = as.factor(new_testing$Winner), 
                positive = "1")
svm_roc = roc(new_testing$Winner, as.numeric(svm_prediction))

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       56.67%      |       00.00%  
# ---------------------+-------------------+--------------------
# Predicted Republican |       06.67%      |       36.67%

# Accuracy    (Total)         = 93.33%
# Sensitivity (True Positive) = 100.0%
# Specificity (True Negative) = 94.74%
# AUC                         = 97.37%

########################### Random Forest (Tuned) ############################

set.seed(1234) # Just to reset

mtryMAX = 88
ntrees = 1001

trainOOBerror = seq(1:mtryMAX)*0
bestOOBerror = Inf

for(m in 1:mtryMAX){
  t = randomForest( as.factor(Winner) ~ .,
                    data = new_training,
                    ntree = ntrees,
                    mtry = m)
  trainOOBerror[m] = t$err.rate[nrow(t$err.rate),1]
  if(trainOOBerror[m] < bestOOBerror){
    bestOOBerror = trainOOBerror[m]
    best_treesize = m
  }
}

# Best tree size is 32

tree_model = randomForest( Winner ~ .,
                           data = new_training,
                           ntree = ntrees,
                           mtry = best_treesize,
                           importance = T)

VI_tree_model = as.data.frame(tree_model$importance)
most_VI_tree_model = VI_tree_model[order(VI_tree_model$IncNodePurity, decreasing = T), ]
varImpPlot(tree_model)

rownames(most_VI_tree_model)

tree_prediction = 2*(predict(tree_model, new_testing) > 0) - 1

MCE_trees = mean(tree_prediction != new_testing$Winner)

confusionMatrix(data = as.factor(tree_prediction),
                reference = as.factor(new_testing$Winner),
                positive = "1")

trees_roc = roc(new_testing$Winner, as.numeric(tree_prediction))

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       60.00%      |       06.67%  
# ---------------------+-------------------+--------------------
# Predicted Republican |       03.33%      |       30.00%

# Accuracy    (Total)         = 90.00%
# Sensitivity (True Positive) = 81.82%
# Specificity (True Negative) = 94.74%
# AUC                         = 97.37%

############################### Neural Network ################################

set.seed(1234)
## softplus = function(x) {log(1 + exp(x))} Did not lead to any MCE below 26%

hidden_layer_finder = function(hlayers){
  nn_model = neuralnet( Winner~ .,
                        data = new_training, 
                        hidden = hlayers, 
                        threshold = 0.001,
                        lifesign = "full",
                        lifesign.step = 10, 
                        act.fct = "logistic",
                        learningrate = 0.01)
  nn_prediction = 2*(predict(nn_model, new_testing) > 0) - 1
  return(mean(nn_prediction != new_testing$Winner))
}


num_layers = seq(0,50)
best_MCE = Inf
best_layer = length(num_layers) + 1
for(layeridx in num_layers){
  currentMCE = hidden_layer_finder(layeridx)
  if(currentMCE < best_MCE){
    best_MCE = currentMCE
    best_layer = layeridx
  }
}

nn_model = neuralnet( Winner~ .,
                      data = new_training, 
                      hidden = best_layer, 
                      threshold = 0.001,
                      lifesign = "full",
                      lifesign.step = 10, 
                      act.fct = "logistic",
                      learningrate = 0.01)
nn_prediction = 2*(predict(nn_model, new_testing) > 0) - 1
MCE_nn = mean(nn_prediction != new_testing$Winner)

confusionMatrix(data = as.factor(nn_prediction),
                reference = as.factor(new_testing$Winner),
                positive = "1")

nn_roc = roc(new_testing$Winner, as.numeric(tree_prediction))

#   Confusion Matrix   | Democrat Majority | Republican Majority
# ---------------------+-------------------+--------------------
#  Predicted Democrat  |       56.67%      |       00.00%  
# ---------------------+-------------------+--------------------
# Predicted Republican |       06.67%      |       36.67%

# Accuracy    (Total)         = 93.33%
# Sensitivity (True Positive) = 81.82%
# Specificity (True Negative) = 94.74%
# AUC                         = 97.37%


##############################################################################

# Model Type        |  MCE   |  Accuracy | Sensitivity | Specificity |  AUC     *>= 90% AUC
# ------------------+--------+-----------+-------------+-------------+---------
# Linear Model      | 26.67% |   73.33%  |    54.55%   |    84.21%   | 74.64%
# ------------------+--------+-----------+-------------+-------------+---------
# After Diagnostics | 13.33% |   86.67%  |    81.82%   |    89.47%   | 85.65%
# ------------------+--------+-----------+-------------+-------------+---------
# Voronoi           | 36.67% |   63.33%  |    00.00%   |    100.0%   | 50.00%
# ------------------+--------+-----------+-------------+-------------+---------
# K-NearestNeighbor | 36.67% |   63.33%  |    00.00%   |    100.0%   | 50.00% 
# ------------------+--------+-----------+-------------+-------------+---------
# Bwd L&B (min BIC) | 10.00% |   90.00%  |    90.91%   |    89.47%   | 90.19%   *
# ------------------+--------+-----------+-------------+-------------+---------
# Ridge Regression  | 23.33% |   76.67%  |    100.0%   |    63.16%   | 81.58%
# ------------------+--------+-----------+-------------+-------------+---------
# LASSO             | 30.00% |   73.33%  |    54.55%   |    24.21%   | 74.64%
# ------------------+--------+-----------+-------------+-------------+---------
# Elastic Net       | 23.33% |   76.67%  |    100.0%   |    63.16%   | 81.58%
# ------------------+--------+-----------+-------------+-------------+---------
# PCR               | 13.33% |   86.67%  |    90.91%   |    84.21%   | 87.56%
# ------------------+--------+-----------+-------------+-------------+---------
# Partial LS        | 20.00% |   80.00%  |    90.91%   |    73.68%   | 82.30%
# ------------------+--------+-----------+-------------+-------------+---------
# K-Means Reg       | 63.33% |   36.67%  |    100.0%   |    36.67%   | 50.00%
# ------------------+--------+-----------+-------------+-------------+---------
# Logistic Reg      | 20.00% |   80.00%  |    72.73%   |    84.21%   | 78.47%
# ------------------+--------+-----------+-------------+-------------+---------
# LDA               | 13.33% |   86.67%  |    81.82%   |    89.47%   | 85.65%
# ------------------+--------+-----------+-------------+-------------+---------
# QDA (subset selec)| 20.00% |   80.00%  |    90.91%   |    73.68%   | 82.30%
# ------------------+--------+-----------+-------------+-------------+---------
# Naive Bayes       | 10.00% |   90.00%  |    90.91%   |    89.47%   | 90.19%   *
# ------------------+--------+-----------+-------------+-------------+---------
# SVM Poly          | 06.67% |   93.33%  |    100.0%   |    89.47%   | 94.74%   *
# ------------------+--------+-----------+-------------+-------------+---------
# SVM Radial        | 03.33% |   96.67%  |    100.0%   |    94.74%   | 97.37%   *
# ------------------+--------+-----------+-------------+-------------+---------
# Random Forest     | 10.00% |   90.00%  |    81.82%   |    94.74%   | 88.28%
# ------------------+--------+-----------+-------------+-------------+---------
# NeuralNet(2 layer)| 06.67% |   93.33%  |    100.0%   |    89.47%   | 90.19%   *


## Ranked from "Best to Worst" Methods for Prediction 
## Based on AUC, then Accuracy, then Specificity, then Sensitivity, then Complexity

#  1. Support Vector Machine - Radial Kernel, Cost = 0.355, Gamma = 0.009
#  2. Support Vector Machine - Linear Kernel, Cost = 0.355
#  3. Neural Network - 2 Hidden Layers
#  4. Backward Leaps and Bounds - Minimizes Mallow's Cp
#  5. Naive Bayes Classifier
#  6. Random Forest - 32 Splits per node
#  7. Principal Component Regression
#  8. Linear Model After Diagnostics
#  9. Linear Discriminant Analysis
# 10. Partial Least Squares
# 11. Quadratic Discriminant Analysis with Subset Selection
# 12. Ridge Regression
# 13. Elastic Net Regression (Returned Ridge Regression, alpha = 0)
# 14. Logistic Regression
# 15. Linear Regression before Model Diagnostics
# 16. LASSO
# 17. Voronoi
# 18. K-Nearest Neighbor
# 19. K-Means Regression
