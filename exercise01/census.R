@@ -0,0 +1,233 @@
#to do: test for interactions
#split data into training, validation, test


#Created View in SQLLite to flatten the dataset:  
#CREATE VIEW "flattened" AS  SELECT r.id, age, w.name as workclass, e.name as education_level, education_num, m.name as marital_status,
#o.name as occupation, relationships.name as relationship, race.name as race,
#s.name as gender, capital_gain, capital_loss, hours_week, c.name as country, over_50k
#FROM records r left join workclasses w on r.workclass_id = w.id 
#left join education_levels e on r.education_level_id = e.id
#left join marital_statuses m on r.marital_status_id = m.id  
#left join occupations o on r.occupation_id= o.id 
#left join races race on r.race_id = race.id
#left join sexes s on r.sex_id= s.id left join countries c on r.country_id = c.id
#left join relationships on r.relationship_id =relationships.id

require(vcd)#for association tests
require(rpart) #for decision tree
require(rpart.plot) #to plot decision tree
require(ROCR) #for predication and performance

###############################################################################
#Import, explore and shape data
###############################################################################

#load "flattened" data file
census <- read.csv("flattened.csv") #strings as factors is default

head(census)
str(census)
#48842 observations. 15 variables
#Beware: eduction_level and education_num convey the same information

#Need to treat education_num as ordinal
census$education_num <- ordered(census$education_num, levels=c(1:16))
class(census$education_num)# confirm: ordered factor

#Question mark is used as placeholder for missing values.  Replace with NA
census[census=="?"] <- NA

#initial exploration of variables
#get list of numeric variables
names(census[sapply(census,is.numeric)])
# "age" "capital_gain" "capital_loss" "hours_week"   "over_50k" 

hist(census$age)
hist(census$capital_gain)# almost all 0
hist(census$capital_loss)# almost all 0
hist(census$hours_week)
hist(census$over_50k)

#look at capital_gain and capital_loss without the 0s
hist(census$capital_gain[census$capital_gain != 0]/100) #scale is /100
hist(census$capital_loss[census$capital_loss != 0]) 

#Examine distribution of target variable
table(census$over_50k) #proportions are ~ 0.76 0, 0.24 1.

#age and hours_week are continuous
mean(census$age) #38.64
median(census$age)#37
sd(census$age)#13.71

mean(census$hours_week)#40.42238
median(census$hours_week)#40
sd(census$hours_week)#12.39

#look for correlations among the numeric variables
cor(census[,c("age","capital_gain", "capital_loss","hours_week","over_50k")])
#No strong correlations to note.  
###Not sure if this is correct approach when comparing continuous to binary.


#Tests of association
#look for relationships between the categorical variables
#get list of categorical variables
names(census[sapply(census,is.factor)])

#workclass
table(census$workclass,census$over_50k)
assocstats(table(census$workclass, census$over_50k))
#there is an association between workclass and income
#There are no values of over_50k and never worked.

#collapse levels - change all instances of Never-worked to Without-pay
census$workclass[census$workclass == "Never-worked"] <- "Without-pay"


#education_level
table(census$education_num, census$over_50k)
assocstats(table(census$education_num, census$over_50k))
#there is an association between education and income

#marital status
table(census$marital_status, census$over_50k)
assocstats(table(census$marital_status, census$over_50k))
#there is an association between marital status and income

#occupation
table(census$occupation, census$over_50k) #Note: low numbers for armed_forces/over50k and priv-house-serv/over50k
#need to do fisher's exact test??
assocstats(table(census$occupation, census$over_50k))

#relationship
table(census$relationship, census$over_50k)
assocstats(table(census$relationship, census$over_50k))

#race
table(census$race, census$over_50k)
assocstats(table(census$race, census$over_50k))

#gender
table(census$gender, census$over_50k)
assocstats(table(census$gender, census$over_50k))

#county
table(census$country, census$over_50k) 
#Note: low numbers for several countries. 0 50k for Holland.
#beware of quasi-complete separation
assocstats(table(census$country, census$over_50k) )

#all categorical variables seem to have an association with over_50k

#split data into training (70%), validation (30%) 
#not sure how to make use of a test data set, so using only training
#and validation for this pass
set.seed(500)
alpha <- 0.7 #training percentage
inTrain <- sample(1:nrow(census), alpha*nrow(census))
censusTrain <- census[inTrain,] #n= 34189 
censusValidation <- census[-inTrain,]

#confirm that proportions are equivalent: 24% event, 76% non-event
table(census$over_50k)
table(censusTrain$over_50k) 
table(censusValidation$over_50k)

#End data exploration and posturing
###############################################################################


###############################################################################
#Analysis: Decision tree
###############################################################################

#Don't have to worry about missing values in decision tree
censusTree <- rpart(over_50k ~ age + workclass + education_num + marital_status + occupation +
                          relationship + race + gender + capital_gain + capital_loss + hours_week + country 
                        ,data=censusTrain, method="class")


rpart.plot(censusTree)
summary(censusTree)
#Variable importance: relationship, marital_status, capital_gain, education_num
plot(censusTree,uniform=TRUE, branch=0.6, margin = 0.05)
text(censusTree, use.n=TRUE, all=TRUE)

printcp(censusTree)
#Variables actually used in tree construction:
#  [1] capital_gain  education_num relationship 
### Why wasn't marital_status used??

#post(censusTree,file="tree.ps")

#Try pruning the tree to minimize complexity parameter.
pfit <- prune(censusTree, cp=
                censusTree$cptable[which.min(censusTree$cptable[,"xerror"]),"CP"])

plot(pfit,uniform=TRUE,
     main="Pruned classification tree")
text(pfit, use.n=TRUE, all=TRUE, cex=0.8)
#result is the same tree

#Model Assessment

tree_pred <- predict(censusTree, newdata=censusValidation, type="class")
summary(tree_pred)

table(censusValidation$over_50k, tree_pred)

tree_pred_Roc  <- predict(censusTree,newdata=censusValidation)
head(tree_pred_Roc)

tree_pred2 <- prediction(tree_pred_Roc[,2], censusValidation$over_50k)
as.numeric(performance(tree_pred2,"auc")@y.values) 
#Area under curve = 0.8439945

plot(performance(tree_pred2,"tpr","fpr"), main = "ROC Curve Tree")


###############################################################################
#Analysis: Logistic regression
###############################################################################

###Remember: In logistic regression, observations with missing values will be removed

lrModel <- glm(over_50k ~ age + workclass + education_num + marital_status + occupation +
                 relationship + race + gender + capital_gain + capital_loss + hours_week + country
               ,data=censusTrain, family=binomial(logit))

#####Warning message:
#####glm.fit: fitted probabilities numerically 0 or 1 occurred 

summary(lrModel)


#Warning message from model with all the parameters.  Remove captial_gain 
#because I'm concerned that it is causing a problem with quasi-complete separation
#Most values are 0.  See histograms above.

#Repeat model with capital gain removed
lrModel2 <- glm(over_50k ~ age + workclass + education_num + marital_status + occupation +
                 relationship + race + gender + capital_loss + hours_week + country 
               ,data=censusTrain, family=binomial(logit))
summary(lrModel2)

### Question to revist:  Do I need to worry about standardizing 
### or transforming any of the numeric variables?

logistic_pred <- predict(lrModel2, newdata=censusValidation, type="response")

###Where's cutoff for calling something 0 or 1?
table(censusValidation$over_50k, logistic_pred >= 0.5)

logistic_pred2 <- prediction(logistic_pred, censusValidation$over_50k)
as.numeric(performance(logistic_pred2,"auc")@y.values)
# 0.8878236


perf_log = performance(logistic_pred2, "tpr", "fpr")
plot(perf_log, main="logistic")

###############################################################################
