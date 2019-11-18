import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv("C://Users//prabhakarona//Documents//Digital General//Learning\Kaggle//Titanic//train.csv")
#print(df1['Survived'].value_counts()) #Frequency table
#print(df1['Sex'].value_counts())
#print(df1['Pclass'].value_counts())

#plt.bar(df1['Pclass'], height = df1['Survived'], align='center', alpha=0.5)
#plt.xlabel('Pclass')
#plt.ylabel('Survived')

#plt.hist2d(df1['Pclass'], df1['Survived'], bins = 3)

#df1["Pclass"].value_counts().plot(kind = 'bar')

#print(df1.columns[df1.isnull().any()]) #Printing column names which have missing values
#print(df1.apply(lambda x: sum(x.isnull()),axis=0))# One line code for no of null values in each column

""" Cabin attribute has 687 missing values oout of 891 (77%). So we remve that column.
Ticket is the ticket number and has no correlation with Survived.  """

del df1['Cabin']
del df1['Ticket']

""" Embarked has only 2 missing values and majority are 'S' (644 out of 891). So we replace those 2 values with 'S' value. """
df1['Embarked'].fillna('S', inplace = True)

#plt.hist(df1['Fare'].dropna()) # to find distribution of age
#plt.title("Age Histogram")
#plt.xlabel("Value")
#plt.ylabel("Frequency")
#plt.show()

#from statsmodels.graphics.mosaicplot import mosaic

#e = df1.groupby(['Pclass', 'Survived']).size() #Grouping to find relation between Pclass and Survived
#print (e)
#mosaic(df1.groupby(['Pclass', 'Survived']).size()) #to visualize the proportion of Survived with dif values of Pclass

#g = df1.groupby(['Sex', 'Survived']).size() #Grouping to find relation between Sex and Survived
#print (g)
#mosaic(df1.groupby(['Sex', 'Survived']).size())

#plt.hist(df1['Fare'].dropna(), bins = (0,50,100,150,200,250,300,350,400,450,500)) # to find distribution of age
#plt.title("Fare Histogram")
#plt.xlabel("Value")
#plt.ylabel("Frequency")
#plt.show()

#df1.boxplot(column='Age', by = 'Survived') # to know age distribution among those who survived and those who didn't
#df1.boxplot(column='Fare', by = 'Survived')

#temp3 = pd.crosstab(df1['Pclass'], df1['Survived']) # Crosstab is Contingency table for categorical variables relationship on Output
#temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

#l = 0
#name = np.zeros(177)
#for k in df1.itertuples(): 
#    if df1.ix[k,'Age'].isnull()==1:
#        name[l] = df1.ix[k,'PassengerId']
#        l=l+1

 
#table = df1.pivot_table(values='Survived', index='Pclass' ,columns='Sex', aggfunc=np.median)
#print (table)

####### Filling in Missing values of Age based on Salutation in name##########

def name_extract(word):
 return word.split(',')[1].split('.')[0].strip()

df2 = pd.DataFrame({'Salutation':df1['Name'].apply(name_extract)}) #creates new data frame for storing salutations

df1 = pd.merge(df1, df2, left_index = True, right_index = True) # merges on index
temp1 = df1.groupby('Salutation').PassengerId.count()
#print (temp1)

def group_salutation(old_salutation):   # to group all other salutations but for Mr, Mrs, Miss and Master into Others
 if old_salutation == 'Mr':
    return('Mr')
 else:
    if old_salutation == 'Mrs':
       return('Mrs')
    else:
       if old_salutation == 'Master':
          return('Master')
       else: 
          if old_salutation == 'Miss':
             return('Miss')
          else:
             return('Others')
             
df3 = pd.DataFrame({'New_Salutation':df1['Salutation'].apply(group_salutation)})

df1 = pd.merge(df1, df3, left_index = True, right_index = True)
temp1 = df1.groupby(['New_Salutation','Sex']).mean() # this is to get the average value grouped based on new salutation and sex
#print (temp1)

"""  
                *******  To replace missing values in Age  *******
"""

"""
df.mask selects those rows based on condition of New Salutation and Sex provided and replaces them with values of mean
for that New Salution and Sex combination in temp variable. However, we need to fill only those Age values which are 
NaN. Hence, we select such NaN Age rows by df1[df1['Age'].isnull()] and applying mask function to these rows. Finally,coming to
LHS of below lines, whatever is returned has to be stored only in the rows whose AGE values is NaN and only the Age column 
values have to be replaced. Hence, in df1.loc[] the row value is df1['Age'].isnull() and column value is 'Age'. 
"""
df1.loc[df1['Age'].isnull(), 'Age']= df1[df1['Age'].isnull()].mask(((df1['New_Salutation'] == 'Master') & (df1['Sex']== 'male')),temp1.iloc[0,3])
df1.loc[df1['Age'].isnull(), 'Age'] = df1[df1['Age'].isnull()].mask(((df1['New_Salutation'] == 'Miss') & (df1['Sex']== 'female')),temp1.iloc[1,3])
df1.loc[df1['Age'].isnull(), 'Age'] = df1[df1['Age'].isnull()].mask(((df1['New_Salutation'] == 'Mr') & (df1['Sex']== 'male')),temp1.iloc[2,3])
df1.loc[df1['Age'].isnull(), 'Age'] = df1[df1['Age'].isnull()].mask(((df1['New_Salutation'] == 'Mrs') & (df1['Sex']== 'female')),temp1.iloc[3,3]) 
df1.loc[df1['Age'].isnull(), 'Age'] = df1[df1['Age'].isnull()].mask(((df1['New_Salutation'] == 'Others') & (df1['Sex']== 'female')),temp1.iloc[4,3])
df1.loc[df1['Age'].isnull(), 'Age'] = df1[df1['Age'].isnull()].mask(((df1['New_Salutation'] == 'Others') & (df1['Sex']== 'male')),temp1.iloc[5,3])
#df1['Age'].isnull().values.any()  #### To check if Age still has any missing values

"""
              *******  To handle outlier value of 512 in Fare  *******    
"""

        ## We will use mean of that class (from Pclass) to fill the value

grp = (df1.groupby(['Pclass']).mean())
   

"""
Find the mean of fare values grouped by Pclass. Then use this value to decide the threshold for outlier.
So, 4 times the mean of Fare values for Plcass = 1 is threshold and all values higher than this will be
replaced with the mean of fare values for THAT Pclass. But mean is not a good way to identify outliers.
So we use MAD measure in next section. 
"""
    
df1['Fare'] = df1['Fare'].mask(((df1['Pclass'] == 1) & (df1['Fare']>= grp.iloc[0,-1]*4)),grp.iloc[0,-1])
df1['Fare'] = df1['Fare'].mask(((df1['Pclass'] == 2) & (df1['Fare']>= grp.iloc[1,-1]*4)),grp.iloc[1,-1])
df1['Fare'] = df1['Fare'].mask(((df1['Pclass'] == 3) & (df1['Fare']>= grp.iloc[2,-1]*4)),grp.iloc[2,-1])

"""
MAD (median absolute deviation) is computed for each combination of Fare and Pclass. Four times this value
is then used as the threshold, i.e. if the value minus the mean is 4 times the MAD, then it is an outlier.  

(4*df1.loc[(df1['Pclass'] == 1), 'Fare'].mad())

To calculate mean, again we use combination of Fare and Pclass. 
 
df1['Pclass'] == 1), 'Fare'].mean()

We then take the absolute value of this difference, then we need to select ONLY the Fare values of such 
rows. 

(df1['Fare']-df1.loc[(df1['Pclass'] == 1), 'Fare'].mean()).abs()

So, these two values are compared to filter only those rows whose Fare value minus mean is more than 4 times
the MAD.

(df1['Fare']-df1.loc[(df1['Pclass'] == 1), 'Fare'].mean()).abs() >= (4*(df1.loc[(df1['Pclass'] == 1), 'Fare'].mad())

For these rows, ONLY the Fare value for THAT PCLASS (hence the AND condition & (df1['Pclass'] == 1) 
(using df1.loc[above line, 'Fare']) is replaced with the respective mean from grp variable. 
All these steps are combined in one single line below. This is repeated for each Pclass (1,2,3).

"""
temp4 = (4*df1.loc[(df1['Pclass'] == 1), 'Fare'].mad())
    
df1.loc[((df1['Fare']-df1.loc[(df1['Pclass'] == 1), 'Fare'].mean()).abs() >= temp4) & (df1['Pclass'] == 1), 'Fare'] = grp.iloc[0,-1]
df1.loc[((df1['Fare']-df1.loc[(df1['Pclass'] == 2), 'Fare'].mean()).abs() >= temp4) & (df1['Pclass'] == 2), 'Fare'] = grp.iloc[1,-1]
df1.loc[((df1['Fare']-df1.loc[(df1['Pclass'] == 3), 'Fare'].mean()).abs() >= temp4) & (df1['Pclass'] == 3), 'Fare'] = grp.iloc[2,-1]

del df1['Salutation']
del df1['New_Salutation']
del df1['PassengerId']
del df1['Name']

""" Even if we retain name and train model on it, test set will have completely new names amd which will have no
correlation. So Name attribute as such has no use. However, New Salutation attribute which is a dervied parameter can
be useful (salutation like Capt may have higher chance of Survival than other salutations. This needs to be explored further).
"""
"""
                            **********   Feature Selection   ********

Need to separate the data into train and validation sets before performing any feature selection methods. 
This is to ensure that the feature selcetion is also not biased and does not see the validation set which
must be seen only during training to improve learning of the model. 

"""

from sklearn.model_selection import train_test_split
df1, df_v = train_test_split(df1, test_size=0.2, random_state = 13)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


"""
You may want to use a label encoder and a one hot encoder to convert string data to numbers.
Also check below link
https://stackoverflow.com/questions/40643831/value-error-could-not-convert-string-to-float-while-using-sklearn-feature-relev 

"""


"""     Pearson's correlation coefficient to generate the coefficient table for all numeric variables      """

df1.drop(['Sex', 'Embarked'], axis=1).corr(method='pearson')
""" To visualize the same correlation matrix, use Seaborn's heatmap"""
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)


"""     ***************     Feature Slection using chi-squared test    ******************     """

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

df1_y = df1['Survived']
df1valid_y = df_v['Survived']

df1_encoded = df1.drop(['Survived'], axis = 1)
df1_encoded['Sex'] = le.fit_transform(df1['Sex'])
df1_encoded['Embarked'] = le.fit_transform(df1['Embarked'])
#df1_encoded['Name'] = le.fit_transform(df1['Name'])

df1valid_x = df_v.drop(['Survived'], axis = 1)
df1valid_x['Sex'] = le.fit_transform(df_v['Sex'])
df1valid_x['Embarked'] = le.fit_transform(df_v['Embarked'])

df1_kbest = SelectKBest(chi2, k=4)
df1_kbest.fit_transform(df1_encoded, df1_y)
""" Note that df1_kbest is an object of class SelectKBest (an instance) and is NOT a dataframe. """
mask = df1_kbest.get_support(indices=True)
""" Above line gives the list of column indexes which are selected"""
df1_kbest_1 = df1_encoded.columns[mask]
""" Above list retrieves the column names from df1_encoded corresponding to the featires that have been selected
by the SelectKBest feature selection method. """
df1_k = df1_encoded.loc[:, df1_encoded.columns.isin(df1_kbest_1)]

""" Should use a better way (create less variables) to do what above chunk of code does"""

"""     Feature Slection using Recursive Feature Elimination (RFE)     """

from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

""" RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier all give consistent results but 
Logistic Regression as model for RFE gives contrasting results, need to understand why. """

clf_rf_3 = RandomForestClassifier()      
rfe = RFE(estimator=clf_rf_3, n_features_to_select=4, step=1)
rfe = rfe.fit(df1_encoded, df1_y)
print('Chosen best 5 feature by rfe:',df1_encoded.columns[rfe.support_])

""" RFECV performs RFE in a cross-validation loop to find the optimal number of features. """

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

# Create the RFE object and compute a cross-validated score.
clf_rf_3 = RandomForestClassifier() 
#svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=clf_rf_3, step=1, cv=6, scoring='accuracy')
rfecv.fit(df1_encoded, df1_y)
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

"""     Feature Importance using   ExtraTreesClassifier   """

#model = ExtraTreesClassifier()
#model.fit(df1_encoded, df1_y)
#print(model.feature_importances_)




"""  *****************     CLASSIFICATION MODELS    ***********************    """

from sklearn import metrics

def train_logistic_regression(train_x, train_y):
    
    logistic_regression_model = LogisticRegression(C = 2)
    logistic_regression_model.fit(train_x, train_y)
    
    return logistic_regression_model

def partial_train_logistic_regression(train_x, train_y):
    
    logistic_regression_model = LogisticRegression(C = 2)
    logistic_regression_model.fit(train_x, train_y)
    
    return logistic_regression_model

def train_linearSVM(train_x, train_y):
    
#     train_x, train_y = make_classification(n_features=7, random_state=0)
     LinearSVM_model = SVC(kernel = "linear", random_state=0)
     LinearSVM_model.fit(train_x, train_y)
     
     return LinearSVM_model

def train_SGDClassifier(train_x, train_y):
    
#     train_x, train_y = make_classification(n_features=7, random_state=0)
     SGD_model = SGDClassifier()
     SGD_model.fit(train_x, train_y)
     
     return SGD_model
 
def train_RandomSearch(train_x, train_y):
    
#     train_x, train_y = make_classification(n_features=7, random_state=0)
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint as sp_randint
    
    RF_model = RandomForestClassifier(random_state=0)
    
    param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 7),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "n_estimators" : sp_randint(50,1000),
              "criterion" : ["entropy", "gini"]
              }

    n_iter_search = 20
    random_search = RandomizedSearchCV(RF_model, param_distributions=param_dist,
                                   n_iter=n_iter_search)


    random_search.fit(train_x, train_y)

    def report(results, n_top=3):
            for i in range(1, n_top + 1):
                candidates = np.flatnonzero(results['rank_test_score'] == i)
                for candidate in candidates:
                    print("Model with rank: {0}".format(i))
                    print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                          results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
                    print("Parameters: {0}".format(results['params'][candidate]))
                    print("")

    report(random_search.cv_results_)
     
    return report(random_search.cv_results_)
 
    
    
def model_accuracy(trained_model, features, targets):
    """
    Get the accuracy score of the model
    :param trained_model:
    :param features:
    :param targets:
    :return:
    """
    accuracy_score = trained_model.score(features, targets)
    
    from sklearn.metrics import precision_score
    y_pred = trained_model.predict(features)
    precision_score = precision_score(targets, y_pred)
    from sklearn.metrics import recall_score
    precision_score = recall_score(targets, y_pred)
    
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(targets, y_pred)
    
    return accuracy_score, precision_score, recall_score, confusion_matrix


#from sklearn import preprocessing
##preprocessing.normalize(df1_encoded['Age'], copy = False)
#""" Normalize is not working, throwing error saying it expects 2D array  """

# Training Logistic regression model
trained_logistic_regression_model = train_logistic_regression(df1_encoded, df1_y)

train_accuracy, precision, recall, confusion_matrix = model_accuracy(trained_logistic_regression_model, df1_encoded, df1_y) 
print ("Train Accuracy for Logistic Regression :: ", train_accuracy)
    
##Testing the logistic regression model
##df1valid_x = df1valid_x.drop(['SibSp', 'Parch', 'Embarked'], axis = 1)
#valid_accuracy, precision, recall,confusion_matrix  = model_accuracy(trained_logistic_regression_model, df1valid_x, df1valid_y) 
#print ("Validation Accuracy for Logistic Regression :: ", valid_accuracy)
#print ("Validation Precision for Logistic Regression :: ", precision)
#print ("Validation Recall for Logistic Regression :: ", recall)
#print ("Confusion Matrix for Logistic Regression :: ", confusion_matrix)
#

trained_logistic_regression_model = partial_train_logistic_regression(df1_encoded, df1_y)

#from sklearn.svm import LinearSVC
## Training Linear SVM model
#LinearSVM_model = train_linearSVM(df1_encoded, df1_y)
#
#train_accuracy, precision, recall, confusion_matrix = model_accuracy(LinearSVM_model, df1_encoded, df1_y) 
#print ("Train Accuracy for SVM :: ", train_accuracy)
#    
## Testing the logistic regression model
##df1valid_x = df1valid_x.drop(['SibSp', 'Parch', 'Embarked'], axis = 1)
#valid_accuracy, precision, recall, confusion_matrix = model_accuracy(LinearSVM_model, df1valid_x, df1valid_y) 
#print ("Validation Accuracy for SVM :: ", valid_accuracy)
#print ("Validation Precision for SVM :: ", precision)
#print ("Validation Recall for SVM :: ", recall)
#print ("Confusion Matrix for SVM :: ", confusion_matrix)   
#    
#from sklearn import linear_model
#SGD_model = train_SGDClassifier(df1_encoded, df1_y)
#train_accuracy, precision, recall, confusion_matrix = model_accuracy(SGD_model, df1_encoded, df1_y) 
#print ("Train Accuracy for SGD :: ", train_accuracy)
#    
## Testing the logistic regression model
##df1valid_x = df1valid_x.drop(['SibSp', 'Parch', 'Embarked'], axis = 1)
#valid_accuracy, precision, recall, confusion_matrix = model_accuracy(SGD_model, df1valid_x, df1valid_y) 
#print ("Validation Accuracy for SGD :: ", valid_accuracy)
#print ("Validation Precision for SGD :: ", precision)
#print ("Validation Recall for SGD :: ", recall)
#print ("Confusion Matrix for SGD :: ", confusion_matrix)
#

"""  ****************** RANDOM FOREST MODEL ************************        """

Random_search = train_RandomSearch(df1_encoded, df1_y)
print ("Best parameters for Random Forest :: ", Random_search)


RF_model = RandomForestClassifier(random_state=0, max_depth = None, min_samples_leaf = 4, n_estimators = 899, max_features = 5, min_samples_split = 4)
RF_model.fit (df1_encoded, df1_y)

train_accuracy, precision, recall, confusion_matrix = model_accuracy(RF_model, df1_encoded, df1_y) 
print ("Train Accuracy for RF :: ", train_accuracy)
    

valid_accuracy, precision, recall, confusion_matrix = model_accuracy(RF_model, df1valid_x, df1valid_y) 
print ("Validation Accuracy for RF :: ", valid_accuracy)
print ("Validation Precision for RF :: ", precision)
print ("Validation Recall for RF :: ", recall)
print ("Confusion Matrix for RF :: ", confusion_matrix)   


"""  ****************** XGBOOST MODEL ************************        """

def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
predictors = [x for x in train.columns if x not in [target, IDcol]]   
import xgboost as xgb

xgb = xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.05,
                max_depth = 4, alpha = 5, n_estimators = 500, n_jobs = -1)

#eval_set = [(df1_encoded, df1_y), (df1valid_x, df1valid_y)]
#eval_metric = ["auc","error"]
#xgb.fit(df1_encoded, df1_y, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

xgb.fit(df1_encoded, df1_y)

train_accuracy, precision, recall, confusion_matrix = model_accuracy(xgb, df1_encoded, df1_y) 
print ("Train Accuracy for XGB :: ", train_accuracy)
    

valid_accuracy, precision, recall, confusion_matrix = model_accuracy(xgb, df1valid_x, df1valid_y) 
print ("Validation Accuracy for XGB :: ", valid_accuracy)
print ("Validation Precision for XGB :: ", precision)
print ("Validation Recall for XGB :: ", recall)
print ("Confusion Matrix for XGB :: ", confusion_matrix) 


#y_pred = xg_reg.predict(df_test_x)
#
#xg_reg.fit(df_train_x.as_matrix(), df_train_y.as_matrix())
#
#train_accuracy = xg_reg.score(df_train_x.as_matrix(), df_train_y.as_matrix())
#print ("Train Accuracy for XGBoost Regressor :: ", train_accuracy)
#
#test_accuracy = xg_reg.score(df_test_x1.as_matrix(), df_test_y.as_matrix())
#print ("Test Accuracy for XGBoost Regressor :: ", test_accuracy)



"""   **************************    MAKING PREDICTIONS ON TEST SET    *********************    """


df_test = pd.read_csv("C://Users//prabhakarona//Documents//Digital General//Learning\Kaggle//Titanic//test.csv")

print(df_test.apply(lambda x: sum(x.isnull()),axis=0))  # No of missing values in each column


""" Replacing missing Age values based on Salutaion in Name"""

df2 = pd.DataFrame({'Salutation':df_test['Name'].apply(name_extract)}) #creates new data frame for storing salutations

df_test = pd.merge(df_test, df2, left_index = True, right_index = True) # merges on index
temp1 = df_test.groupby('Salutation').PassengerId.count()
df3 = pd.DataFrame({'New_Salutation':df_test['Salutation'].apply(group_salutation)})
df_test = pd.merge(df_test, df3, left_index = True, right_index = True)
temp1 = df_test.groupby(['New_Salutation','Sex']).mean()

df_test.loc[df_test['Age'].isnull(), 'Age'] = df_test[df_test['Age'].isnull()].mask(((df_test['New_Salutation'] == 'Master') & (df_test['Sex']== 'male')),temp1.iloc[0,2])
df_test.loc[df_test['Age'].isnull(), 'Age'] = df_test[df_test['Age'].isnull()].mask(((df_test['New_Salutation'] == 'Miss') & (df_test['Sex']== 'female')),temp1.iloc[1,2])
df_test.loc[df_test['Age'].isnull(), 'Age'] = df_test[df_test['Age'].isnull()].mask(((df_test['New_Salutation'] == 'Mr') & (df_test['Sex']== 'male')),temp1.iloc[2,2])
df_test.loc[df_test['Age'].isnull(), 'Age'] = df_test[df_test['Age'].isnull()].mask(((df_test['New_Salutation'] == 'Mrs') & (df_test['Sex']== 'female')),temp1.iloc[3,2]) 
df_test.loc[df_test['Age'].isnull(), 'Age'] = df_test[df_test['Age'].isnull()].mask(((df_test['New_Salutation'] == 'Others') & (df_test['Sex']== 'female')),temp1.iloc[4,2])
df_test.loc[df_test['Age'].isnull(), 'Age'] = df_test[df_test['Age'].isnull()].mask(((df_test['New_Salutation'] == 'Others') & (df_test['Sex']== 'male')),temp1.iloc[5,2])

""" Replacing the missing Fare values with average for that Pclass   """
         
df_test.loc[df_test['Fare'].isnull(), 'Fare'] = df_test[df_test['Fare'].isnull()].mask((df1['Pclass'] == 1), grp.iloc[0,-1])
df_test.loc[df_test['Fare'].isnull(), 'Fare'] = df_test[df_test['Fare'].isnull()].mask((df1['Pclass'] == 2), grp.iloc[1,-1])
df_test.loc[df_test['Fare'].isnull(), 'Fare'] = df_test[df_test['Fare'].isnull()].mask((df1['Pclass'] == 3), grp.iloc[2,-1])


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df_test['Sex'] = le.fit_transform(df_test['Sex'])
df_test['Embarked'] = le.fit_transform(df_test['Embarked'])

df_out = pd.DataFrame(df_test['PassengerId'])
del df_test['PassengerId']
del df_test['Name']
del df_test['Cabin']
del df_test['Ticket']
del df_test['Salutation']
del df_test['New_Salutation']

test_pred = xgb.predict(df_test)

df_out.loc [:, 'Survived'] = test_pred

df_out.to_csv("C://Users//prabhakarona//Documents//Digital General//Learning\Kaggle//Titanic//XGB_1.csv", index = False, header = {'PassengerId' , 'Survived'})

temp = np.arange(25)
import math
temp1 = temp.apply(lambda x: min(x, 24-x))

def f(x):
    # return math.sqrt(x)
    return min(x, 24-x)
vf = np.vectorize(f)
temp1 = vf(temp)

plt.plot(temp1)

plt.plot(np.sin(temp * math.pi/12))
