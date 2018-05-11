import pandas as pd
import numpy as np
import csv as csv
import matplotlib.pyplot as plt

df1 = pd.read_csv("C://Users//prabhakarona//Documents//Digital General//Learning\Kaggle//Titanic//train.csv")
#print(df1['Survived'].value_counts()) #Frequency table
#print(df1['Sex'].value_counts())
#print(df1['Pclass'].value_counts())
#print(df1['SibSp'].value_counts())

plt.bar(df1['Pclass'], height = df1['Survived'], align='center', alpha=0.5)
plt.xlabel('Pclass')
plt.ylabel('Survived')

plt.hist2d(df1['Pclass'], df1['Survived'], bins = 3)

df1["Pclass"].value_counts().plot(kind = 'bar')

print(df1.columns[df1.isnull().any()]) #Printing column names which have missing values
print(df1.apply(lambda x: sum(x.isnull()),axis=0))# One line code for no of null values in each column

#plt.hist(df1['Age'].dropna(), bins = (0,10,20,30,40,50,60,70,80,90,100)) # to find distribution of age
#plt.title("Age Histogram")
#plt.xlabel("Value")
#plt.ylabel("Frequency")
#plt.show()

from statsmodels.graphics.mosaicplot import mosaic

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

temp3 = pd.crosstab(df1['Pclass'], df1['Survived']) # Crosstab is Contingency table for categorical variabkes relationship on Output
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

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

df1.loc[(df1['New_Salutation'] == 'Master') & (df1['Sex'] == 'male'),'Age'].fillna(temp1.iloc[0,3], inplace = True)
#df1.ix[(df1['New_Salutation'] == 'Miss') & (df1['Sex'] == 'female'),'Age'].fillna(temp1.iloc[1,3], inplace = True)
#df1.ix[(df1['New_Salutation'] == 'Mr') & (df1['Sex'] == 'male'),'Age'].fillna(temp1.iloc[2,3], inplace = True)
#df1.ix[(df1['New_Salutation'] == 'Mrs') & (df1['Sex'] == 'female'),'Age'].fillna(temp1.iloc[3,3], inplace = True)
#df1.ix[(df1['New_Salutation'] == 'Others') & (df1['Sex'] == 'female'),'Age'].fillna(temp1.iloc[4,3], inplace = True)
#df1.ix[(df1['New_Salutation'] == 'Others') & (df1['Sex'] == 'male'),'Age'].fillna(temp1.iloc[5,3], inplace = True)



###########To handle outlier value of 512 in Fare##############
## We will use mean of that class (from Pclass) to fill the value



grp = (df1.groupby(['Pclass']).mean())
#print (grp)
#print(grp.iloc[:,-1])
#np.where
  
df_filt1 = df1.query('Fare>336 and Pclass == 1') # if fare is more than 4 times mean of (Pclass=1) as metric to treat outlier
#Can come with better thresholds (3sigma) basaed on Fare distribution
df_filt2 = df1.query('Fare>336 and Pclass == 2')
df_filt3 = df1.query('Fare>336 and Pclass == 3')

for m in range(0, len(df_filt1.index)): #iterate over length of df_filt1
    df1.iloc[df_filt1.iloc[0,m]-1, 9] = grp.iloc[0,-1] #assign mean of Pclasss '1' to fare by matching PassengerId of df_filt1 with df1
    #print(df1.ix[df_filt1.iloc[0,m]-1, 9])
  
for m in range(0, len(df_filt2.index)):
    df1.iloc[df_filt2.iloc[0,m]-1, 9] = grp.iloc[0,-1]
    #print(df1.ix[df_filt1.iloc[0,m]-1, 9])

for m in range(0, len(df_filt3.index)):
    df1.iloc[df_filt3.iloc[0,m]-1, 9] = grp.iloc[0,-1]
    #print(df1.ix[df_filt1.iloc[0,m]-1, 9])    
    