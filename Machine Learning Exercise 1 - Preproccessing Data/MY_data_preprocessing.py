"""
Created on Wed Sept 9 2020

Creator: William Moss

Purpose: To practice core data preprocessing skills
"""

# importing libraries  
import numpy as np   #Package for scientific calculations
import matplotlib.pyplot as plt  #Package to plot data 
import pandas as pd  #Package to importing and managing datasets
  
#importing datasets  
data_set = pd.read_csv('Data.csv')  
  
#Extracting Independent Variable  
x = data_set.iloc[:, :-1].values  
  
#Extracting Dependent variable  
y = data_set.iloc[:, 3].values  
  
#======Handling missing data(Replacing missing data with the mean value)=======  
"""
If there is any data missing/corrupted (in this case, it's missing), then we must deal with it
We can remove the row with the missing data, but this destroys precious data we need to train our machine
So, we fill the missing slots with the average of the column.
This way, it doesn't affect our calculations but also doesn't crash our machine

NOTE: Sometimes, we must remove the row because taking the average would be impossible.
      For example, with categorical data (I mean, how can you take the average of your Country?!)
"""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')  
  
#Fitting imputer object to the independent varibles x.   
imputer = imputer.fit(x[:, 1:3])  
  
#Replacing missing data with the calculated mean value  
x[:, 1:3] = imputer.transform(x[:, 1:3])  

#======Dealing with Categorial Data=======
"""
Our machine solely uses numbers, so we cannot simply give it categorial data like Country
So, we can either assign each country a number or make a column for each Country, with 1 and 0 telling the machine if it's your country
The first method cannot be used here, since the machine will compare the Country's incorrectly
For example, saying that 3 (For Spain) > 2 (for France). 
So, we go with the second method
"""
#Country Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
label_encoder_x = LabelEncoder()
x[:,0] = label_encoder_x.fit_transform(x[:,0])

#Encoding for dummy variables
onehot_encoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough' ) # Leave the rest of the columns untouched   
x = onehot_encoder.fit_transform(x)

#Purchased Variable
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

#======Splitting the dataset into Training set and Testing set======
"""
We need data to test that our machine is precise and accurate
We could technically use a seperate dataset, but that makes our machine imprecise since it comes from "seperate testing"
So, we seperate the dataset we have into training and testing piles. 
That way, the dataset comes from the "same testing" and corelations are clearer to see
"""
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#======Feature Scaling======
"""
Notice that our 2 "inital" numerical data (Age and Salary) cannot be directly compared to one another. 
You cannot say Age < Salary is bad, for example.
So, to combat this problem, we must perform feature scaling.
This can be done by either Standardization or Normalization
"""
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test) #We only need to transform it because the training data has already been fitted

