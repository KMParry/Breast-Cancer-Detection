import pandas as pd
import io
import requests
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
#import statsmodels.discrete.discrete_model as sm
from sklearn.linear_model import LogisticRegression
import numpy as np
import seaborn as sns

# from data description, these are the column names
names_list = ['id','clump_thickness','uniform_size','uniform_shape','adhesion','epithel_size','bare_nuclei','bland_chromatin','nucleoli','mitoses','Class']

# import the data into pandas dataframe
link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
f = requests.get(link).content
data = pd.read_csv(io.StringIO(f.decode('utf-8')), names = names_list)

# keep only numeric types
data = data.select_dtypes(include=[np.number])

# lets look at the histograms
data.hist()
plt.savefig('histograms.pdf')


# and some scatter plots for these values
plt.figure(1)

# just looking at these, the variables are ALL over the map, it's going to be hard to separate these
sns.pairplot(data,vars = ['clump_thickness','uniform_size','uniform_shape','adhesion','epithel_size'],kind = 'reg')
plt.savefig('correlation_vars.pdf')
plt.close()

# lets first start with a logistic regression to see if we can 
# accurately classify the data into benign or malignant
# first change the benign and malignant scores to 0 and 1, respectively 
data = data.replace(to_replace=[2,4], value = [0,1])

#### need to split up training/testing set 80/20 ########
train = data.sample(frac = 0.8, random_state = 1)
test = data.loc[~data.index.isin(train.index)]

### train a logistic regression on a selection of the columns ###
train_cols = ['clump_thickness', 'uniform_size', 'uniform_shape', 'epithel_size','adhesion','nucleoli']

# subset the set used for the training data and add a constant
logit_train_data = train[train_cols]
logit_train_data = sm.add_constant(logit_train_data)

logit = sm.Logit(train['Class'],logit_train_data)
result = logit.fit()

print result.summary()

logit_test_data = test[train_cols]
logit_test_data = sm.add_constant(logit_test_data)

predict = result.predict(logit_test_data)
#print predict



# I think we'll have to use a nueral net 



    
