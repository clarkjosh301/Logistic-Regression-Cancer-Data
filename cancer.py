#dataset from University of Wisconsin www.uwhealth.org, attempting to predict whether breast tumors are cancerous or not 
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
cancer_data = r'C:\Users\CHESTER\Downloads\cancer data.csv'
cancer_df = pd.read_csv(cancer_data)
#must convert malignant and benign data
diagnosis_clean = {'diagnosis': {'M':1, 'B':0}}
cancer_df.replace(diagnosis_clean, inplace=True)
X = cancer_df.drop(['diagnosis'], axis=1)
Y = cancer_df['diagnosis']
#run correlation matrix between independent and dependent variables
corrMatrix = cancer_df[cancer_df.columns].corr()
sns.heatmap(corrMatrix, cmap='YlGnBu', annot = True)
plt.show()
#get rid of indepedent correlation coefficients greater then .9 due to multicollinearity
corr = cancer_df.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
	for j in range(i+1, corr.shape[0]):
		       if corr.iloc[i,j] >= .9:
			       if columns[j]:
				       columns[j] = False
selected_columns = cancer_df.columns[columns]
cancer_df = cancer_df[selected_columns]
#run logistic model result and get rid of P-value greater than .05
logit_model=sm.Logit(Y,X)
result=logit_model.fit()
print(result.summary2())
X = cancer_df.drop(['radius_mean', 'compactness_mean', 'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean'], axis=1)
#finally run machine learning logistic model 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic model on test set:{:.3f}'.format(logreg.score(X_test, Y_test)))
#predicted malignant and benign breat tumors to an accuracy of 100%
