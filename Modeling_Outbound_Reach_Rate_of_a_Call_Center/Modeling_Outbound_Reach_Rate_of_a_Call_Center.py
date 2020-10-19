
# coding: utf-8

# ## 1. Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import BaseEstimator
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics


# ## 2. Data Organization 

# #### 2.1 Reading the Data

# In[ ]:


X = pd.read_csv('training_data.csv') # N = 300000 D = 143
y = pd.read_csv('training_labels.csv',header=None) # N = 300000 D = 1
X_test = pd.read_csv('test_data.csv') # N = 186226 D = 143


# In[ ]:


X.shape


# In[ ]:


y.shape


# #### 2.2 Class Ratio:

# In[ ]:


print('Ratio of y=1: '+str(round(float(y[y==1].count()/y.count()),2))+ ' # of y=1: '+str(int(y[y==1].count())))
print('Ratio of y=0: '+str(round(float(y[y==0].count()/y.count()),2))+ ' # of y=0: '+str(int(y[y==0].count())))


# Class weights are not equal: y=1 only 15% of the time. So I need to give class 1 more weight when building model.

# #### 2.3 Removing low variant binary features

# In[ ]:


#    I will separately consider the binary and continuous features as I will use Bernoulli density variance for binary 
#    features: p*(1-p)
p = 0.01


# In[ ]:


binary_features = [col for col in X if X[col].value_counts().index.isin([0,1]).all()]
cont_features = [col for col in X if X[col].value_counts().index.isin([0,1]).all()==False]
print('There are '+ str(len(binary_features))+' binary features')
print('           '+ str(len(cont_features))+' continuous features')


# In[ ]:


X_bin = X[binary_features].copy()
sel = VarianceThreshold(threshold=(p * (1 - p)))


# In[ ]:


sel.fit_transform(X_bin)


# In[ ]:


binary_features_selected = [pd.Series(list(X_bin.columns)).loc[col] 
                            for col in list(pd.Series(list(X_bin.columns)).index)
                            if col in list(sel.get_support(indices = True))]


# In[ ]:


print('# of binary features selected: '+ str(len(binary_features_selected)))
print('# of binary features dropped : '+ str(len(binary_features)-(len(binary_features_selected))))


# #### 2.4 Removing low variant continuous features

# In[ ]:


X_cont = X[cont_features].copy()


# In[ ]:


X_cont_norm = (X_cont - X_cont.mean()) / (X_cont.max() - X_cont.min())
X_cont_norm.fillna(0,inplace=True)


# In[ ]:


cov_matrix = X_cont_norm.cov()
cov_matrix_diag = pd.DataFrame(np.diag(cov_matrix), index=cov_matrix.index,columns=['covariance'])


# In[ ]:


no_cov_features = list(cov_matrix_diag[cov_matrix_diag['covariance']<=0.01].index)
cont_features_selected = [col for col in list(X_cont.columns) if col not in no_cov_features]

print('# of continuous features selected: '+ str(len(cont_features_selected)))
print('# of continuous features dropped : '+ str(len(cont_features)-(len(cont_features_selected))))
#X.drop(columns=no_cov_features,inplace=True)


# In[ ]:


X = X[binary_features_selected+cont_features_selected]


# In[ ]:


X.shape


# #### 2.4 Selecting Best 30 Features

# In[ ]:


rf_feature_selection = RandomForestClassifier(n_estimators=10,
                                criterion='entropy',
                                max_depth=4,
                                min_samples_split=50,
                                min_samples_leaf=20,
                                max_features='auto',
                                class_weight= {1:1.5,0:0.5})


# In[ ]:


rf_feature_selection.fit(X,np.ravel(y))


# In[ ]:


best_features = pd.DataFrame(rf_feature_selection.feature_importances_,index=X.columns)


# In[ ]:


best_features = best_features.sort_values(by=0,ascending=False)


# In[ ]:


X = X[best_features.index[:30]]


# In[ ]:


X.shape


# ## 3. Separate Test Data

# In[ ]:


X, X_spare, y, y_spare = train_test_split(X,y,test_size=0.20,stratify = y)


# In[ ]:


X.shape


# ## 4. Random Forest Pipeline

# In[ ]:


rf_pipeline = Pipeline([
              ('clf', RandomForestClassifier(n_jobs=-1))
              ])


# In[ ]:


rf_params = {'clf__n_estimators':[1,10,50,200,600,1000],
              'clf__criterion':['gini','entropy'], 
              'clf__max_depth':[4,6,8,12],
              'clf__min_samples_split':[50,200,1000], 
              'clf__min_samples_leaf':[20,80,400],
              'clf__random_state':[421],
              'clf__class_weight':[{1:1,0:1},{1:1.5,0:0.5}]}


# In[ ]:


rfsearch = GridSearchCV(rf_pipeline,rf_params,cv=RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=421),
                      return_train_score=True,scoring='roc_auc')


# In[ ]:


rfsearch.fit(X,y[0].ravel())


# In[ ]:


results = pd.DataFrame.from_dict(data=rfsearch.cv_results_,orient='index')
results


# In[ ]:


rfsearch_y_predicted = rfsearch.predict(X_spare)
rfsearch_y_predicted_train = rfsearch.predict(X)
rfsearch_y_predicted_df = pd.DataFrame(rfsearch_y_predicted)
rfsearch_probs  = rfsearch.predict_proba(X_spare)
rfsearch_probs_train  = rfsearch.predict_proba(X)


# In[ ]:


rfsearch_train_auroc = metrics.roc_auc_score(y, rfsearch_probs_train[:,1])
rfsearch_train_auroc


# In[ ]:


rfsearch_test_auroc = metrics.roc_auc_score(y_spare, rfsearch_probs[:,1])
rfsearch_test_auroc


# In[ ]:


rfsearch_conf_matrix = metrics.confusion_matrix(y_spare, rfsearch_y_predicted)
rfsearch_conf_matrix


# ## 5. Adaptive Boosting Pipeline

# In[ ]:


ada_pipeline = Pipeline([('clf', AdaBoostClassifier())])


# In[ ]:


#RandomForestClassifier(n_estimators = 2, max_depth = 3, random_state=421, criterion='entropy')
ada_params = {'clf__base_estimator': [LogisticRegression(penalty='l2', 
                                                         dual=False, 
                                                         tol=0.01, 
                                                         C=1.0,
                                                         class_weight=None, 
                                                         random_state=421, 
                                                         max_iter=4)],
          'clf__n_estimators':[1000],
          'clf__learning_rate':[0.8],
          'clf__random_state':[421]}


# In[ ]:


adasearch = GridSearchCV(ada_pipeline,ada_params,cv=RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=421),
                      return_train_score=True,scoring='roc_auc')


# In[ ]:


adasearch.fit(X,y[0].ravel())


# In[ ]:


results = pd.DataFrame.from_dict(data=adasearch.cv_results_,orient='index')
results


# In[ ]:


adasearch.best_params_


# In[ ]:


adasearch_y_predicted = adasearch.predict(X_spare)
adasearch_y_predicted_train = adasearch.predict(X)
adasearch_y_predicted_df = pd.DataFrame(adasearch_y_predicted)
adasearch_probs = adasearch.predict_proba(X_spare)
adasearch_probs_train  = adasearch.predict_proba(X)


# In[ ]:


adasearch_train_auroc = metrics.roc_auc_score(y, adasearch_probs_train[:,1])
adasearch_train_auroc


# In[ ]:


adasearch_test_auroc = metrics.roc_auc_score(y_spare, adasearch_probs[:,1])
adasearch_test_auroc


# In[ ]:


adasearch_conf_matrix = metrics.confusion_matrix(y_spare, adasearch_y_predicted)
adasearch_conf_matrix


# ## 6. Gradient Boosting Pipeline

# In[ ]:


gdb_pipeline = Pipeline([('clf', GradientBoostingClassifier())])


# In[ ]:


gdb_params = {'clf__loss':['deviance'], 
              'clf__learning_rate':[0.01], 
              'clf__n_estimators':[400], 
              'clf__criterion':['friedman_mse'], 
              'clf__min_samples_split':[100],
              'clf__min_samples_leaf':[40], 
              'clf__min_weight_fraction_leaf':[0.0], 
              'clf__max_depth':[7], 
              'clf__min_impurity_decrease':[0.0], 
              'clf__min_impurity_split':[None], 
              'clf__init':[None],
              'clf__random_state':[421],
              'clf__max_features':[None]}


# In[ ]:


gdbsearch = GridSearchCV(gdb_pipeline,gdb_params,cv=RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=421),
                      return_train_score=True,scoring='roc_auc')


# In[ ]:


gdbsearch.fit(X.values,y[0].ravel())


# In[ ]:


results = pd.DataFrame.from_dict(data=gdbsearch.cv_results_,orient='index')
results


# In[ ]:


gdbsearch.best_params_


# In[ ]:


gdbsearch_y_predicted = gdbsearch.predict(X_spare)
gdbsearch_y_predicted_train = gdbsearch.predict(X)
gdbsearch_y_predicted_df = pd.DataFrame(gdbsearch_y_predicted)
gdbsearch_probs = gdbsearch.predict_proba(X_spare)
gdbsearch_probs_train  = gdbsearch.predict_proba(X)


# In[ ]:


gdbsearch_train_auroc = metrics.roc_auc_score(y, gdbsearch_probs_train[:,1])
gdbsearch_train_auroc


# In[ ]:


gdbsearch_test_auroc = metrics.roc_auc_score(y_spare, gdbsearch_probs[:,1])
gdbsearch_test_auroc


# In[ ]:


gdbsearch_conf_matrix = metrics.confusion_matrix(y_spare, gdbsearch_y_predicted)
gdbsearch_conf_matrix


# ## 7. Test

# In[ ]:


test_probs = rfsearch.predict_proba(X_test[best_features.index[:30]])
test_probs


# In[ ]:


test_probs_df = pd.DataFrame(test_probs)


# In[ ]:


test_probs_df.to_csv('0030223_hw07_probs.csv',sep=',',index=False)

