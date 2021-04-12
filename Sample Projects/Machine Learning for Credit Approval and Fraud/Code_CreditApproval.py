# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:05:11 2021

@author: mhaskariyeh
"""
in_path = '...'
out_path = '...'

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
#from sklearn.externals import joblib
import seaborn as sns
from datetime import datetime
import os
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import random

##set the random seed for reproduciblity
random.seed(0)
# =============================================================================
# 
# =============================================================================

df = pd.read_csv(f'{in_path}crxdata.csv', header = 0)
df.replace({"A9" : { "f" : 0, "t" : 1},
            "A10" : { "f" : 0, "t" : 1},
            "A12" : { "f" : 0, "t" : 1},
            "A16" : { "-" : 0, "+" : 1}}, inplace = True)
df.replace("?", np.nan, inplace = True)

df = df.astype({"A2":np.float64, 'A14':np.float64})

to_get_dummies = ["A1", "A4", "A5", "A6", "A7", "A13"]
df = pd.get_dummies(df,
                    columns=to_get_dummies, # List of column names
                    drop_first = True)

# =============================================================================
# 
# =============================================================================

# Plot Variables Percentages
df_copy = df.copy()
df_yr = df_copy['A16'].value_counts()
df_yr = df_yr.reset_index()
df_yr ['index'] = ['Rejected', 'Approved']
df_yr ['index'] = df_yr ['index'].astype(str)
color = df_yr['A16'].astype(str)

fig= px.bar(df_yr,
            x = 'index',
            y = df_yr['A16']/df_yr['A16'].sum(),
            color = color,
            title='Credit Card Approval Rate',
            labels = dict(index = 'Approval or Rejection',
                          y = 'Percentage',
                          color = 'Approval or Rejection<br>(Counts)')
            )
fig.update_layout(yaxis = dict(tickformat = ',.0%')) 
fig.write_html(f'{out_path}plots/CreditCardApproval_1.html') 


# Drop response variable before imputation
response_variable = "A16"
columns = df.columns
response = df[[response_variable]]
df = df.drop(response_variable, axis = 1)
#imp = SimpleImputer(missing_values=np.nan,
#                    strategy="mean")
imp = IterativeImputer(random_state=0,
                       estimator=ExtraTreesRegressor(n_estimators=10,
                                                     random_state=None))
imp.fit(df)
columns = df.columns
data = pd.DataFrame(imp.transform(df), columns = columns)
X_train, X_test, y_train, y_test = train_test_split(data, response,
                                                    test_size=0.2,
                                                    stratify = response,
                                                    random_state=0)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# preparing response columns to feed to classifiers
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()


                                    
# =============================================================================
# 'Random_Forest_param_grid' : dict(clf__n_estimators = [250, 500, 1000],
#                           clf__min_samples_split = [2, 4, 6, 8, 10]),
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
                
classifier =  'Random Forest'

# 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000
# 250, 500, 1000
n_estimators = [50, 100, 250, 500, 750, 1000]
min_samples_splits = [2, 4, 6, 8, 10]

df_pr =  pd.DataFrame(columns = ['N_Estimator',
                                 'Min Sample Split',
                                 'f1_score',
                                 'Precision',
                                 'Recall',
                                 'Support',
                                 'AUC',
                                 'Accuracy'])

for n_estimator in n_estimators:
    for min_samples_split  in min_samples_splits:
        clf = RandomForestClassifier(n_estimators = n_estimator,
                                     min_samples_split = min_samples_split,
                                     random_state = None)

        clf.fit(X_train, y_train)        
        predict = clf.predict(X_test)
        AUC_score = clf.score(X_test, y_test)
        report = classification_report(y_test, predict, target_names=["No", "Yes"])
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            #print(line)
            row = {}
            row_data = line.split(' ')
            row_data = list(filter(None, row_data))
            if len(row_data)>0:
                if row_data[0] == "accuracy":
                    row[response_variable] = row_data[0]
                    row['Precision'] = None
                    row['Recall'] = None
                    row['f1_score'] = float(row_data[1])
                    row['Support'] = float(row_data[2])
                else:
                    row[response_variable] = row_data[0]
                    row['Precision'] = float(row_data[1])
                    row['Recall'] = float(row_data[2])
                    row['f1_score'] = float(row_data[3])
                    row['Support'] = float(row_data[4])
                report_data.append(row)
        row = {}
        row[response_variable] = 'AUC'
        row['Precision'] = None
        row['Recall'] = None
        row['f1_score'] = AUC_score
        row['Support'] = None
        report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe['N_Estimator'] = n_estimator
        dataframe['Min Sample Split'] = min_samples_split
        print(f"The AUC score for random forest with N Estimator = {n_estimator} and Min Sample Split = {min_samples_split} is: {AUC_score}")
        df_pr = pd.concat([df_pr, dataframe], ignore_index=True)
        


        from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, accuracy_score, confusion_matrix
        y_score = clf.predict_proba(X_test)[:, 1]
        y_score_list = list(set(y_score))
        y_score_list.sort()
        
        
        sens = []
        spec = []
        for current_threshold in y_score_list:
            current_y_pred = np.where(y_score>=current_threshold,1,0)
            tn, fp, fn, tp = confusion_matrix(y_test,current_y_pred).ravel()
            sens.append(tp/(tp+fn))
            spec.append(tn/(tn+fp))
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred, adjusted = False)
        
        roc_fig = px.area(x=fpr,
                          y=tpr,
                          title=f'ROC Curve (AUC={auc(fpr, tpr):.4f}, Accuracy={accuracy:.4f}, Balanced Accuracy={balanced_accuracy:.4f})',
                          labels=dict(x='False Positive Rate (1-Specificity)',
                                      y='True Positive Rate (Sensitivity)'),
                          width=1000,
                          height=800,
                          )
        roc_fig.add_shape(type='line',
                          line=dict(dash='dash'),
                          x0=0,
                          x1=1,
                          y0=0,
                          y1=1
                          )
        
        roc_fig.update_yaxes(scaleanchor="x", scaleratio=1)
        roc_fig.update_xaxes(constrain='domain')
        roc_fig.update_traces(hovertemplate ='Sensitivity: %{y:.2f}<br>1-Specificity: %{x:.2f}<extra></extra>')
        roc_fig.write_html(f'{out_path}rf/{n_estimator}_{min_samples_split}_ROC.html')
        
        sens_spec_fig = go.Figure()
        
        # Add traces
        sens_spec_fig.add_trace(go.Scatter(x=y_score_list,
                                           y=sens,
                                           mode='lines',
                                           name='Sensitivity'))
        sens_spec_fig.add_trace(go.Scatter(x=y_score_list,
                                           y=spec,
                                           mode='lines',
                                           name='Specificity'))
        sens_spec_fig.update_layout(title = f"Sensitivity and Specificity for Different Predicted Probability Threshold and {n_estimator}_{min_samples_split}",
                                    xaxis=dict(title = "Probaility cut-off"),
                                    yaxis=dict(title = "Percent",),
                                    hovermode="x unified")
        sens_spec_fig.write_html(f'{out_path}rf/{n_estimator}_{min_samples_split}_Sensitivity_Specificity.html')
            
# =============================================================================
# # Classify the data and plo the learning curve
# =============================================================================

df_pr_rf_ac = df_pr[df_pr['A16'] == 'accuracy']
df_pr_rf_au = df_pr[df_pr['A16'] == 'AUC']

fig = px.line(df_pr_rf_ac, 
              x='N_Estimator', 
              y='f1_score', 
              color='Min Sample Split',
              title ='Accuracy of Random Forest- Credit Approval',
              labels = dict(f1_score = 'Accuracy',
                            N_Estimator = 'The number of trees in the forest'))
fig.write_html(f'{out_path}plots/CreditApproval_rf_accuracy.html') 

fig = px.line(df_pr_rf_au,
              x='N_Estimator', 
              y='f1_score', 
              color='Min Sample Split',
              title ='AUC for Random Forest- Credit Approval',
              labels = dict(f1_score = 'AUC'))
fig.write_html(f'{out_path}plots/CreditApproval_rf_auc.html') 

# exit()
# =============================================================================
# 'Gradient_boosted_regression_trees_param_grid' : dict(clf__n_estimators = [250, 500, 1000],
#                                                   clf__learning_rate = [.001, 0.01, 0.05, 0.1],
#                                                   clf__max_depth = [3, 5]),
# =============================================================================

from sklearn.ensemble import GradientBoostingClassifier
                
classifier =  'Gradient boosted regression trees'

n_estimators = [250, 500, 1000]
learning_rates = [.001, 0.01, 0.05, 0.1]
max_depths = [3, 5]


df_pr =  pd.DataFrame(columns = ['N_Estimators',
                                 'Learning_Rate',
                                 'f1_score',
                                 'Precision',
                                 'Recall',
                                 'Support',
                                 'AUC',
                                 'Accuracy'])

for n_estimator in n_estimators:
    for learning_rate  in learning_rates:
        for max_depth  in max_depths:
            clf = GradientBoostingClassifier(n_estimators = n_estimator,
                                             learning_rate = learning_rate,
                                             max_depth = max_depth,
                                             random_state = None)
    
            clf.fit(X_train, y_train)        
            predict = clf.predict(X_test)
            AUC_score = clf.score(X_test, y_test)
            report = classification_report(y_test, predict, target_names=["No", "Yes"])
            report_data = []
            lines = report.split('\n')
            for line in lines[2:-3]:
                #print(line)
                row = {}
                row_data = line.split(' ')
                row_data = list(filter(None, row_data))
                if len(row_data)>0:
                    if row_data[0] == "accuracy":
                        row[response_variable] = row_data[0]
                        row['Precision'] = None
                        row['Recall'] = None
                        row['f1_score'] = float(row_data[1])
                        row['Support'] = float(row_data[2])
                    else:
                        row[response_variable] = row_data[0]
                        row['Precision'] = float(row_data[1])
                        row['Recall'] = float(row_data[2])
                        row['f1_score'] = float(row_data[3])
                        row['Support'] = float(row_data[4])
                    report_data.append(row)
            row = {}
            row[response_variable] = 'AUC'
            row['Precision'] = None
            row['Recall'] = None
            row['f1_score'] = AUC_score
            row['Support'] = None
            report_data.append(row)
            dataframe = pd.DataFrame.from_dict(report_data)
            dataframe['N_Estimators'] = n_estimator
            dataframe['Learning_Rate'] = learning_rate
            dataframe['Max_Depth'] = max_depth
            print(f"The AUC score for GBR with n_estimators = {n_estimator} and learning rate = {learning_rate} and max depth = {max_depth} is: {AUC_score}")
            df_pr = pd.concat([df_pr, dataframe], ignore_index=True)
            
    
    
            from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, accuracy_score, confusion_matrix
            y_score = clf.predict_proba(X_test)[:, 1]
            y_score_list = list(set(y_score))
            y_score_list.sort()
            
            
            sens = []
            spec = []
            for current_threshold in y_score_list:
                current_y_pred = np.where(y_score>=current_threshold,1,0)
                tn, fp, fn, tp = confusion_matrix(y_test,current_y_pred).ravel()
                sens.append(tp/(tp+fn))
                spec.append(tn/(tn+fp))
            fpr, tpr, thresholds = roc_curve(y_test, y_score)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred, adjusted = False)
            
            roc_fig = px.area(x=fpr,
                              y=tpr,
                              title=f'ROC Curve (AUC={auc(fpr, tpr):.4f}, Accuracy={accuracy:.4f}, Balanced Accuracy={balanced_accuracy:.4f})',
                              labels=dict(x='False Positive Rate (1-Specificity)',
                                          y='True Positive Rate (Sensitivity)'),
                              width=1000,
                              height=800,
                              )
            roc_fig.add_shape(type='line',
                              line=dict(dash='dash'),
                              x0=0,
                              x1=1,
                              y0=0,
                              y1=1
                              )
            
            roc_fig.update_yaxes(scaleanchor="x", scaleratio=1)
            roc_fig.update_xaxes(constrain='domain')
            roc_fig.update_traces(hovertemplate ='Sensitivity: %{y:.2f}<br>1-Specificity: %{x:.2f}<extra></extra>')
            roc_fig.write_html(f'{out_path}gbr/{n_estimator}_{learning_rate}_{max_depth}_ROC.html')
            
            sens_spec_fig = go.Figure()
            
            # Add traces
            sens_spec_fig.add_trace(go.Scatter(x=y_score_list,
                                               y=sens,
                                               mode='lines',
                                               name='Sensitivity'))
            sens_spec_fig.add_trace(go.Scatter(x=y_score_list,
                                               y=spec,
                                               mode='lines',
                                               name='Specificity'))
            sens_spec_fig.update_layout(title = f"Sensitivity and Specificity for Different Predicted Probability Threshold and {n_estimator}_{learning_rate}_{max_depth}",
                                        xaxis=dict(title = "Probaility cut-off"),
                                        yaxis=dict(title = "Percent",),
                                        hovermode="x unified")
            sens_spec_fig.write_html(f'{out_path}gbr/{n_estimator}_{learning_rate}_{max_depth}_Sensitivity_Specificity.html')
                
# =============================================================================
# # Classify the data and plo the learning curve
# =============================================================================

df_pr_GB_ac = df_pr[df_pr['A16'] == 'accuracy']
df_pr_GB_au = df_pr[df_pr['A16'] == 'AUC']

color_df_pr_GB_ac = df_pr_GB_ac['Max_Depth'].astype(str)
fig = px.scatter_3d(df_pr_GB_ac, 
                    x='N_Estimators', 
                    y='Learning_Rate', 
                    z='f1_score',
                    color=color_df_pr_GB_ac,
                    title = 'Accuracy of Gradient Boosting Regression- Credit Approval',
                    labels = dict(f1_score = 'Accuracy',
                                  N_Estimators = 'The number of boosting<br>stages to perform',
                                  Learning_Rate = 'Learning Rate',
                                  color = 'Maximum Depth'))
fig.write_html(f'{out_path}plots/CreditApproval_GB_ac.html') 

color_df_pr_GB_au = df_pr_GB_au['Max_Depth'].astype(str)
fig = px.scatter_3d(df_pr_GB_au, 
                    x='N_Estimators', 
                    y='Learning_Rate', 
                    z='f1_score',
                    color=color_df_pr_GB_au,
                    title = 'AUC of Gradient Boosting Regression- Credit Approval',
                    labels = dict(f1_score = 'AUC',
                                  N_Estimators = 'The number of boosting stages to perform',
                                  Learning_Rate = 'Learning Rate',
                                  color = 'Maximum Depth'))
fig.write_html(f'{out_path}plots/CreditApproval_GB_auc.html') 


# =============================================================================
# 'K-Nearest_Neighbors_param_grid' : dict(clf__n_neighbors = [20, 25, 30],
#                                      clf__weights = ['uniform', 'distance'])
# =============================================================================

from sklearn.neighbors import KNeighborsClassifier
                
classifier =  'KNeighborsClassifier'

n_neighbors =  [2, 3, 5, 10, 15, 20, 25, 30, 35]  # range(5, 30, 1)
weights = ['uniform', 'distance']

df_pr =  pd.DataFrame(columns = ['Weight',
                                 'Neighbors',
                                 'f1_score',
                                 'Precision',
                                 'Recall',
                                 'Support',
                                 'AUC',
                                 'Accuracy'])

for weight in weights:
    for n_neighbor  in n_neighbors:
        clf = KNeighborsClassifier(weights = weight,
                                   n_neighbors=n_neighbor)

        clf.fit(X_train, y_train)
        
        predict = clf.predict(X_test)
        AUC_score = clf.score(X_test, y_test)
        report = classification_report(y_test, predict, target_names=["No", "Yes"])
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            #print(line)
            row = {}
            row_data = line.split(' ')
            row_data = list(filter(None, row_data))
            if len(row_data)>0:
                if row_data[0] == "accuracy":
                    row[response_variable] = row_data[0]
                    row['Precision'] = None
                    row['Recall'] = None
                    row['f1_score'] = float(row_data[1])
                    row['Support'] = float(row_data[2])
                else:
                    row[response_variable] = row_data[0]
                    row['Precision'] = float(row_data[1])
                    row['Recall'] = float(row_data[2])
                    row['f1_score'] = float(row_data[3])
                    row['Support'] = float(row_data[4])
                report_data.append(row)
        row = {}
        row[response_variable] = 'AUC'
        row['Precision'] = None
        row['Recall'] = None
        row['f1_score'] = AUC_score
        row['Support'] = None
        report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe['Weight'] = weight
        dataframe['Neighbors'] = n_neighbor
        print(f"The AUC score for KNN with weight = {weight} and neighbor = {n_neighbor} is: {AUC_score}")
        df_pr = pd.concat([df_pr, dataframe], ignore_index=True)
        

        from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, accuracy_score, confusion_matrix
        y_score = clf.predict_proba(X_test)[:, 1]
        y_score_list = list(set(y_score))
        y_score_list.sort()
        
        
        sens = []
        spec = []
        for current_threshold in y_score_list:
            current_y_pred = np.where(y_score>=current_threshold,1,0)
            tn, fp, fn, tp = confusion_matrix(y_test,current_y_pred).ravel()
            sens.append(tp/(tp+fn))
            spec.append(tn/(tn+fp))
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred, adjusted = False)
        
        roc_fig = px.area(x=fpr,
                          y=tpr,
                          title=f'ROC Curve (AUC={auc(fpr, tpr):.4f}, Accuracy={accuracy:.4f}, Balanced Accuracy={balanced_accuracy:.4f})',
                          labels=dict(x='False Positive Rate (1-Specificity)',
                                      y='True Positive Rate (Sensitivity)'),
                          width=1000,
                          height=800,
                          )
        roc_fig.add_shape(type='line',
                          line=dict(dash='dash'),
                          x0=0,
                          x1=1,
                          y0=0,
                          y1=1
                          )
        
        roc_fig.update_yaxes(scaleanchor="x", scaleratio=1)
        roc_fig.update_xaxes(constrain='domain')
        roc_fig.update_traces(hovertemplate ='Sensitivity: %{y:.2f}<br>1-Specificity: %{x:.2f}<extra></extra>')
        roc_fig.write_html(f'{out_path}knn/{weight}_{n_neighbor}_ROC.html')
        
        sens_spec_fig = go.Figure()
        
        # Add traces
        sens_spec_fig.add_trace(go.Scatter(x=y_score_list,
                                           y=sens,
                                           mode='lines',
                                           name='Sensitivity'))
        sens_spec_fig.add_trace(go.Scatter(x=y_score_list,
                                           y=spec,
                                           mode='lines',
                                           name='Specificity'))
        sens_spec_fig.update_layout(title = f"Sensitivity and Specificity for Different Predicted Probability Threshold and {weight}_{n_neighbor}",
                                    xaxis=dict(title = "Probaility cut-off"),
                                    yaxis=dict(title = "Percent",),
                                    hovermode="x unified")
        sens_spec_fig.write_html(f'{out_path}knn/{weight}_{n_neighbor}_Sensitivity_Specificity.html')

# =============================================================================
# # Classify the data and plo the learning curve
# =============================================================================

df_pr_knn_ac = df_pr[df_pr['A16'] == 'accuracy']
df_pr_knn_au = df_pr[df_pr['A16'] == 'AUC']


fig = px.line(df_pr_knn_ac, 
              x='Neighbors', 
              y='f1_score', 
              color='Weight',
              title ='Accuracy of KNN- Credit Approval',
              labels = dict(f1_score = 'Accuracy',
                            Neighbors = 'Number of neighbors'))
fig.write_html(f'{out_path}plots/CreditApproval_knn_accuracy.html') 

fig = px.line(df_pr_knn_au,
              x='Neighbors', 
              y='f1_score', 
              color='Weight',
              title ='AUC of KNN- Credit Approval',
              labels = dict(f1_score = 'AUC',
                            Neighbors = 'Number of neighbors'))
fig.write_html(f'{out_path}plots/CreditApproval_knn_auc.html')             

# =============================================================================
# 'Support Vector Machine_param_grid' : dict(clf__C=[0.1, 1, 10, 100],
#                                        clf__gamma=[0.001, 0.01, 0.1, 1, 10]),
# =============================================================================

from sklearn.svm import SVC
                
classifier =  'SVM'

Cs = [0.1, 1, 10, 100]
gammas = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5, 6, 7, 8, 9, 10]

df_pr =  pd.DataFrame(columns = ['C',
                                 'Gamma',
                                 'f1_score',
                                 'Precision',
                                 'Recall',
                                 'Support',
                                 'AUC',
                                 'Accuracy'])

for C in Cs:
    for gamma  in gammas:
        clf = SVC(C = C,
                  gamma = gamma,
                  probability=True)

        clf.fit(X_train, y_train)        
        predict = clf.predict(X_test)
        AUC_score = clf.score(X_test, y_test)
        report = classification_report(y_test, predict, target_names=["No", "Yes"])
        report_data = []
        lines = report.split('\n')
        for line in lines[2:-3]:
            #print(line)
            row = {}
            row_data = line.split(' ')
            row_data = list(filter(None, row_data))
            if len(row_data)>0:
                if row_data[0] == "accuracy":
                    row[response_variable] = row_data[0]
                    row['Precision'] = None
                    row['Recall'] = None
                    row['f1_score'] = float(row_data[1])
                    row['Support'] = float(row_data[2])
                else:
                    row[response_variable] = row_data[0]
                    row['Precision'] = float(row_data[1])
                    row['Recall'] = float(row_data[2])
                    row['f1_score'] = float(row_data[3])
                    row['Support'] = float(row_data[4])
                report_data.append(row)
        row = {}
        row[response_variable] = 'AUC'
        row['Precision'] = None
        row['Recall'] = None
        row['f1_score'] = AUC_score
        row['Support'] = None
        report_data.append(row)
        dataframe = pd.DataFrame.from_dict(report_data)
        dataframe['C'] = C
        dataframe['Gamma'] = gamma
        print(f"The AUC score for SVM with C = {C} and Gamma = {gamma} is: {AUC_score}")
        df_pr = pd.concat([df_pr, dataframe], ignore_index=True)
        


        from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, accuracy_score, confusion_matrix
        y_score = clf.predict_proba(X_test)[:, 1]
        y_score_list = list(set(y_score))
        y_score_list.sort()
        
        
        sens = []
        spec = []
        for current_threshold in y_score_list:
            current_y_pred = np.where(y_score>=current_threshold,1,0)
            tn, fp, fn, tp = confusion_matrix(y_test,current_y_pred).ravel()
            sens.append(tp/(tp+fn))
            spec.append(tn/(tn+fp))
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred, adjusted = False)
        
        roc_fig = px.area(x=fpr,
                          y=tpr,
                          title=f'ROC Curve (AUC={auc(fpr, tpr):.4f}, Accuracy={accuracy:.4f}, Balanced Accuracy={balanced_accuracy:.4f})',
                          labels=dict(x='False Positive Rate (1-Specificity)',
                                      y='True Positive Rate (Sensitivity)'),
                          width=1000,
                          height=800,
                          )
        roc_fig.add_shape(type='line',
                          line=dict(dash='dash'),
                          x0=0,
                          x1=1,
                          y0=0,
                          y1=1
                          )
        
        roc_fig.update_yaxes(scaleanchor="x", scaleratio=1)
        roc_fig.update_xaxes(constrain='domain')
        roc_fig.update_traces(hovertemplate ='Sensitivity: %{y:.2f}<br>1-Specificity: %{x:.2f}<extra></extra>')
        roc_fig.write_html(f'{out_path}svm/{C}_{gamma}_ROC.html')
        
        sens_spec_fig = go.Figure()
        
        # Add traces
        sens_spec_fig.add_trace(go.Scatter(x=y_score_list,
                                           y=sens,
                                           mode='lines',
                                           name='Sensitivity'))
        sens_spec_fig.add_trace(go.Scatter(x=y_score_list,
                                           y=spec,
                                           mode='lines',
                                           name='Specificity'))
        sens_spec_fig.update_layout(title = f"Sensitivity and Specificity for Different Predicted Probability Threshold and {C}_{gamma}",
                                    xaxis=dict(title = "Probaility cut-off"),
                                    yaxis=dict(title = "Percent",),
                                    hovermode="x unified")
        sens_spec_fig.write_html(f'{out_path}svm/{C}_{gamma}_Sensitivity_Specificity.html')

# =============================================================================
# # Classify the data and plo the learning curve
# =============================================================================

df_pr_svm_ac = df_pr[df_pr['A16'] == 'accuracy']
df_pr_svm_au = df_pr[df_pr['A16'] == 'AUC']


fig = px.line(df_pr_svm_ac, 
              x='Gamma', 
              y='f1_score', 
              color='C',
              title ='Accuracy of SVM- Credit Approval',
              labels = dict(f1_score = 'Accuracy',
                            C = 'Regularization<br>Parameter (C)'))
fig.write_html(f'{out_path}plots/CreditApproval_svm_accuracy.html') 

fig = px.line(df_pr_svm_au,
              x='Gamma', 
              y='f1_score', 
              color='C',
              title ='AUC for SVM- Credit Approval',
              labels = dict(f1_score = 'AUC',
                            C = 'Regularization<br>Parameter (C)'))
fig.write_html(f'{out_path}plots/CreditApproval_svm_auc.html')  

# exit()
# =============================================================================
# 'Neural_network_param_grid' : dict(clf__hidden_layer_sizes = [(50,10), (50,20), (50,30), (40,10), (40,20), (30,10), (30), (20), (10)],
#                                clf__alpha = [0.00001, 0.0001, 0.001, 0.01],
#                                clf__activation = ['logistic', 'relu', 'tanh']),
# =============================================================================
            

from sklearn.neural_network import MLPClassifier
                
classifier =  'Neural Network'

hidden_layer_sizes = [(40,10), (40,20), (37,30), (25,10), (20,10), (20,5), (30), (20), (10)]
alphas = [0.00001, 0.0001, 0.001, 0.01] # 0.000001,  , 0.1
activations = ['logistic', 'relu', 'tanh']

df_pr =  pd.DataFrame(columns = ['Hidden Layer Size',
                                 'Alpha',
                                 'Activation',
                                 'f1_score',
                                 'Precision',
                                 'Recall',
                                 'Support',
                                 'AUC',
                                 'Accuracy'])

for hidden_layer_size in hidden_layer_sizes:
    for alpha  in alphas:
        for activation  in activations:
            clf = MLPClassifier(hidden_layer_sizes = hidden_layer_size,
                                alpha = alpha,
                                activation = activation,
                                random_state=None,
                                max_iter = 500)
    
            clf.fit(X_train, y_train)        
            predict = clf.predict(X_test)
            AUC_score = clf.score(X_test, y_test)
            report = classification_report(y_test, predict, target_names=["No", "Yes"])
            report_data = []
            lines = report.split('\n')
            for line in lines[2:-3]:
                #print(line)
                row = {}
                row_data = line.split(' ')
                row_data = list(filter(None, row_data))
                if len(row_data)>0:
                    if row_data[0] == "accuracy":
                        row[response_variable] = row_data[0]
                        row['Precision'] = None
                        row['Recall'] = None
                        row['f1_score'] = float(row_data[1])
                        row['Support'] = float(row_data[2])
                    else:
                        row[response_variable] = row_data[0]
                        row['Precision'] = float(row_data[1])
                        row['Recall'] = float(row_data[2])
                        row['f1_score'] = float(row_data[3])
                        row['Support'] = float(row_data[4])
                    report_data.append(row)
            row = {}
            row[response_variable] = 'AUC'
            row['Precision'] = None
            row['Recall'] = None
            row['f1_score'] = AUC_score
            row['Support'] = None
            report_data.append(row)
            dataframe = pd.DataFrame.from_dict(report_data)
            dataframe['Hidden Layer Size'] = str(hidden_layer_size)
            dataframe['Alpha'] = alpha
            dataframe['Activation'] = activation
            print(f"The AUC score for nn with hidden_layer_size = {hidden_layer_size} and alpha = {alpha} and activation = {activation} is: {AUC_score}")
            df_pr = pd.concat([df_pr, dataframe], ignore_index=True)
            
    
    
            from sklearn.metrics import roc_curve, auc, balanced_accuracy_score, accuracy_score, confusion_matrix
            y_score = clf.predict_proba(X_test)[:, 1]
            y_score_list = list(set(y_score))
            y_score_list.sort()
            
            
            sens = []
            spec = []
            for current_threshold in y_score_list:
                current_y_pred = np.where(y_score>=current_threshold,1,0)
                tn, fp, fn, tp = confusion_matrix(y_test,current_y_pred).ravel()
                sens.append(tp/(tp+fn))
                spec.append(tn/(tn+fp))
            fpr, tpr, thresholds = roc_curve(y_test, y_score)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred, adjusted = False)
            
            roc_fig = px.area(x=fpr,
                              y=tpr,
                              title=f'ROC Curve (AUC={auc(fpr, tpr):.4f}, Accuracy={accuracy:.4f}, Balanced Accuracy={balanced_accuracy:.4f})',
                              labels=dict(x='False Positive Rate (1-Specificity)',
                                          y='True Positive Rate (Sensitivity)'),
                              width=1000,
                              height=800,
                              )
            roc_fig.add_shape(type='line',
                              line=dict(dash='dash'),
                              x0=0,
                              x1=1,
                              y0=0,
                              y1=1
                              )
            
            roc_fig.update_yaxes(scaleanchor="x", scaleratio=1)
            roc_fig.update_xaxes(constrain='domain')
            roc_fig.update_traces(hovertemplate ='Sensitivity: %{y:.2f}<br>1-Specificity: %{x:.2f}<extra></extra>')
            roc_fig.write_html(f'{out_path}nn/{hidden_layer_size}_{alpha}_{activation}_ROC.html')
            
            sens_spec_fig = go.Figure()
            
            # Add traces
            sens_spec_fig.add_trace(go.Scatter(x=y_score_list,
                                               y=sens,
                                               mode='lines',
                                               name='Sensitivity'))
            sens_spec_fig.add_trace(go.Scatter(x=y_score_list,
                                               y=spec,
                                               mode='lines',
                                               name='Specificity'))
            sens_spec_fig.update_layout(title = f"Sensitivity and Specificity for Different Predicted Probability Threshold and {hidden_layer_size}_{alpha}_{activation}",
                                        xaxis=dict(title = "Probaility cut-off"),
                                        yaxis=dict(title = "Percent",),
                                        hovermode="x unified")
            sens_spec_fig.write_html(f'{out_path}nn/{hidden_layer_size}_{alpha}_{activation}_Sensitivity_Specificity.html')
        
# =============================================================================
# # Classify the data and plo the learning curve
# =============================================================================

df_pr_nn_ac = df_pr[df_pr['A16'] == 'accuracy']
df_pr_nn_au = df_pr[df_pr['A16'] == 'AUC']

ystr = df_pr_nn_ac['Alpha'].astype(str)
fig = px.scatter_3d(df_pr_nn_ac, 
                    x='Hidden Layer Size', 
                    y=ystr, 
                    z='f1_score',
                    color='Activation',
                    title ='Accuracy of Neural Network- Credit Approval',
                    labels = dict(f1_score = 'Accuracy',
                                  y = 'Alpha',
                                  Activation = 'Activation Function'))
fig.write_html(f'{out_path}plots/CreditApproval_NN_ac.html') 

ystr = df_pr_nn_au['Alpha'].astype(str)
fig = px.scatter_3d(df_pr_nn_au, 
                    x = 'Hidden Layer Size', 
                    y = ystr, 
                    z = 'f1_score',
                    color = 'AUC',
                    labels = dict(f1_score = 'Accuracy',
                                  y = 'Alpha',
                                  Activation = 'Activation Function'))
fig.write_html(f'{out_path}plots/CreditApproval_NN_auc.html') 




