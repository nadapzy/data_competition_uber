# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:36:35 2016

@author: Clement Peng
"""

import numpy as np
import xgboost 
import matplotlib.pyplot as plt
import matplotlib
from sklearn.learning_curve import learning_curve
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
import pylab
import requests
import os.path
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc,roc_auc_score
from sklearn.feature_selection import SelectFromModel

def to_percent(y, position):
    # This is a formatting function for showing percentage in yaxis
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

def bar_chart(df_raw,var,width,color='b',xlim=None):
    #function to visualize first trip completion % by any feature in dataframe
    #adjusting for the range of x axis
    grouped=df_raw.groupby(by=var)
    fig, ax = plt.subplots()
    N=grouped.ngroups
    ind = np.arange(N) 
    plt.bar(ind,grouped.first_completed.mean(), width=width, color=color,align='center')
    plt.xticks(ind,grouped.first_completed.mean().index)
    if xlim is not None:
        plt.xlim(xlim)
    formatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    if var in ('bgc_gap','car_add_gap'):    
        formatter1 = FuncFormatter(lambda x,pos:"%d" % (x))
        plt.gca().xaxis.set_major_formatter(formatter1)
    plt.title('First Trip Completed % by '+var)

def autolabel(rects,ax):
    # function for data labeling on top of each bar
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '% 6.2f' % float(height),
                ha='center', va='bottom')    

def bar_funnel(df):
    # creating a bar chart to represent % of success for each of the key step after drivers sign up
    # this is a chart for easy diagnosis of bottleneck in the process
    y=[len(df),sum(df.bgc_gap.notnull()),sum(df.car_add_gap.notnull()),sum(df.first_completed)]
    y=[x*1.00/len(df) for x in y]
    ind = np.arange(len(y)) 
    width=0.35
    #plt.figure(figsize=(12, 8))
    f = pylab.figure()
    ax = f.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.bar(ind,y, width=width, color='c',align='edge')
    x_labels=['Sign Up','BGC','Add Car','First Trip']
    ax.set_xticks(ind + width*1.00/2)
    ax.set_xticklabels((x_labels))
    #plt.xticks(ind,['Sign Up','BGC','Add Car','First Trip'])
    rects = ax.patches
    # Now make some labels
    autolabel(rects,ax)

def car_price(df_raw,file_name='price_v2.xlsx'):
    # function to retrieve car price from Edmund's API based on car make, model, and year
    # function writes to 'price_v2.xlsx' for further reference 
    if not os.path.isfile(file_name):
        prices=[]
        grouped=df_raw.groupby(by=['vehicle_make','vehicle_model','vehicle_year'])
        df_loop=pd.DataFrame(grouped.size()).reset_index()
        #df=df_raw.iloc[:5,]
        for index,row in df_loop.iterrows():
            # for each row in dataframe, look for vehicle make and model from SPEC: STYLE module; and return the style_id
            if not (pd.isnull(row['vehicle_make']) or pd.isnull(row['vehicle_model'])):
                if row['vehicle_year']<>np.nan:
                    style_response = requests.get("https://api.edmunds.com/api/vehicle/v2/"+row['vehicle_make']+"/"+row['vehicle_model']+"/"+str(int(row['vehicle_year']))+"/styles?fmt=json&api_key=cxn9vqyce8jxykb3n462g4fb")
                    if style_response.status_code==200:
                        if len(style_response.json()['styles'])>0:
                            style_id=style_response.json()['styles'][0]['id']
                            #for each style_id, return their true market value module
                            if style_id<>None and row['vehicle_year']<=2016:
                                price_response = requests.get("https://api.edmunds.com/v1/api/tmv/tmvservice/calculatetypicallyequippedusedtmv?styleid="+str(style_id)+"&zip=72712&fmt=json&api_key=cxn9vqyce8jxykb3n462g4fb")
                                price=price_response.json()['tmv']['nationalBasePrice']['usedTmvRetail']
                                prices.append((row.vehicle_make,row.vehicle_model,row.vehicle_year,price))
                            else:
                                price_response = requests.get("https://api.edmunds.com/v1/api/tmv/tmvservice/calculatenewtmv?styleid="+str(style_id)+"&zip=72712&fmt=json&api_key=cxn9vqyce8jxykb3n462g4fb")
                                price=price_response.json()['tmv']['nationalBasePrice']['tmv']
                                prices.append((row.vehicle_make,row.vehicle_model,row.vehicle_year,price))
            else:
                # if failed to find style or price, return np.nan
                prices.append((row.vehicle_make,row.vehicle_model,row.vehicle_year,np.nan))
            print('Finishing #{0:4d} records'.format(index))
        with pd.ExcelWriter(file_name) as writer:
            pd.DataFrame(prices,columns=['vehicle_make','vehicle_model','vehicle_year','price']) .to_excel(writer,sheet_name='Sheet1',index=False)

def model_fit(alg, X_train, y_train, performCV=True, printFeatureImportance=True, cv_folds=3):
    # function to diagnose the fit of model
    # we have precision in cross validation as the main metric, along with area under ROC, accuracy and recall.    
    # in the meanwhile, we will plot a feature importance chart

    #Fit the algorithm on the data
    alg.fit(X_train.values, y_train.values)
        
    #Predict training set:
    dtrain_predictions = alg.predict(X_train.values)
    dtrain_predprob = alg.predict_proba(X_train.values)[:,1]
    
    #Perform cross-validation:
    if performCV:
        cv_score = cross_validation.cross_val_score(alg, X_train,y_train, cv=cv_folds, scoring='precision')
    
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % accuracy_score(y_train.values, dtrain_predictions)
    print "Precision : %.4g" % precision_score(y_train.values, dtrain_predictions)
    print "Recall : %.4g" % recall_score(y_train.values, dtrain_predictions)
    print "AUC Score (Train): %f" % roc_auc_score(y_train, dtrain_predprob)
    
    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))
        
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, X_train.columns).sort_values(ascending=False)[:20]
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        
def plot_roc(model,X_train,y_train,test_size,seed):
    #plot ROC chart for diagnosis purpose    
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train, test_size=.2,random_state=seed)
    y_score = model.fit(X_train1, y_train1).predict_proba(X_test1)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes=2
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test1==i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# main program

#import data from csv
df_raw=pd.read_csv('ds_challenge_v2_1_data.csv',sep=',',parse_dates=[4,5,6,10])
df_price=pd.read_excel('price_v2.xlsx')

#feature engineering, as explained in the document
df_raw['first_completed']=1-df_raw['first_completed_date'].isnull()
df_raw['bgc_gap']=(df_raw['bgc_date']-df_raw['signup_date']).astype('timedelta64[D]')     # 39.8% not null
df_raw['bgc_completed']=1-df_raw['bgc_gap'].isnull()
#df_raw['signup_os_miss']=df_raw['signup_os'].isnull()
df_raw['car_add_gap']=(df_raw['vehicle_added_date']-df_raw['signup_date']).astype('timedelta64[D]')  #75.9% not null
df_raw['car_add_completed']=1-df_raw['vehicle_added_date'].isnull()
# new feature signup day of week: monday=0 and sunday =6
df_raw['signup_dow']=df_raw['signup_date'].dt.dayofweek.astype('int')
df_raw['car_add_dow']=df_raw['vehicle_added_date'].dt.dayofweek
df_raw['bgc_dow']=df_raw['bgc_date'].dt.dayofweek

#generate car price from Edmunds' Car Price API and output it to a excel file
#then left join w/ the orignal table, to populate price
car_price(df_raw,file_name='price_v2.xlsx')
df=pd.merge(left=df_raw,right=df_price,how='left',left_on=['vehicle_make','vehicle_model','vehicle_year'],right_on=['vehicle_make','vehicle_model','vehicle_year'])


#bar chart for most of the variables, change to true for plotting
Bar_plot=False
if Bar_plot:
    bar_chart(df,'city_name',width=0.35,color='b')
    bar_chart(df,'signup_os',width=0.35,color='r')
    bar_chart(df,'signup_channel',width=0.35,color='g')
    bar_chart(df,'signup_dow',width=0.35,color='c')
    bar_chart(df,'bgc_gap',width=0.35,color='y',xlim=(0,30))
    bar_chart(df,'car_add_gap',width=0.35,color='k',xlim=(0,30))
    bar_chart(df,'car_add_completed',width=0.35,color='m')

# plot a bar chart to understand which step lost most drivers
bar_funnel(df)

#histogram and sanity check for data
hist_vehicle_year=False
if hist_vehicle_year:
    n, bins, patches = plt.hist(df_raw[df_raw.vehicle_year.notnull()]['vehicle_year'], 50, normed=1, facecolor='green', alpha=0.75)
df.describe()



# we figured out that there are some anomalies w/i the vehicle_year and car_add_gap column
# vehicle_year: 4 vehicles, year of manufacturing is 0; we will delete them
# car_add_gap: 1 car added 5 days before the driver sisn uped; very unlikely to happen and we will delete the record
df=df[(df.vehicle_year<>0) & (df.car_add_gap<>-5)]

#export data for charting in Tableau:
#with pd.ExcelWriter('output.xlsx') as writer:
#    df.to_excel(writer,sheet_name='Sheet1',index=False)


#in order to get most of the scikit-learn model running, we need to impute all the missing values in the dataset
#1. for categorical variables, get_dummies in pandas has already handled it with grace
#2. for numerical variabless, we have bgc_gap, vehicle_year, car_add_gap, car_add_dow,bgc_dow, price
# the most intuitive way is to use Imputer in scikit learn, but in our case, all the missing values are due to user did NOT finish a certain step/process
# and thus shall inherently be different from any of the key-in value
# so my idea is to replace missing value with -1, to be different from any other values
df_miss=df.copy()
df=df.fillna(value={'bgc_gap':-1,'car_add_gap':-1,'price':-1})


# transforming all features to one-hot encoding
df_train=pd.get_dummies(df,prefix=['os','channel','city','car_year','su_dow','car_dow','bgc_dow','car_make'],columns=['signup_os','signup_channel','city_name','vehicle_year','signup_dow','car_add_dow','bgc_dow','vehicle_make'],dummy_na=True,drop_first=True)
df_train.columns


# set up general parameters for machine learning model
seed=25
y_train = df_train['first_completed']
X_train = df_train.drop(['id','first_completed','signup_date','bgc_date','vehicle_added_date','vehicle_model','first_completed_date'],axis=1)
variable_list=list(X_train.columns)
cv=StratifiedShuffleSplit(y_train,n_iter=3,test_size=0.2,random_state=seed)


#check whether all missing values are imputed
if X_train.isnull().sum().sum()==0:
    print('All imputed, dataset ready to go!')


# general xgboosting model
xgmodel = xgboost.XGBClassifier(objective= 'binary:logistic',nthread=4,seed=seed)
#xgmodel.fit(X_train.values, y_train.values)

xg_importance=pd.DataFrame(xgmodel.feature_importances_.T,index=variable_list,columns=['impo'])
print(xg_importance.sort(columns='impo',ascending=False))


# let's do a quick feature section to see how well it works  
feature_seletion=False
if feature_seletion:
    sfm = SelectFromModel(xgmodel, threshold=1e-5,prefit=False)
    sfm.fit(X_train.values, y_train.values)
    X_sfm=sfm.transform(X_train)
    model_fit(xgmodel, X_train, y_train, performCV=True, printFeatureImportance=True, cv_folds=cv)
    model_fit(xgmodel, pd.DataFrame(X_sfm), y_train, performCV=True, printFeatureImportance=True, cv_folds=cv)
#result: minimal improvement on Cross-validation score from feature selection....


#fine-tuning xgboosting
# 1st: learning rate, number of trees; 2nd: maximum depth, min_child_weight; 3rd: cosample by tree, and subsample
param_test0={'learning_rate':[0.03,0.1,0.3],\
'n_estimators':[200,300,400]}   #best 300,0.1
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}                                      #max_depth=3; min_child_weight=1    score: 0.7303

param_test2 = {
 'max_depth':[2,3,4],
 'min_child_weight':[4,5,6]            #max_depth=3; min_child_wegith=6
}        
param_test3={'colsample_bytree':[0.55,0.6,0.65,0.7],\
'subsample':[0.7,0.75,0.8,0.85]}               #colsample_bytree= 0.85, subsample: 0.65
gsearch1 = GridSearchCV(estimator = xgmodel,\
param_grid = param_test3, scoring='precision',n_jobs=4,iid=False, cv=cv)
xg_gridsearch=False
if xg_gridsearch:
    gsearch1.fit(X_train,y_train)
    gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


#final model, diagnosis and feature importance 
xgmodel = xgboost.XGBClassifier(learning_rate =0.1,n_estimators=300,max_depth=3,min_child_weight=6,gamma=0,subsample=0.85,colsample_bytree=0.65,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=seed)
model_fit(xgmodel, X_train, y_train, performCV=True, printFeatureImportance=True, cv_folds=cv)


#random forecast model
rfmodel = RandomForestClassifier(state=seed)
rfmodel.fit(X_train, y_train)
print(rfmodel.feature_importances_)
rf_importance=pd.DataFrame(rfmodel.feature_importances_.T,index=variable_list,columns=['impo'])
pd.options.display.float_format = '{:,.3f}'.format
#print(rf_importance.sort(columns='impo',ascending=False))
cv_score = cross_validation.cross_val_score(rfmodel,  X_train,y_train,cv=cv,  scoring='precision')
model_fit(rfmodel, X_train, y_train, performCV=True, printFeatureImportance=True, cv_folds=cv)


#fine-tuning random forest
param_rf_test0={'n_estimators':[300,500,700,900]}  #best:700
param_rf_test1 = {
 'max_depth':range(3,10,2),
 'max_features':['auto','sqrt','log2'],
 'min_weight_fraction_leaf':[0,0.001,0.003,0.01]   #best: 'sqrt' , min_weight_fraction=0, max_depth=9
}
param_rf_test2 = {
 'max_depth':[8,9,10],
 'max_features':['sqrt'],
 'min_weight_fraction_leaf':[0,0.001,0.003]   #best: 'sqrt' , min_weight_fraction=0, max_depth=10
}

gsearchrf1 = GridSearchCV(estimator = rfmodel,\
param_grid = param_rf_test2, scoring='precision',n_jobs=4,iid=False, cv=cv)
rf_gridsearch=False
if rf_gridsearch:
    gsearchrf1.fit(X_train,y_train)
    gsearchrf1.grid_scores_, gsearchrf1.best_params_, gsearchrf1.best_score_

#final model, diagnosis and feature importance
rfmodel = RandomForestClassifier(n_estimators=700,max_features='sqrt',\
min_weight_fraction_leaf=0,max_depth=10,random_state=seed,n_jobs=4)    
model_fit(rfmodel, X_train, y_train, performCV=True, printFeatureImportance=True, cv_folds=cv)


