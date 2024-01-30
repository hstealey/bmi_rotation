# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:12:58 2023

@author: hanna
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold #RepeatedStratifiedKFold, KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 


##from statsmodels.stats.power import TTestIndPower

'___________________________________________________'
def Ftest(v1, v2):
    'F-test: are variances equal?'
    'https://www.geeksforgeeks.org/how-to-perform-an-f-test-in-python/'
    n1, n2 = len(v1), len(v2)
    m1, m2 = np.mean(v1), np.mean(v2)
    var1, var2 = np.var(v1, ddof=1), np.var(v2, ddof=1)

    #F-test:  are variances equal?
    f = var1/var2
    num_DoF = n1-1
    den_DoF = n2-1
    p_value = 1-stats.f.cdf(f,num_DoF,den_DoF)
    print('F-Test: {:.4f} ({:.4f})'.format(f,p_value))

'___________________________________________________'

# def effectSize_means(var1, var2):
#     n1 = len(var1)
#     n2 = len(var2)
    
#     s1 = np.std(var1, ddof=1)
#     s2 = np.std(var2, ddof=1)
    
#     m1 = np.mean(var1)
#     m2 = np.mean(var2)
    
#     DOF = n1 + n2 - 2
#     s_pooled = np.sqrt( ( (n1-1)*(s1**2) + (n2-1)*(s2**2) )/DOF)

#     ESsmd = (m1 - m2)/(s_pooled*np.sqrt((1/n1)+(1/n2)))
    
#     Vsmd = ((n1+n2)/(n1*n2)) + (ESsmd)**2/(2*(n1+n2))
#     SEsmd = np.sqrt(Vsmd)
    
#     ta2 = stats.t.isf(0.05/2, df=DOF)
    
#     CI_lower = ESsmd - (ta2*SEsmd)
#     CI_upper = ESsmd + (ta2*SEsmd)
    
#     return(ESsmd, CI_lower, CI_upper)

'_____________________________________________________________________________'


def sigSlope(X_,y, alpha, returnSE=False):
    
    n1 = len(X_)
    n2 = len(y)
    
    dof = n1-1
    
    'Slope'
    X = np.array(X_).reshape(-1,1)

    model = LinearRegression()
    res = model.fit(X,y)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    

    'Standard Error of Slope (se)'
    yhat = slope * X[:,0] + intercept
    SSE = np.sum((y-yhat)**2)
    MSE = SSE/(n1-2)
    SEMx = np.sum( (X[:,0]-np.mean(X[:,0]))**2 )
    mult = 1/(n1-2)
    
    se = np.sqrt(mult*(SSE/SEMx))
    #alt_se = np.sqrt(MSE)/np.sqrt(SEMx)

    
    'Test Statistic'    
    t_test = slope/se


    'p-value' #t distribution
    p = stats.t.sf(np.abs(t_test), df=dof)*2
    
    
    'Confidence Interval'
    t_crit =  stats.t.isf(alpha/2, df=dof)

    LL = slope-(t_crit*se)
    UL = slope+(t_crit*se)
    
    print('\tslope: {:.4f} ({:.4e})({:.2f} {:.2f})'.format(slope, p, LL, UL))

    print('\tModel Score: {:.4f}'.format(model.score(X,y)))
    
    if returnSE == False:
        return(slope, intercept, p, LL, UL)
    else:
        return(slope,se,n1)
    
'_____________________________________________________________________________'

# def cvLDA(dComp, X, y, n_folds, n_repeats):
#     ind = dComp['s_ind']
#     deg = np.abs(dComp['s_degs'])
    
#     model = LinearDiscriminantAnalysis()
    
#     scoresMODEL   = []
#     scoresCHANCE  = []
            
#     for i in range(n_repeats):
    
#         'Model Performance'
#         cv = StratifiedKFold(n_splits=n_folds)
#         scoresTRUE = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        
#         'Random Chance Performance'
#         y_permuted = np.random.permutation(y)
#         scoresRC = cross_val_score(model, X, y_permuted, scoring='accuracy', cv=cv, n_jobs=-1)
        
#         scoresMODEL.append(scoresTRUE)
#         scoresCHANCE.append(scoresRC)
      
#     return(np.concatenate((scoresMODEL)), np.concatenate((scoresCHANCE)))