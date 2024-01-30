# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 07:19:58 2023

@author: hanna
"""

import os
import glob
import pickle
import tables
import numpy as np
import pandas as pd
import seaborn as sb
import scipy as sp
from   scipy import stats
from statsmodels.stats.power import TTestIndPower
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from statsmodels.stats.power import TTestPower
#https://www.geeksforgeeks.org/introduction-to-power-analysis-in-python/


from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

from sklearn import svm

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn import metrics
#https://stackoverflow.com/questions/40057049/using-confusion-matrix-as-scoring-metric-in-cross-validation-in-scikit-learn


'IBM Color Scheme'
blue    = [100/255, 143/255, 255/255]
purple  = [120/255, 94/255, 240/255]
magenta = [220/255, 38/255, 127/255]
orange  = [254/255, 97/255, 0/255]
yellow  = [255/255, 176/255, 0/255]


plt.rcParams.update({'font.sans-serif': 'Arial', 'lines.linewidth':1, 'lines.color':'k'})

os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Functions')
#from fractionOfRecovery import getAdaptation, corrRec, fractionRecovery_wTime, behaviorBL
from behavioralMetric_fxns import getAdaptation, behaviorBL
from basicFunctions     import ttest, sigStar
from getDataDictionary_fxn import getDataDictionary


'Adjustable Parameters'
c1 = magenta
c2 = yellow
c3 = blue

dMonkey = {'Subject A': 'Monkey A', 'Subject B': 'Monkey B'}
subject_list = ['Subject A', 'Subject B']
dDegs = {'50': 50, 'neg50': -50, '90': 90, 'neg90': -90}
dCols = {50:orange ,90:purple, -50:orange, -90:purple}

subject_list = ['Subject A', 'Subject B']
dComp, Aev, Bev, Adegs, Bdegs, adt, dfADT, BL_behavior = getDataDictionary(subject_list, True)


#%%
# from statsmodels.stats.power import TTestIndPower
# alpha = 0.05

# timeA = np.array([np.mean(BL_behavior['Subject A'][d]['time']) for d in range(63)])
# distA = np.array([np.mean(BL_behavior['Subject A'][d]['dist']) for d in range(63)])
# timeB = np.array([np.mean(BL_behavior['Subject B'][d]['time']) for d in range(57)])
# distB = np.array([np.mean(BL_behavior['Subject B'][d]['dist']) for d in range(57)])

# ind50A = dComp['Subject A']['b_ind50']
# ind90A = dComp['Subject A']['b_ind90']
# ind50B = dComp['Subject B']['b_ind50']
# ind90B = dComp['Subject B']['b_ind90']

# print('Monkey A')
# F1 = np.var(timeA[ind50A],ddof=1) / np.var(timeA[ind90A],ddof=1)
# t, p = stats.ttest_ind(timeA[ind50A], timeA[ind90A], equal_var=True)
# print('\tMean Baseline Time: {:.3f} ({:.2e}) (F={:.3f})'.format(t,p,F1))


# power = TTestIndPower()
# power1 = power.solve_power(power=None, effect_size=t, alpha=alpha, nobs1=len(ind90A), ratio=len(ind50A)/len(ind90A), alternative='two-sided')

# F2 = np.var(distA[ind90A],ddof=1) / np.var(distA[ind50A],ddof=1)
# t, p = stats.ttest_ind(distA[ind50A], distA[ind90A], equal_var=False)
# print('\tMean Baseline Dist: {:.3f} ({:.2e}) (F={:.3f})'.format(t,p, F2))

# print('Monkey B')
# F3 = np.var(timeB[ind50B],ddof=1) / np.var(timeB[ind90B],ddof=1)
# t, p = stats.ttest_ind(timeA[ind50B], timeA[ind90B], equal_var=False)
# print('\tMean Baseline Time: {:.3f} ({:.2e}) (F={:.3f})'.format(t,p, F3))
# F4 = np.var(distB[ind50B],ddof=1) / np.var(distB[ind90B],ddof=1)
# t, p = stats.ttest_ind(distA[ind50B], distA[ind90B], equal_var=False)
# print('\tMean Baseline Dist: {:.3f} ({:.2e}) (F={:.3f})'.format(t,p, F4))



#%%

for S in subject_list:
    
    ind50 = dComp[S]['b_ind50']
    ind90 = dComp[S]['b_ind90']
    
    a50 = np.array(dComp[S]['bMR'][ind50])
    a90 = np.array(dComp[S]['bMR'][ind90])
    
    g50 = len(np.where(a50 >= 0)[0])
    g90 = len(np.where(a90 >= 0)[0])
    
    print(S)
    print('\t50: {:.1f} / {:.1f}'.format(g50, len(ind50)))
    print('\t50: {:.1f} / {:.1f}'.format(g90, len(ind90)))

#%%

"""
______________________________________________________________________________

STATISTICS
______________________________________________________________________________

"""

def Ftest(v1, v2):
    F = np.var(v1, ddof=1)/np.var(v2, ddof=1)
    n1 = len(v1)
    n2 = len(v2)
    
    p = 1-sp.stats.f.cdf(F, n1, n2)
    
    return(F, p)



'___________________________________________________'
def ttest(v1, v2, equal_var=True):
    
    'https://www.statology.org/two-sample-t-test-python/'
    'https://www.statology.org/welchs-t-test/'
    'https://www.statology.org/p-value-from-t-score-python/'
    
    """
    ASSUMES ALPHA OF 0.05
    This function comfirms that the built-in function scipy.stats.ttest_ind produces the same results as performed by hand.
    When equal_var=True, a Student t-test is run.  When equal_var=False, Welch test is performed.
    Both tests assume that the samples come from normal distributions.
        To test for equality of variances: F-test
    """
    
    n1, n2 = len(v1), len(v2)
    m1, m2 = np.mean(v1), np.mean(v2)
    var1, var2 = np.var(v1, ddof=1), np.var(v2, ddof=1)

    #Student's t-test: assume equal variances (ratio < 4:1)
    student_DoF = n1+n2-2
    student_s_pooled = np.sqrt(((n1-1)*var1 + (n2-1)*var2)/student_DoF)
    Snum = m1-m2
    Sden = student_s_pooled*(np.sqrt( (1/n1) + (1/n2) ))
    student_testStat =  Snum/Sden
    
    #Welch's t-test: assume unequal variances (but still normal distribution)
    welch_DoF_num = ((var1/n1) + (var2/n2))**2
    welch_DoF_den = ((var1/n1)**2/(n1-1)) + ((var2/n2)**2/(n2-1))
    welch_DoF = welch_DoF_num/welch_DoF_den
    Wnum = m1-m2 
    Wden = np.sqrt( (var1/n1)+(var2/n2) )
    welch_testStat = Wnum/Wden
    
    #CRITICAL VALUE
    student_critStat = sp.stats.t.ppf(q=1-(0.05/2),df=student_DoF)
    welch_critStat = sp.stats.t.ppf(q=1-(0.05/2),df=welch_DoF)
    
    print('Student: {:.2f} -  {:.4f}, {:.4f} ({:.4f})'.format(student_DoF, student_testStat, student_critStat, stats.t.sf(np.abs(student_testStat),student_DoF)*2))
    print('Welch:   {:.2f} -  {:.4f}, {:.4f} ({:.4f})'.format(welch_DoF, welch_testStat, welch_critStat, stats.t.sf(np.abs(welch_testStat),welch_DoF)*2))
    print('T: ', stats.ttest_ind(v1,v2,equal_var=True))
    print('F: ', stats.ttest_ind(v1,v2,equal_var=False))
    print('\n')
    

'___________________________________________________'
def effectSize_means(v1, v2, cl=0.95, equal_var=True):
    """
    This function computes the effect size.
    
    For two-sided t-tests: Cohen's d (AKA standardized mean difference (smd))'
        -uses pooled variance
    For two-side welchs test: ???
        -uses "average" variance
    For correlation test: ???
    
    'Confidence Interval Interpretation: There is a 90% (95%) change that the confidence interval of [CI_lower, CI_upper] contains the true mean difference.'
        For normal distribution (z-values):
        '90% confidence interval: z=1.645'
        '95% confidence interval: z=1.960'
        For t-distribution, the corresponding "t" depends on DoF.  As DoF increase to +infinity, the t-value conveges to the z-value [?].

    Delta Degrees of Freedom (ddof):
        'In standard statistical practice, 
            ddof=1 provides an unbiased estimator of the variance of a hypothetical infinite population. 
            ddof=0 provides a maximum likelihood estimate of the variance for normally distributed variables.'

    
    Resources:
        'https://www-sciencedirect-com.ezproxy.lib.utexas.edu/science/article/pii/S0020748912000442'
        'https://machinelearningmastery.com/effect-size-measures-in-python/'
        
    """


    n1, n2 = len(v1), len(v2)
    m1, m2 = np.mean(v1), np.mean(v2)
    var1, var2 = np.var(v1, ddof=1), np.var(v2, ddof=1)


    if equal_var == False:
        'Cohens d for two-sample welchs'
        num = ((var1/n1) + (var2/n2))**2
        den = ((var1/n1)**2/(n1-1)) + ((var2/n2)**2/(n2-1))
        DOF = num/den
        ESsmd = (m1-m2)/np.sqrt(((var1/n1)+(var2/n2)))
        Vsmd = (var1/n1) + (var2/n2)

    else:
        'Cohens d for two-sample t-test'
        DOF = n1 + n2 - 2
        s_pooled = np.sqrt( ( (n1-1)*(var1) + (n2-1)*(var2) )/DOF)
        ESsmd = (m1-m2)/(s_pooled*np.sqrt((1/n1)+ (1/n2)))
        Vsmd = (s_pooled/n1) + (s_pooled/n2)
        
    '% Confidence Interval'
    t = sp.stats.t.ppf(q=1-((1-cl)/2), df=DOF)
    SEsmd = np.sqrt(Vsmd)
    CI_lower = ESsmd - (t*SEsmd)
    CI_upper = ESsmd + (t*SEsmd)
    
    
    return(ESsmd, CI_lower, CI_upper)


'________________________________________'

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
    
    #print('     slope: {:.2f} ({:.4f})({:.2f} {:.2f})'.format(slope, p, LL, UL))

    
    if returnSE == False:
        return(slope, intercept, p, LL, UL)
    else:
        return(slope,se,n1)




#%%

"""

Support for grouping +/-50 and +/-90 (TWO, TWO-WAY ANOVA) 

"""



S = 'Subject B'

'Test if [adatation/behavior] is different for +/-.'

degs4 = np.array(dComp[S]['b_degs'])

indNeg50 = list(np.where(degs4 == -50)[0])
indNeg90 = list(np.where(degs4 == -90)[0])

indPos50 = list(np.where(degs4 ==  50)[0])
indPos90 = list(np.where(degs4 ==  90)[0])


r1 = dComp[S]['bID'][indNeg90]
r2 = dComp[S]['bIR'][indNeg90]
r3 = dComp[S]['bMR'][indNeg90]

r4 = dComp[S]['bID'][indPos90]
r5 = dComp[S]['bIR'][indPos90]
r6 = dComp[S]['bMR'][indPos90]


            
deg_list = [-50]*(len(r1)+len(r2)+len(r3)) + [50]*(len(r4)+len(r5)+len(r6))
tp = ['ID']*len(r1) + ['IR']*len(r2) + ['MR']*len(r3) + ['ID']*len(r4) + ['IR']*len(r5) + ['MR']*len(r6)

df = pd.DataFrame({'sv': np.concatenate((r1, r2, r3, r4, r5, r6)),
                   'deg': deg_list})


model = ols('sv ~ C(deg) + C(tp) + C(deg):C(tp)', data=df).fit()

print(sm.stats.anova_lm(model, typ=2))





#%%

S = 'Subject B' #'Subject A'

ind50 = dComp[S]['b_ind50']
ind90 = dComp[S]['b_ind90']

TSN50 = dComp[S]['maxIND'][ind50]
TSN90 = dComp[S]['maxIND'][ind90] 

v1 = (dComp[S]['bMR'][ind50] - dComp[S]['bID'][ind50])/TSN50
v2 = (dComp[S]['bMR'][ind90] - dComp[S]['bID'][ind90])/TSN90

F1,p = Ftest(v1,v2)
F2,p = Ftest(v2,v1)
print(F1, F2)

print(stats.ttest_ind(v1,v2, equal_var=True, alternative='two-sided'))

#%%

S = 'Subject B' #'Subject A'

ind50 = dComp[S]['b_ind50']
ind90 = dComp[S]['b_ind90']

v1 = dComp[S]['bID'][ind50]
v2 = dComp[S]['bID'][ind90] 

F1,p = Ftest(v1,v2)
F2,p = Ftest(v2,v1)
print(F1, F2)

print(stats.ttest_ind(v1,v2, equal_var=True, alternative='two-sided'))


ES, LL, UL = effectSize_means(v1,v2,equal_var=True)

print(ES, LL, UL)


#%%
"""
___________________________

Number of Decoder Units (average (STD))
___________________________
"""

    # Parameters Saved:
    #          [0]   [1]    [2]  [3]  [4]     [5]    [6]           
    #   Sev =  degs, dates, dTC, dEV, dTimes, dDist, subject

nUnits_A = np.array([np.shape(Aev[2][d]['PE']['loadings'])[-1] for d in range(63)])
nUnits_B = np.array([np.shape(Bev[2][d]['PE']['loadings'])[-1] for d in range(57)])


for S, Sev in zip(subject_list, [Aev, Bev]):
    
    nUnits = np.array([np.shape(Sev[2][d]['PE']['loadings'])[-1] for d in range(len(Sev[0]))])
    
    degs = np.abs(dComp[S]['b_degs'])
    inds = dComp[S]['s_ind']
    
    df = pd.DataFrame({'nU': nUnits[inds], 'degs': degs[inds]})
    
    v1 = df.loc[df['degs'] == 50, 'nU'].values
    v2 = df.loc[df['degs'] == -50, 'nU'].values

    v50 = np.concatenate((v1,v2))

    v1 = df.loc[df['degs'] == 90, 'nU'].values
    v2 = df.loc[df['degs'] == -90, 'nU'].values

    v90 = np.concatenate((v1,v2))
    
    print(S)
    print('\t50 - {:.2f} ({:.2f})'.format(np.mean(v50), np.std(v50)))
    print('\t90 - {:.2f} ({:.2f})'.format(np.mean(v90), np.std(v90)))
    print('\n')

    F1,p1 = Ftest(v50, v90)
    F2,p2 = Ftest(v90, v50)
    
    #print('\t50: {:.4f} ({:.4f}), 90: {:.4f} ({:.4f})'.format(F1, p1, F2, p2))

    if (F1 > 3) or (F2 > 3):
        equal_var=False
    else:
        equal_var=True
    

    t, p = stats.ttest_ind(v50, v90, equal_var=equal_var)
    print('\t t={:.4f} ({:.2e}) (equal_var? {})'.format(t,p, equal_var))
          

   #print('\n{}, {}'.format(np.sum(v50), np.sum(v90)))

#%%

#%%
"""
_____________________

FIGURE 2 - Behavior
_____________________

"""

'Compare rotation conditions (50 v 90) at each timepoint (ID, IA, MA).'

dF = {'Subject A': {}, 'Subject B': {}}
dT = {'Subject A': {}, 'Subject B': {}}
dTmax = {'Subject A': {'two_sided': {}, 'less': {}, 'greater': {}},
         'Subject B': {'two_sided': {}, 'less': {}, 'greater': {}}}

for S in subject_list:
    
    ind50 = dComp[S]['b_ind50']
    ind90 = dComp[S]['b_ind90']
    
    
    for k in ['bID', 'bIR', 'bMR']:
        
        dF[S][k] = {}
        dT[S][k] = {}
        
        v1 = dComp[S][k][ind50]
        v2 = dComp[S][k][ind90]
        
        'F-test to compare variances.' #https://www.geeksforgeeks.org/how-to-perform-an-f-test-in-python/
        F1, p1 = Ftest(v1, v2)
        F2, p2 = Ftest(v2, v1)
        
        dF[S][k]['F1'] = F1
        dF[S][k]['p1'] = p1
        
        dF[S][k]['F2'] = F2
        dF[S][k]['p2'] = p2
        
        'Check to determine to use WELCH for t-test.'
        if  (F1 > 3) or (F2 > 3):
            equal_var = False
           # print('F-test: REJECT NULL - {} - {} - {:.4f} or {:.4f}'.format(S,k,F1,F2))
        else:
            equal_var = True
        
        #print(equal_var)
            
    
        # if (S=='Subject A') and (k=='bIR'):
        #     ttest(v1,v2,equal_var=equal_var)
    
        #print('Equal Var?: ', S, k, equal_var)
        #'Two-Sample, One-Sided t-test'
        #t, p = stats.ttest_ind(v1,v2,equal_var=equal_var, alternative='greater')
        
        'Two-Sample, Two-Sided t-test'
        t, p = stats.ttest_ind(v1,v2,equal_var=equal_var, alternative='two-sided')
        dT[S][k]['t'] = t
        dT[S][k]['p'] = p
        
        '95% CI'
        ES, LL, UL = effectSize_means(v1, v2, cl=0.95, equal_var=equal_var)
        dT[S][k]['ES'] = ES
        dT[S][k]['CI'] = (LL, UL)
        
    
    'Compare each rotation condition to 0 (which is BL).'
    
    'one-sample two-sided t-test'
    dTmax[S]['two-sided'] = {'50':{}, '90':{}}
    
    '...comparing 50 to 0 ....'
    t, p = stats.ttest_1samp(dComp[S]['bMR'][ind50],0, alternative='two-sided')
    dTmax[S]['two-sided']['50']['t'] = t
    dTmax[S]['two-sided']['50']['p'] = p
    
    '...comparing 90 to 0 ....'
    t, p = stats.ttest_1samp(dComp[S]['bMR'][ind90],0, alternative='two-sided')
    dTmax[S]['two-sided']['90']['t'] = t
    dTmax[S]['two-sided']['90']['p'] = p
    
    '_________________________________________________________________________'
    'one-sample one-sided t-tests'
    dTmax[S]['less']    = {'50':{}, '90':{}}
    dTmax[S]['greater'] = {'50':{}, '90':{},'50v90':{}}
    
    '...LESS? comparing 50 to 0 ....'
    t, p = stats.ttest_1samp(dComp[S]['bMR'][ind50],0, alternative='less')
    dTmax[S]['less']['50']['t'] = t
    dTmax[S]['less']['50']['p'] = p
    
    '...GREATER? comparing 50 to 0 ....'
    t, p = stats.ttest_1samp(dComp[S]['bMR'][ind50],0, alternative='greater')
    dTmax[S]['greater']['50']['t'] = t
    dTmax[S]['greater']['50']['p'] = p
    
    '...LESS? comparing 90 to 0 ....'
    t, p = stats.ttest_1samp(dComp[S]['bMR'][ind90],0, alternative='less')
    dTmax[S]['less']['90']['t'] = t
    dTmax[S]['less']['90']['p'] = p
    
    '...GREATER? comparing 90 to 0 ....'
    t, p = stats.ttest_1samp(dComp[S]['bMR'][ind90],0, alternative='greater')
    dTmax[S]['greater']['90']['t'] = t
    dTmax[S]['greater']['90']['p'] = p
    
    '_________________________________________________________________________'
    '...GREATER? 50 v 90 ....'
    t, p = stats.ttest_ind(dComp[S]['bMR'][ind50], dComp[S]['bMR'][ind90], alternative='greater')
    dTmax[S]['greater']['50v90']['t'] = t
    dTmax[S]['greater']['50v90']['p'] = p
    

print('________________\n')
print('F-TEST RESULTS:')
print('________________\n')
for S in subject_list:
    print(S)
    for k in ['bID', 'bIR', 'bMR']:
        if dF[S][k]['F1'] > 1:
            print('\t{} -  F: {:.4f}  ({:.4e})'.format(k, dF[S][k]['F1'], dF[S][k]['p1']) )
        else:
            print('\t\t{} -  F: {:.4f}  ({:.4e})'.format(k, dF[S][k]['F2'], dF[S][k]['p2']) )

print('_______________________________________________________\n')
print('50 v 90 - AT EACH TIMEPOINT - TWO -SIDED (alt="two-sided"):')
print('_______________________________________________________\n')
for S in subject_list:
    print(S)
    for k in ['bID', 'bIR', 'bMR']:
        print('\t'+k)
        print('\t\tt: {:.4f}  ({:.4e})'.format(dT[S][k]['t'], dT[S][k]['p']) )
        #print('\t\tES: {:.4f}  ({:.4f}, {:.4f})'.format(dT[S][k]['ES'], dT[S][k]['CI'][0], dT[S][k]['CI'][1]))

print('____________________\n')
print('50 or 90 v 0(BL):')
print('____________________\n')

for S in subject_list:
    print(S)
    for k in ['two-sided']:#, 'less', 'greater']:
        print('\t'+k)
        print('\t\t50 vs 0: {:.4f} ({:.4e})'.format( dTmax[S][k]['50']['t'], dTmax[S][k]['50']['p']) )
        print('\t\t90 vs 0: {:.4f} ({:.4e})'.format( dTmax[S][k]['90']['t'], dTmax[S][k]['90']['p']) )
    #print('\t\t\t50 v 90: {:.4f} ({:.4e})'.format( dTmax[S]['greater']['50v90']['t'], dTmax[S]['greater']['50v90']['p']) )





#%%


'#####################'
'PRIOR TO REVISIONS'
'#######################'
#%%

# 'Compare change from ID to IR  for 50v90.'

# for S in subject_list:
    
#     ind50 = dComp[S]['b_ind50']
#     ind90 = dComp[S]['b_ind90']
    
#     ID50 = dComp[S]['bID'][ind50]
#     IR50 = dComp[S]['bIR'][ind50]
    
#     diff50 = IR50 - ID50
    
#     ID90 = dComp[S]['bID'][ind90]
#     IR90 = dComp[S]['bIR'][ind90]
    
#     diff90 = IR90 - ID90
    
#     F,p = Ftest(diff50, diff90)

#     print(F,p)
    
#     t,p = stats.ttest_ind(diff90, diff50, equal_var=True, alternative='two-sided')
#     ES, LL, UL = effectSize_means(diff90, diff50)#, cl=0.95, equal_var=True)
    
#     #print(ES, LL, UL)

#     print(t,p)

#%%




# """
# _____________________

# FIGURE 3 
# _____________________

# """

# 'TWO-WAY ANOVA FOR %sv'

# for S in subject_list:

#         inds = dComp[S]['s_ind']
#         degs = dComp[S]['s_degs']
        
#         mBL = dComp[S]['mBL'][inds]
#         sID = dComp[S]['sID'][inds]
#         sIR = dComp[S]['sIR'][inds]
#         sMR = dComp[S]['sMR'][inds]
        
#         deg_list = np.concatenate((degs, degs, degs, degs))
#         tp = ['BL']*len(mBL) + ['ID']*len(sID) + ['IR']*len(sIR) + ['MR']*len(sMR)
        
        
#         deg_tp = [str(i)+'_'+j  for i,j in zip(np.abs(deg_list), tp)]

#         df = pd.DataFrame({'sv': np.concatenate((mBL, sID, sIR, sMR)),
#                             'deg': np.abs(deg_list),
#                             'tp': tp,
#                             'deg_tp':deg_tp})
                           
        
#         model = ols('sv ~ C(deg) + C(tp) + C(deg):C(tp)', data=df).fit() 

      


#         print(S)
#         print(sm.stats.anova_lm(model, typ=2))#['PR(>F)']['C(deg)'])
#         print('\n')
        

# df.to_excel('my_dataframe.xlsx')

# import csv

# with open('my_file2.csv', 'w', newline='') as csvfile:
    
#   for i,j,k in zip(df.sv, df.deg, df.tp):
#       writer = csv.writer(csvfile)
#       writer.writerow([i,j,k])





# 'POST-HOC (TUKEY-KRAMER): TWO-WAY ANOVA (with interactions)'

# """

# Involves comparing cell means, BUT we don't compare every possible pair of cell means...

# CONFOUNDED & UNCONFOUNDED COMPARISONS
#     confounded: cells differ along more than one factor
#     unconfounded: the cells only differ in one factor <- These can be tested.

# not using an adjusted k...

# #https://real-statistics.com/one-way-analysis-of-variance-anova/unplanned-comparisons/tukey-hsd/
# #https://www.graphpad.com/support/faqid/1688/

# """

# from scipy.stats import studentized_range

# for S in subject_list:
#     ind50 = dComp[S]['s_ind50']
#     ind90 = dComp[S]['s_ind90']
    
#     BL50 = dComp[S]['mBL'][ind50]
#     ID50 = dComp[S]['sID'][ind50]
#     IR50 = dComp[S]['sIR'][ind50]
#     MR50 = dComp[S]['sMR'][ind50]
    
#     BL90 = dComp[S]['mBL'][ind90]
#     ID90 = dComp[S]['sID'][ind90]
#     IR90 = dComp[S]['sIR'][ind90]
#     MR90 = dComp[S]['sMR'][ind90]
    
#     alpha = 0.05
#     N = len(ind50)*4 + len(ind90)*4  #total number of observations
#     n50 = len(ind50) #number of observations (per comparison) - group 1 
#     n90 = len(ind90) #number of observations (per comparison) - group 2
#     k = 8 #number of cell means (unadjusted)
#     dfw = N - k

    
#     _, qcrit = studentized_range.ppf([alpha/2,1-alpha/2], k=k, df=dfw) #two-sided alternative
    
#     observations = [BL50, ID50, IR50, MR50, BL90, ID90, IR90, MR90]
    
#     groupMeans = [np.mean(BL50), np.mean(ID50), np.mean(IR50), np.mean(MR50),
#                   np.mean(BL90), np.mean(ID90), np.mean(IR90), np.mean(MR90)]
    
#     SSW_ = []
    
#     for j in range(8):
#             groupMean = groupMeans[j]
#             obs = observations[j]
#             SSW_.append(np.sum([ (i-groupMean)**2  for i in obs]))
           
    
#     SSW = np.sum(SSW_)
#     MSW = SSW/dfw
    
#     """
#     SE for 50-50
#     SE for 90-90
#     SE for 50-90
#     """
    
#     se1 = np.sqrt(2) * np.sqrt(MSW/n50)
#     se2 = np.sqrt(2) * np.sqrt(MSW/n90)
#     se3 = np.sqrt(2) * np.sqrt(MSW/((n50+n90)/2))
    
#     #print('s50: {:.5f}\ns90: {:.5f}\ns50-90: {:.5f}'.format(se1, se2, se3))
    
    
    
#     """
#     _____________
    
#     ACROSS ROTATION CONDITIONS
#     ____________
    
#     """
    
#     k1 = ['BL50', 'ID50', 'IA50', 'MA50']
    
#     k2 = ['BL90', 'ID90', 'IA90', 'MA90']
    
#     group1 = [BL50, ID50, IR50, MR50]
#     group2 = [BL90, ID90, IR90, MR90]
    
#     print(S)
#     print('\tACROSS ROTATION CONDITIONS')
    
#     for v1, v2, k1, k2 in zip(group1, group2, k1, k2):
        
#         m1 = np.mean(v1)
#         m2 = np.mean(v2)
        
#         qtest = np.abs(m1-m2)/se3
    
        
    
#         p = studentized_range.sf(x=qtest*np.sqrt(2), k=k, df=dfw) 
        
#         #tcrit = qcrit/np.sqrt(2)
        
#         print('\t\t{} - {}: qcrit = {:.4f}, qtest = {:.4f} (P={:.4e})'.format(k1, k2,qcrit, qtest, p))
        
    
#     """
#     _____________
    
#     WITHIN 50
#     ____________
    
#     """
    
#     # k1 = ['BL50', 'BL50', 'BL50', 'ID50', 'ID50', 'IA50']
#     # k2 = ['ID50', 'IA50', 'MA50', 'IA50', 'MA50', 'MA50']
    
#     # group1 = [BL50, BL50, BL50, ID50, ID50, IR50]
#     # group2 = [ID50, IR50, MR50, IR50, MR50, MR50]
    
    
#     k1 = ['BL50']
#     k2 = ['MA50']
    
#     group1 = [BL50]
#     group2 = [MR50]
    
#     print('\tWITHIN 50')
    
#     for v1, v2, k1, k2 in zip(group1, group2, k1, k2):
        
#         m1 = np.mean(v1)
#         m2 = np.mean(v2)
        
#         qtest = np.abs(m1-m2)/se1
    
#         p = studentized_range.sf(x=qtest*np.sqrt(2), k=k, df=dfw) #sf = 1 - CDF
        
#         #tcrit = qcrit/np.sqrt(2)
        
#         print('\t\t{} - {}: qcrit = {:.4f}, qtest = {:.4f} (P={:.4e})'.format(k1, k2,qcrit, qtest, p))
    
    
#     """
#     _____________
    
#     WITHIN 90
#     ____________
    
#     """
    
    
#     # k1 = ['BL90', 'BL90', 'BL90', 'ID90', 'ID90', 'IA90']
#     # k2 = ['ID90', 'IA90', 'MA90', 'IA90', 'MA90', 'MA90']
    
#     # group1 = [BL90, BL90, BL90, ID90, ID90, IR90]
#     # group2 = [ID90, IR90, MR90, IR90, MR90, MR90]
#     print('\tWITHIN 90')
    
#     k1 = ['BL90']
#     k2 = ['MA90']
    
#     group1 = [BL90]
#     group2 = [MR90]
    
#     for v1, v2, k1, k2 in zip(group1, group2, k1, k2):
        
#         m1 = np.mean(v1)
#         m2 = np.mean(v2)
        
#         qtest = np.abs(m1-m2)/se2
    
#         p = studentized_range.sf(x=qtest*np.sqrt(2), k=k, df=dfw)/2 #sf = 1 - CDF
        
#         #tcrit = qcrit/np.sqrt(2)
        
#         print('\t\t{} - {}: qcrit = {:.4f}, qtest = {:.4f} (P={:.4e})'.format(k1, k2,qcrit, qtest, p))

# #%%

# fig, ax = plt.subplots(1,1)

# x = np.linspace(studentized_range.ppf(0.01, k, dfw),
#                 studentized_range.ppf(0.99, k, dfw), 100)
# ax.plot(x, studentized_range.pdf(x, k, dfw),
#         'r-', lw=5, alpha=0.6, label='studentized_range pdf')


# _, temp = studentized_range.ppf([0,0.95], k, dfw)

# plt.axvline(temp)

# plt.scatter(temp, studentized_range.cdf(temp,k,dfw))
# plt.scatter(temp, studentized_range.sf(temp,k,dfw))

# plt.axhline(0.05, color='grey', ls='--')
# plt.axhline(0.95, color='grey', ls='--')

# _, qcrit = studentized_range.ppf([0,1-(alpha/2)], k=k, df=dfw) #alpha/2????

# print(studentized_range.pdf(1.865,k,dfw))
# print(studentized_range.pdf(qcrit,k,dfw))







