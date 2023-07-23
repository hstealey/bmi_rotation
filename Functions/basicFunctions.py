# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:04:00 2023

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


from statsmodels.stats.power import TTestPower
#https://www.geeksforgeeks.org/introduction-to-power-analysis-in-python/



'Projection'
'https://flexbooks.ck12.org/cbook/ck-12-college-precalculus/section/9.6/primary/lesson/scalar-and-vector-projections-c-precalc/'

"""
stat, star                = sigStar(v1, v2, test)
                            Ftest(v1, v2)
                            ttest(v1, v2)
ESsmd, CI_lower, CI_upper = effectSize_means(v1, v2, cl=0.95, welchs=True)
                            testPower(v1, v2, cl=0.95, welchs=True, eS=None)
                            plotLR(X,y,ax, c)
"""

'___________________________________________________'
def sigStar(v1, v2, test):
    
    if test == 'corr':
        r, p = stats.pearsonr(v1, v2)
    elif test == 'ttest':
        r, p = stats.ttest_ind(v1, v2, equal_var=False)
        #Assumes variances are NOT equal  (failed F-test).
    
    if p < 0.001:
        star = '***'
    elif p < 0.01:
        star = '**'
    elif p <= 0.05:
        star = '*'
    elif (p >0.05) and (p<=0.1):
        star = 't.'
    else:
        star = 'n.s. (>0.1)'

    return(r, star)

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
def ttest(v1, v2):
    
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
def effectSize_means(v1, v2, cl=0.95, welchs=True):
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


    if welchs == True:
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
    
    'Correction for Small Sample Sizes'
    #J = 1 - (3/(4*DOF-1))
    
    'Hedges g (unbiased estimate)'
    # ESgsmd = J*ESsmd
    # Vgsmd  = (J**2)*Vsmd
    # SEgsmd = np.sqrt(Vgsmd)
    # gCI_lower = ESsmd - (ci*SEgsmd)
    # gCI_upper = ESsmd + (ci*SEgsmd)
    
    # print('Welchs: {}'.format(welchs))
    # print('Effect Size (CI):  {:.4f} ({:.4f}, {:.4f})'.format(ESsmd, CI_lower, CI_upper))
    
    return(ESsmd, CI_lower, CI_upper)


'________________________________________'

def testPower(v1, v2, cl=0.95, welchs=True, eS=None):
    'https://machinelearningmastery.com/statistical-power-and-power-analysis-in-python/'
    'https://www.statsmodels.org/dev/_modules/statsmodels/stats/power.html#TTestPower'
    'effect size must be positive'
    
    
    print('Welchs: {}'.format(welchs))
    if eS == None:
        eS, LL, UL  = effectSize_means(v1, v2, cl, welchs)
        print('Effect Size (CI):  {:.4f} ({:.4f}, {:.4f})'.format(eS, LL, UL))
    
    
    n1, n2 = len(v1), len(v2)
    if welchs == True:
        var1, var2 = np.var(v1, ddof=1), np.var(v2, ddof=1)
        num = ((var1/n1) + (var2/n2))**2
        den = ((var1/n1)**2/(n1-1)) + ((var2/n2)**2/(n2-1))
        DOF = num/den
    else:
        DOF    = n1+n2-2
    result = TTestIndPower().power(effect_size = eS, nobs1=n1, alpha=0.05, ratio=n2/n1, alternative='two-sided')
    print('Power:  {:.4f}'.format(result))

    
    #return(result)



# alpha = 0.05

# sample1 = [14, 15, 15, 15, 16, 18, 22, 23, 24, 25, 25]
# sample2 = [10, 12, 14, 15, 18, 22, 24, 27, 31, 33, 34, 34, 34]

# v1 = sample1
# v2 = sample2

# n1, n2 = len(v1), len(v2)


# eS, LL, UL = effectSize_means(v1,v2, 0.95, welchs=False)

# nobs = 1./ (1. / n1 + 1. / n2)
# df = n1+n2-2

# alpha_ = 0.05/2

# crit_upp = stats.t.isf(alpha_, df)
# crit_low = stats.t.ppf(alpha_, df)

# from scipy import special
# def nct_sf(x, df, nc):
#     return 1 - special.nctdtr(df, nc, x)
# power1_ = stats.nct._sf(crit_upp, df, eS*np.sqrt(nobs))
# power1 = nct_sf(crit_upp, df, eS*np.sqrt(nobs)) 

# def nct_cdf(x, df, nc):
#     return special.nctdtr(df, nc, x)
# power2_ = stats.nct._cdf(crit_low, df, eS*np.sqrt(nobs))
# power2  = nct_cdf(crit_low, df, eS*np.sqrt(nobs))

# print(power1_+power2_)
# print(power1+power2)


# power = TTestIndPower().solve_power(power=None, effect_size = eS, nobs1=n1, alpha=0.05, ratio=n2/n1, alternative='two-sided')
# print(power)


