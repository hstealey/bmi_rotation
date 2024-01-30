# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:45:20 2024

@author: hanna
"""

import os
import glob
import pickle
import numpy as np
import pandas as pd
from scipy import stats


os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Functions')

from behavioralMetric_fxns import getAdaptation, behaviorBL


#%%
def var_check(x1, x2):
    
    v1 = np.var(x1, ddof=1)
    v2 = np.var(x2, ddof=1)
    
    ratio1 = v1/v2
    ratio2 = v2/v2
    
    if (ratio1 > 3) or (ratio2 > 3):
        equal_var = False
    else:
        equal_var = True
    
    return(equal_var)


def ttest(x1, x2, returnDOF=False):
    
    equal_var = var_check(x1, x2)
    
    dof = len(x1) + len(x2) - 2
    
    t,p = stats.ttest_ind(x1, x2, equal_var=equal_var)
    
    if returnDOF == False:
        return(equal_var, t, p)
    else:
        return(equal_var, t, p, dof)


#%%


def getDataDictionary(subject_list, returnEV=False):

    #subject_list = ['Subject A', 'Subject B']
    dDegs = {'50': 50, 'neg50': -50, '90': 90, 'neg90': -90}
    
    
    
    """
    _______________________________________________________________
    _______________________________________________________________  
    
        Number of Factors used to Fit Model
        
    ___________________________________________________  
    
    File Name: subject+'numFactors_20231225.pkl'
    From Script: Functions/generatePickles_fxns_ALT_kfold-90.py *is actually LOO
    
    
    """
    
    
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    #fn = glob.glob('*numFactors_20231225.pkl')
    fn  = glob.glob('*_numFactors_20240124_experimental90Var.pkl')
    open_file = open(fn[0], "rb")
    loaded = pickle.load(open_file)
    AnumFactors = loaded[0]
    open_file.close()
    
    open_file = open(fn[1], "rb")
    loaded = pickle.load(open_file)
    BnumFactors = loaded[0]
    open_file.close()
        
    
    
    
    """
    _______________________________________________________________
    _______________________________________________________________
    
        Decoder Unit Tuning Properties (over sets of 8 trials ("time"))
        ___________________________________________________    
        
        File Name:    subject+'_FA_loadings_40sets_alt_LOOCV-90_intMean_withNumBins_withSCpreZ_20240116.pkl' 
        From Script:  Functions/generatePickles_fxns_ALT_kfold-90.py *is actually LOO
        
        Parameters Saved:
                             [0]    [1]   [2]   [3]     [4]     [5]     [6]       
         previously: Sev =  degs, dates, dEV, dTimes, dDist, numUnits, CV results
          
                 [0]       [1]    [2]  [3]      [4]       [5]     [6]      [7]
          Sev = deg_list, dates, dEV, dTimes_, dDist_, numUnits, numBins, sc_preZ
          
          
          NUMBER OF UNITS (numUnits) IS WRONG????
          DOUBLE CHECK NUM BINS??????
    
    ________________________________________________________________
    """
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    #fn = glob.glob('*_FA_loadings_40sets_noTC.pkl')
    #fn = glob.glob('*_FA_loadings_40sets_noTC_noCV_results_10factors.pkl')
    #fn = glob.glob('*_FA_loadings_40sets_alt_LOOCV-90_intMean.pkl')
    fn = glob.glob('*_FA_loadings_40sets_alt_LOOCV-90_experimental90Var_20240124.pkl') #fn  = glob.glob('*_FA_loadings_40sets_alt_LOOCV-90_intMean_withNumBins_withSCpreZ_20240116.pkl')
    
    
    open_file = open(fn[0], "rb")
    Aev = pickle.load(open_file)
    open_file.close()
    
    open_file = open(fn[1], "rb")
    Bev = pickle.load(open_file)
    open_file.close()
    
    '___________________________________________________'
    
    
    
    Adegs = [dDegs[d] for d in Aev[0]]
    Bdegs = [dDegs[d] for d in Bev[0]]
    
    
    
    """
    ______________________________________________________________________________
    
    Fraction of ADAPTATION & Single Unit Communality Over Time (sets)
    ______________________________________________________________________________
    """
    
    'Behavior - BASELINE'
    BL_behavior = {}
    for Sev, subject in zip([Aev, Bev], ['Subject A', 'Subject B']):
        BL_behavior[subject] = behaviorBL(Sev)
    
    'Behavior - ADAPTATION'
    adt   = {}
    dfADT = {}
    
    for Sev, subject in zip([Aev, Bev], ['Subject A', 'Subject B']):
        adt[subject], dfADT[subject] = getAdaptation(Sev)
    
    
    
    
    
    """
    ______________________________________________________________________________
    
    DATA DICTIONARIES - Behavioral Aaptation & Population Shared Variance 
    ______________________________________________________________________________
    
    
    """
    
    dS = {}
    
    for S, Sev, degs, SnumFactors  in zip(subject_list, [Aev, Bev], [Adegs, Bdegs], [AnumFactors, BnumFactors]):
    
        dS[S] = {'degs':degs, 'numUnits':[], 'numBinsBL':[], 'numBinsPE':[],
                 'sharedBL':[], 'privateBL':[], 'sharedPE':[], 'privatePE':[], 'totalBL':[], 'totalPE':[],
                 'sBL':[], 'sPE':[], 'pBL':[], 'pPE':[],
                 'scBL':[], 'scPE':[],
                 'numFactorsBL': [], 'numFactorsPE':[]}
    
    
        for d in range(len(Sev[1])):
            
            # numBinsBL_ = []
            # numBinsPE_ = []
            # # for i in range(40):
            # #     numBinsBL_.append(Sev[7][d]['BL'][i].shape[0])
            # #     numBinsPE_.append(Sev[7][d]['PE'][i].shape[0])
                
            # dS[S]['numBinsBL'].append(numBinsBL_)
            # dS[S]['numBinsPE'].append(numBinsPE_)
            
            dS[S]['numBinsBL'].append(Sev[6][d]['BL'])
            dS[S]['numBinsPE'].append(Sev[6][d]['PE'])
            
            #dS[S]['numUnits'].append(Sev[7][d]['BL'][0].shape[1])
            numUnits = Sev[5][d]['BL'][0]
            dS[S]['numUnits'].append(numUnits)
            
            sBL = np.array(Sev[2][d]['BL']['shared'])
            sPE = np.array(Sev[2][d]['PE']['shared'])
            
            pBL = np.array(Sev[2][d]['BL']['private'])
            pPE = np.array(Sev[2][d]['PE']['private'])
            
            tBL = np.array(Sev[2][d]['BL']['total'])
            tPE = np.array(Sev[2][d]['PE']['total'])
            
            
            msBL_ratio = np.mean(sBL/tBL)
            mpBL_ratio = np.mean(pBL/tBL)
            
            sPE_ratio = sPE/tPE
            pPE_ratio = pPE/tPE
            
            dS[S]['sharedBL'].append(sBL)
            dS[S]['privateBL'].append(pBL)
            dS[S]['sharedPE'].append(sPE)
            dS[S]['privatePE'].append(pPE)
            dS[S]['totalBL'].append(tBL)
            dS[S]['totalPE'].append(tPE)
            
            dS[S]['sBL'].append(msBL_ratio)
            dS[S]['pBL'].append(mpBL_ratio)
            dS[S]['sPE'].append(100 * ( (sPE_ratio-msBL_ratio)/msBL_ratio ))
            dS[S]['pPE'].append(100 * ( (pPE_ratio-mpBL_ratio)/mpBL_ratio ))
            
            dS[S]['scBL'].append(Sev[7][d]['BL'])
            dS[S]['scPE'].append(Sev[7][d]['PE'])
            
            dS[S]['numFactorsBL'].append(np.array(SnumFactors[d]['BL']))
            dS[S]['numFactorsPE'].append(np.array(SnumFactors[d]['PE']))
            
    
    
    dComp = {'Subject A':{}, 'Subject B':{}}
    
    for S, degs in zip(subject_list, [Adegs, Bdegs]):
    
        ind = np.arange(len(degs))
        
        dfIndDeg = pd.DataFrame({'ind':ind,'deg': np.abs(degs) })
        
        dComp[S]['b_degs']  = degs
        dComp[S]['b_ind']   = ind
        dComp[S]['b_ind50'] = dfIndDeg.loc[dfIndDeg['deg']==50, 'ind'].values.tolist()
        dComp[S]['b_ind90'] = dfIndDeg.loc[dfIndDeg['deg']==90, 'ind'].values.tolist()
        
        if S == 'Subject B':
            ind_ = ind
            degs_ = degs
            
            ind = np.delete(ind_, [38,41,48])
            degs = np.delete(degs_, [38,41,48])
            
        dfIndDeg = pd.DataFrame({'ind':ind,'deg': np.abs(degs) })
        
        dComp[S]['s_ind']   = ind
        dComp[S]['s_degs']  = degs
        dComp[S]['s_ind50'] = dfIndDeg.loc[dfIndDeg['deg']==50, 'ind'].values
        dComp[S]['s_ind90'] = dfIndDeg.loc[dfIndDeg['deg']==90, 'ind'].values
    
        'Behavior'
        maxIND = np.array(dfADT[S]['indMR'].values.tolist())
        ID     = np.array(dfADT[S]['ID'].values)
        IR     = np.array(dfADT[S]['IR'].values)
        MR     = np.array(dfADT[S]['MR'].values)
        n      = len(maxIND)
        adtS   = np.array([adt[S][i] for i in range(n)]).reshape((n,40))
         
        dComp[S]['maxIND']  = maxIND
        dComp[S]['bID']     = ID
        dComp[S]['bIR']     = IR
        dComp[S]['bMR']     = MR
        dComp[S]['rec']     = adtS
        
        '%change in %sv (Shared Variance)'
        sBL  = np.array(dS[S]['sBL'])
        sPE  = np.array(dS[S]['sPE'])
        
        #mBL = np.array([np.nanmean([sBL[i][j] if sBL[i][j] <0.9 else float('nan') for j in range(40)]) for i in range(len(sBL))])
        #dComp[S]['mBL'] = mBL
        dComp[S]['mBL'] = sBL
        dComp[S]['sPE'] = sPE
        
        dComp[S]['sID'] = sPE[:,0]
        dComp[S]['sIR'] = sPE[:,1]
        dComp[S]['sMR'] = np.array([sPE[i][maxIND[i]] for i in range(len(maxIND))])    
        
        
        'Shared, Private, Total Variance'
        dComp[S]['totalBL'] = np.array(dS[S]['totalBL'])
        dComp[S]['totalPE'] = np.array(dS[S]['totalPE'])
        
        dComp[S]['sharedBL'] = np.array(dS[S]['sharedBL'])
        dComp[S]['sharedPE'] = np.array(dS[S]['sharedPE'])
        
        dComp[S]['privateBL'] = np.array(dS[S]['privateBL'])
        dComp[S]['privatePE'] = np.array(dS[S]['privatePE'])
    
        'Number of Units'
        dComp[S]['numUnits'] = np.array(dS[S]['numUnits'])
        
        'Number of Bins'
        dComp[S]['numBinsBL'] = np.array(dS[S]['numBinsBL'])
        dComp[S]['numBinsPE'] = np.array(dS[S]['numBinsPE'])
        
        'Spike Counts (pre z-scored)'
        dComp[S]['scBL'] = np.array(dS[S]['scBL'])
        dComp[S]['scPE'] = np.array(dS[S]['scPE'])
        
        'Number of Factors'
        dComp[S]['numFactorsBL'] = np.array(dS[S]['numFactorsBL'])
        dComp[S]['numFactorsPE'] = np.array(dS[S]['numFactorsPE'])
    
    if returnEV == True:
        return(dComp, Aev, Bev, Adegs, Bdegs, adt, dfADT, BL_behavior)
    else:
        return(dComp)

# #%%

# subject_list = ['Subject A', 'Subject B']
# dComp = getDataDictionary(subject_list)

# S = 'Subject A'
# print(np.shape(dComp[S]['numFactorsBL']))




