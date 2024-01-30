# -*- coding: utf-8 -*-
"""
Purpose: 
    
    [1] To decompose sources of variability over "time"
        Method: factor analysis (varianceSharedPrivate_byTrial)
        "time": FA is performed over sets of 8 trials; 8 is significant because
                it's the number of target locations; 
                each appears randomly within a set of 8 trials.
    [2] To combine behavioral metrics (distance cursor travels, time of trials) from all sessions
    
    
Last update: January 29, 2024
@author: hanna
"""


import os
import pickle
import statistics
import numpy as np
from glob import glob
from scipy import stats
import pandas as pd
from bioinfokit.analys import stat

os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Functions')
from generatePickles_fxns import  getBehavior_TimeDist, zeroUnits, varianceSharedPrivate_byTrial, number_of_factors, varianceSharedPrivate_byTrial, factor_analysis_final_model

dSubjects  = {0:['airp','Airport'], 1:['braz','Brazos']}
targetDeg = np.arange(0,360,45)

subject_ind = 1
subject = dSubjects[subject_ind][0]
subj = dSubjects[subject_ind][1]
root_path = r'C:\BMI_Rotation\Data'+'\\'+subj


'###################################################################'
'''  Extracting Aligned Spike Counts, Trial Times, Distances      '''
'###################################################################'

try:
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    open_file = open(subject+'_Behavior_TimeDist.pkl', "rb") 
    loaded = pickle.load(open_file)
    deg_list, date_list, dSC_, dTime, dDist, dTN, dTargs = loaded
    open_file.close()

except:
    
    dSC_   = {}
    dTime = {}
    dDist = {}
    dTN = {}#trial_num = {}
    dTargs = {}#target_loc = {}
    
    date_list = []
    deg_list = []
    
    
    os.chdir(root_path)
    degFolders = glob('*')
    
    for degFolder in degFolders:
        os.chdir(root_path+'\\'+degFolder)
        dates = glob('*')
        
        
        for date in dates:
            dSC_[date], dTime[date], dDist[date], dTN[date], dTargs[date] = getBehavior_TimeDist(root_path+'\\'+degFolder+'\\'+date)
            deg_list.append(degFolder)
            date_list.append(date)
            
        
            print('Finished:', degFolder, date)

    '''Saving values to .pkl'''     
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    obj_to_pickle = [deg_list, date_list, dSC_, dTime, dDist, dTN, dTargs]
    filename = subject+'_Behavior_TimeDist.pkl'
    open_file = open(filename, "wb")
    pickle.dump(obj_to_pickle, open_file)
    open_file.close()
 

'#############################################################################'
'''  Determining Decoder Units with "0" Spike Count in any Set of Trials    '''
'#############################################################################'

try:
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    filename = subject+'_zeroUNITS.pkl' #'_FA_units_failed.pkl'
    open_file = open(filename, "rb")
    loaded = pickle.load(open_file)
    dfZERO = loaded
    open_file.close()

except:

    datesZERO = {}
    unitsZERO = {}
    
    os.chdir(root_path)
    degFolders = glob('*')

    for degFolder in degFolders:
        os.chdir(root_path+'\\'+degFolder)
        dates = glob('*')
        
        for date in dates:
            datesZERO[date], unitsZERO[date] = zeroUnits(dTN[date], dTargs[date], dSC_[date], date)

    dates = []
    units = []
    
    count=0
    for d in datesZERO.keys():
        if len(unitsZERO[d]) > 0:
            for i in np.unique(unitsZERO[d]):
                dates.append(d)
                units.append(i)

    
    dfZERO = pd.DataFrame({'d':dates, 'unit': units})
    
    '''Saving values to .pkl'''     
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    obj_to_pickle = dfZERO
    filename = subject+'_zeroUNITS.pkl'
    open_file = open(filename, "wb")
    pickle.dump(obj_to_pickle, open_file)
    open_file.close()


'###########################'
'Formatting Imported Data'
'###########################'

dates = date_list

'Time of Trials'
tBL_list = [dTime[d]['BL'] for d in dates]
tPE_list = [dTime[d]['PE'] for d in dates]

'Distance'
TN = [dTN[d] for d in dates]
TL = [dTargs[d] for d in dates]
DX = [dDist[d] for d in dates]


'Spike Counts + Trial Information'
dSC     = [ dSC_[d] for d in dates]
targets = TL



#%%
'#############################################################################'
'''   Determining the Number of Factors CROSS-VALIDATION                '''
'#############################################################################'


try:
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    filename = subject+'numFactors_20240123_10fold_fascores.pkl'
    open_file = open(filename, "rb")
    loaded = pickle.load(open_file)
    numFactorsCV = loaded[2] #dates,degs,numFactorsCV = loaded
    open_file.close()

except:
    
    dates_, numFactorsCV, numUnits_, numBins_ = number_of_factors(dfZERO, dates, tBL_list, tPE_list, TN, TL, DX, dSC, targets, subject, 'cv')

    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    obj_to_pickle = [numUnits_, numBins_, numFactorsCV, dates_]
    filename = subject+'numFactors_20240123_10fold_fascores.pkl'
    open_file = open(filename, "wb")
    pickle.dump(obj_to_pickle, open_file)
    open_file.close()
    
#%%
'#############################################################################'
'''   Determining the Number of Factors 90% OF SHARED VARIANCE              '''
'#############################################################################'


try:
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    filename = subject+'_numFactors_90sharedvariance_20240123.pkl'
    open_file = open(filename, "rb")
    loaded = pickle.load(open_file)
    
    numUnits     = loaded[0]
    numBins      = loaded[1]
    numFactorsSV = loaded[2]
    
    open_file.close()

except:
    
    dates_, numFactorsSV, numUnits, numBins = number_of_factors(dfZERO, dates, tBL_list, tPE_list, TN, TL, DX, dSC, targets, subject, 'sv')

    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    obj_to_pickle = [numUnits, numBins, numFactorsSV, dates_]
    filename = subject+'_numFactors_90sharedvariance_20240123.pkl'
    open_file = open(filename, "wb")
    pickle.dump(obj_to_pickle, open_file)
    open_file.close()
    
#%%


'Percent Difference in Log-Likelihood Scores for Estimated Number of Factors'


# nfSV = {}
# nfCV = {}
# ll   = {}

# change = {}

# for d in range(len(dates)): 

#     nfSV[d] = {}
#     nfCV[d] = {}
#     ll[d]   = {}
    
#     change[d] = {}
  
#     for k in ['BL', 'PE']:
        
#         nfSV[d][k] = []
#         nfCV[d][k] = []
#         ll[d][k]   = []
        
#         change[d][k] = []
        

#         for j in range(40):
            
#             cv_mean = np.mean(np.array(numFactorsCV[d][k])[j,:,:], axis=1)
#             ll[d][k].append(cv_mean)
        
#             sv90 = np.array(numFactorsSV[d][k]['shared90'])[j]
#             sv   = np.array(numFactorsSV[d][k]['shared'])[j,:]
            
#             diff = []
#             for i in range(len(sv)):
#                 diff.append( np.abs(sv[i] - sv90) )
        
#             nf_sv = np.where(np.array(diff) == np.min(diff))[0][0]
#             nf_cv = np.where(cv_mean == np.max(cv_mean))[0][0]
            
#             nfSV[d][k].append(nf_sv)
#             nfCV[d][k].append(nf_cv)
        
#         minNF = np.min(nfSV[d][k])
#         nfSV[d][k] = np.ones(40)*minNF
        
#         for i in range(40):
            
#             scores = np.array(ll[d][k][i])
#             nf1 = int(nfSV[d][k][i])
#             nf2 = int(nfCV[d][k][i])
            
#             percentChange = ((scores[nf1] - scores[nf2])/ scores[nf2])*100
#             change[d][k].append(percentChange)

# vals = []
# for d in range(len(dates)):
#     for k in ['BL', 'PE']:
#         vals.append(change[d][k])

# #import matplotlib.pyplot as plt
# vals = np.concatenate((vals))
# #plt.hist(vals)
# m = np.mean(vals)
# s = np.std(vals)
# print('{} - % Change: {:.4f} ({:.4f})'.format(subject,m,s))
           
            
            
        
        
    
    
    
#%%
'#############################################################################'
' Resolving the Number of Factors based on CROSS-VALIDATION & 90% Shared Variance'
'#############################################################################'


try:
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    filename = subject+'_numFactors_20240124_experimental90Var.pkl'
    open_file = open(filename, "rb")
    loaded = pickle.load(open_file)
    numFactors = loaded[0]

except:
    numFactors = {}
    nfSV = {}
    nfCV = {}
    
    for d in range(len(dates)): 
        numFactors[d] = {}
        nfSV[d] = {}
        nfCV[d] = {}
      
        for k in ['BL', 'PE']:
            
            nfSV[d][k] = []
            nfCV[d][k] = []
            
            nf = []
            for j in range(40):
                
                cv_mean = np.mean(np.array(numFactorsCV[d][k])[j,:,:], axis=1)
            
                sv90 = np.array(numFactorsSV[d][k]['shared90'])[j]
                sv   = np.array(numFactorsSV[d][k]['shared'])[j,:]
                
                diff = []
                for i in range(len(sv)):
                    diff.append( np.abs(sv[i] - sv90) )
            
                nf_sv = np.where(np.array(diff) == np.min(diff))[0][0]
                nf_cv = np.where(cv_mean == np.max(cv_mean))[0][0]
                
                nfSV[d][k].append(nf_sv)
                nfCV[d][k].append(nf_cv)
                
                nf.append(nf_sv)
            
            minNF = np.min(nf)
            #TODO: check minNF >= np.mean(nfCV[d][k])
            
            numFactors[d][k] = np.ones(40)*minNF
            
            
    os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
    obj_to_pickle = [numFactors, nfCV, nfSV]
    filename = subject+'_numFactors_20240124_experimental90Var.pkl'
    pickle.dump(obj_to_pickle, open_file)
    open_file.close()
            










#%%
'#############################################################################'
'            Decoder Population - Factor Analysis '
'#############################################################################'



deg_list, dates_, dEV, dTimes_, dDist, numUnits, numBins, sc_preZ = factor_analysis_final_model(numFactors, dfZERO, dates, tBL_list, tPE_list, TN, TL, DX, dSC, targets, subject)


os.chdir(r'C:\Users\hanna\OneDrive\Documents\BMI_Rotation\Pickles')
obj_to_pickle = [deg_list, dates_, dEV, dTimes_, dDist_, numUnits, numBins, sc_preZ]
filename = subject+'_FA_loadings_40sets_noTC_new.pkl'#filename = subject+'_FA_loadings_40sets_noTC.pkl'
open_file = open(filename, "wb")
pickle.dump(obj_to_pickle, open_file)
open_file.close()