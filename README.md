# bmi_rotation


Usage Information

This code is used to analyze brain-machine interface (BMI) data from a visusomotor rotation task.  

For task code, see https://github.com/santacruzlab  (branch: bmi_ssd) bmi_python>built_in_tasks>error_clamp_tasks.py: class BMICursorVisRotErrorClamp.


**main_plots.py**
	This script is designed to visually explore (i.e., plot) the relationship of neural spiking variance (as determined by factor analysis) and behavior (adaptation to a rotation perturbation).

	Pre-requisites:
		Decoder unit factor loadings to compute variance components; Behavior (cursor distance, trial times)
		File Name:   subject+_FA_loadings_40sets.pkl'
		From Script: Neural-And-Behavior-Relationships.py
			Sev = degs, dates, dTC, dEV, dTimes, dDist, subject


**main_stats.py**
	This script is used to compute statistics (F-test, t-test, ANOVA + post-hoc Tukey Kramer)

	Test for Differences in Rotation Condition
		2x two-way ANOVA to test if [1] +/-50 and [2] +/-90 produce significant differences in behavior at each timepoint
	Test for Difference in Number of Decoder Units 
		F-test + t-test 
	Behavioral
		F-test + t-test for adaptation at each timepoint
	Neural
		two-way ANOVA + post-hoc testing for %sv at each timepoint


	Pre-requisites:


**generatePickles.py**

	Prerequisites:
		generatePickles_fxns.py


**stats_fxns.py**

**adaptation_fxns.py** 
	Change fractionOfRecovery to adaptation

	



#Updates
04.20.2023: intial commit