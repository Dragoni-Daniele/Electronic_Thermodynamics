#!/usr/bin/env python
# Daniele Dragoni 07/02/2015
# Fit with Gaussian process

import numpy as np
import scipy
from scipy import optimize
import scipy.interpolate
import matplotlib.pylab as plt
from matplotlib import rc  
import warnings
from datetime import datetime
from tabulate import tabulate

startTime=datetime.now()
###########  INPUT PARAMS LOAD #################################################################################
################################################################################################################
BtoAA    = 0.529177249
eVtoRy   = 0.073498618
RyAtoGPa = 4.5874e-4

print "Use eV, AA for energy and distance"
print

data=np.genfromtxt("/home/dragoni/daint.rsync/data/dragoni/test/EOS_tight/Edef.dat")
x_training=data[:,2]*(BtoAA**3)
t_training=data[:,3]
x_prediction=np.linspace(10.6,12.1,301)   # test set

print "Define correlation length, hight and error of training set"
Theta_train  = 0.0004
Lambda_train = 0.05
Sigma_train  = 0.00008   	# standard deviation, not variance !!!

################################################################################################################
################################################################################################################

############ DEFINE FUNCTIONS ##################################################################################
################################################################################################################
def covariance_DER00(x,y,amplitude,length_scale):      # Covariance function-function
   return amplitude**2*np.exp(-(x-y)**2/(4*length_scale**2))

## Definition of covariance/kernel between first derivative function and function or vice-versa
def covariance_DER10(x,y,amplitude,length_scale):   # Covariance 1st_derivative-function        
   return ((-x+y)/(2*length_scale**2))*amplitude**2*np.exp(-(x-y)**2/(4*length_scale**2)) 
def covariance_DER01(x,y,amplitude,length_scale):   # Covariance function-1st_derivative: equivalent to covariance_DER10
   return ((x-y)/(2*length_scale**2))*amplitude**2*np.exp(-(x-y)**2/(4*length_scale**2)) 

## Definition of covariance/kernel between first derivative function and first derivative function
def covariance_DER11(x,y,amplitude,length_scale):
   return ((2*length_scale**2-(x-y)**2)/(4*length_scale**4))*amplitude**2*np.exp(-(x-y)**2/(4*length_scale**2)) 

## Definition of covariance/kernel between second derivative function
def covariance_DER20(x,y,amplitude,length_scale): 
   return amplitude**2 * (-2*length_scale**2 +(x-y)**2  )/(4*length_scale**4) * np.exp(-(x-y)**2/(4*length_scale**2)) 

## Definition of covariance/kernel between second derivative function and second derivative function
def covariance_sqexp_DER22(x,y,amplitude,length_scale):
      return amplitude**2 * (12*length_scale**4 -12*length_scale**2 *(x-y)**2 + (x-y)**4)/(16*length_scale**8) * np.exp(-(x-y)**2/(4*length_scale**2))
################################################################################################################
################################################################################################################

############ LOAD COVARIANCE MATRICES ##########################################################################
################################################################################################################
Kov_train_train         =np.zeros((x_training.size,x_training.size))
Kov_train_train_noisy   =np.zeros((x_training.size,x_training.size))
Kov_predict_predict     =np.zeros((x_prediction.size,x_prediction.size))
Kov_predict_train       =np.zeros((x_prediction.size,x_training.size))
Kov_train_predict       =np.zeros((x_training.size,x_prediction.size))
Kov_predict_train_DER10 =np.zeros((x_prediction.size,x_training.size))
Kov_predict_train_DER20 =np.zeros((x_prediction.size,x_training.size))
Kov_train_train_DER11   =np.zeros((x_training.size,x_training.size))
Kov_train_train_DER22   =np.zeros((x_training.size,x_training.size))

for line,lvalue in enumerate(x_training):
  for column,cvalue in enumerate(x_training):
   	Kov_train_train[line,column]          = covariance_DER00(lvalue,cvalue,Theta,Lambda)
   	if line==column:
      Kov_train_train_noisy[line,column]  = Kov_train_train[line,column] + (Sigma_train**2)
    else:
      Kov_train_train_noisy[line,column]  = Kov_train_train[line,column]
    
    
for line,lvalue in enumerate(x_prediction):
  for column,cvalue in enumerate(x_prediction):
	  Kov_predict_predict[line,column]      = covariance_DER00(lvalue,cvalue,Theta,Lambda)
    Kov_predict_predict_DER11[line,column]= covariance_DER11(lvalue,cvalue,Theta,Lambda)   #derivata1 incrociata: 
    Kov_predict_predict_DER22[line,column]= covariance_DER22(lvalue,cvalue,Theta,Lambda)   #derivata2 incrociata:

for line,lvalue in enumerate(x_prediction):
  for column,cvalue in enumerate(x_training):
	  Kov_predict_train[line,column]        = covariance_DER00(lvalue,cvalue,Theta,Lambda)
	  Kov_train_predict[line,column]        = Kov_predict_train[column,line]                 #trasposta
	  Kov_predict_train_DER10[line,column]  = covariance_DER10(lvalue,cvalue,Theta,Lambda)   #derivata1 su prediction
	  Kov_predict_train_DER20[line,column]  = covariance_DER20(lvalue,cvalue,Theta,Lambda)   #derivata2 su prediction

################################################################################################################
################################################################################################################


############ CALCULATE #########################################################################################
################################################################################################################
# Prior (unconditioned) distribution in function space 
  prior_mean                     = np.zeros(x_prediction.size)
  prior_covariance               = Kov_predict_predict

# POSTERIOR (conditioned) distribution in function space 
  Inverted_Kov_train_train       = np.linalg.inv(Kov_train_train)
  posterior_mean                 = np.dot(np.dot(Kov_predict_train,Inverted_Kov_train_train),t_training)
  posterior_covariance           = Kov_predict_predict-np.dot(Kov_predict_train,np.dot(Inverted_Kov_train_train,Kov_train_predict))
  posterior_mean_der1            = np.dot(np.dot(Kov_predict_train_DER10,Inverted_Kov_train_train),t_training)
  posterior_covariance_der1      = Kov_predict_predict_DER11-np.dot(Kov_predict_train_DER10,np.dot(Inverted_Kov_train_train,-1 * Kov_predict_train_DER10.T))
  posterior_mean_der2            = np.dot(np.dot(Kov_predict_train_DER20,Inverted_Kov_train_train),t_training)
  posterior_covariance_der2      = Kov_predict_predict_DER22-np.dot(Kov_predict_train_DER20,np.dot(Inverted_Kov_train_train,-1 * Kov_predict_train_DER20.T))

# POSTERIOR (conditioned) NOISY distribution in function space with Gaussian error on training data
  Inverted_Kov_train_train_noisy = np.linalg.inv(Kov_train_train_noisy)
  posterior_mean_noisy           = np.dot(np.dot(Kov_predict_train,Inverted_Kov_train_train_noisy),t_training)
  posterior_covariance_noisy     = Kov_predict_predict-np.dot(Kov_predict_train,np.dot(Inverted_Kov_train_train_noisy,Kov_predict_train))
  posterior_mean_der1_noisy      = np.dot(np.dot(Kov_predict_train_DER10,Inverted_Kov_train_train_noisy),t_training)
  posterior_covariance_der1_noisy= Kov_predict_predict_DER11-np.dot(Kov_predict_train_DER10,np.dot(Inverted_Kov_train_train_noisy,-1 * Kov_predict_train_DER10.T))
  posterior_mean_der2_noisy      = np.dot(np.dot(Kov_predict_train_DER20,Inverted_Kov_train_train_noisy),t_training)
  posterior_covariance_der2_noisy= Kov_predict_predict_DER22-np.dot(Kov_predict_train_DER20,np.dot(Inverted_Kov_train_train_noisy,-1 * Kov_predict_train_DER20.T))

################################################################################################################
################################################################################################################
