#!/usr/bin/env python
# Daniele Dragoni 07/02/2015
# Fit with Gaussian process - FUNCTION
###########  INPUT PARAMS LOAD #################################################################################
################################################################################################################
BtoAA    = 0.529177249
eVtoRy   = 0.073498618
RyAtoGPa = 4.5874e-4

import numpy as np
#data=np.genfromtxt("/home/dragoni/daint.rsync/data/dragoni/test/EOS_tight/Edef.dat")
data=np.genfromtxt("/home/dragoni/bellatrix.rsync/data/dragoni/elastic-DAN-FM/isotropic/REdo/Edef.dat")

x_training=data[:,2]*(BtoAA**3)
t_training=data[:,3]
x_prediction=np.linspace(10.6,12.1,101)   # test set

print "Define correlation length, hight and error of training set"
Theta  = 0.0002		# Amplitude of oscillations
Lambda = 0.08		# Correlation length
Sigma  = 0.00008   	# standard deviation, not variance !!!
################################################################################################################
################################################################################################################


####### MAIN FUNCTION ##########################################################################################
################################################################################################################
def  GP_fit(x_prediction,x_training,t_training,Theta,Lambda,Sigma):
   import numpy as np
   import scipy
   import sys
   from scipy import optimize
   import scipy.interpolate
   import matplotlib.pylab as plt
   from matplotlib import rc  
   import warnings
   from datetime import datetime
   from tabulate import tabulate
   
   startTime=datetime.now()
   #print "Use eV, AA for energy and distance !!"
   #print
   ############ DEFINE SUB-FUNCTIONS ##############################################################################
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
   def covariance_DER22(x,y,amplitude,length_scale):
     return amplitude**2 * (12*length_scale**4 -12*length_scale**2 *(x-y)**2 + (x-y)**4)/(16*length_scale**8) * np.exp(-(x-y)**2/(4*length_scale**2))
   ################################################################################################################
   ################################################################################################################
   ############ FIT quadratic polynomial ##########################################################################
   ################################################################################################################
   degpoly        = 2                   # QUadratic fit to start with
   pol            = np.polyfit(x_training,t_training,degpoly)
   polder1        = np.polyder(pol,1) 
   polder2        = np.polyder(pol,2) 
   poly_func      = np.poly1d(pol)
   poly_func_der1 = np.poly1d(polder1)
   poly_func_der2 = np.poly1d(polder2)
   t_training     = t_training-poly_func(x_training)
   ################################################################################################################
   ################################################################################################################
   ############ LOAD COVARIANCE MATRICES ##########################################################################
   ################################################################################################################
   Kov_train_train           =np.zeros((x_training.size,x_training.size))
   Kov_train_train_noisy     =np.zeros((x_training.size,x_training.size))
   Kov_predict_predict       =np.zeros((x_prediction.size,x_prediction.size))
   Kov_predict_train         =np.zeros((x_prediction.size,x_training.size))
   Kov_train_predict         =np.zeros((x_training.size,x_prediction.size))
   Kov_predict_train_DER10   =np.zeros((x_prediction.size,x_training.size))
   Kov_predict_train_DER20   =np.zeros((x_prediction.size,x_training.size))
   Kov_predict_predict_DER11 =np.zeros((x_prediction.size,x_prediction.size))
   Kov_predict_predict_DER22 =np.zeros((x_prediction.size,x_prediction.size))

   for line,lvalue in enumerate(x_training):
     for column,cvalue in enumerate(x_training):
       Kov_train_train[line,column]          = covariance_DER00(lvalue,cvalue,Theta,Lambda)
       if line==column:
         Kov_train_train_noisy[line,column]  = Kov_train_train[line,column] + (Sigma**2)
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
       #Kov_train_predict[line,column]        = Kov_predict_train[column,line]                 #trasposta
       Kov_predict_train_DER10[line,column]  = covariance_DER10(lvalue,cvalue,Theta,Lambda)   #derivata1 su prediction
       Kov_predict_train_DER20[line,column]  = covariance_DER20(lvalue,cvalue,Theta,Lambda)   #derivata2 su prediction
   Kov_train_predict=Kov_predict_train.T
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
   posterior_covariance_noisy     = Kov_predict_predict-np.dot(Kov_predict_train,np.dot(Inverted_Kov_train_train_noisy,Kov_train_predict))
   posterior_mean_der1_noisy      = np.dot(np.dot(Kov_predict_train_DER10,Inverted_Kov_train_train_noisy),t_training)
   posterior_covariance_der1_noisy= Kov_predict_predict_DER11-np.dot(Kov_predict_train_DER10,np.dot(Inverted_Kov_train_train_noisy,-1 * Kov_predict_train_DER10.T))
   posterior_mean_der2_noisy      = np.dot(np.dot(Kov_predict_train_DER20,Inverted_Kov_train_train_noisy),t_training)
   posterior_covariance_der2_noisy= Kov_predict_predict_DER22-np.dot(Kov_predict_train_DER20,np.dot(Inverted_Kov_train_train_noisy,-1 * Kov_predict_train_DER20.T))
   # ERRORS on predictions
   check_negative=np.where(np.diag(posterior_covariance)<0.)
   if check_negative[0] is not None:
     warnings.warn("Negative variance elements ... std ?!!!")
     #sys.exit("Negative variance elements -- std not possibile")
   devstd            =np.sqrt(np.diag(posterior_covariance))
   devstd_noisy      =np.sqrt(np.diag(posterior_covariance_noisy))
   devstd_der1       =np.sqrt(np.diag(posterior_covariance_der1))
   devstd_der1_noisy =np.sqrt(np.diag(posterior_covariance_der1_noisy))
   devstd_der2       =np.sqrt(np.diag(posterior_covariance_der2))
   devstd_der2_noisy =np.sqrt(np.diag(posterior_covariance_der2_noisy))
   ################################################################################################################
   ################################################################################################################
   ############# RECONSTRUCT ORIGINAL FUNCTION ####################################################################
   ################################################################################################################
   # Reconstruct the optimized function
   posterior_mean            = posterior_mean+poly_func(x_prediction)
   posterior_mean_der1       = posterior_mean_der1+poly_func_der1(x_prediction)
   posterior_mean_der2       = posterior_mean_der2+poly_func_der2(x_prediction)
   posterior_mean_noisy      = posterior_mean_noisy+poly_func(x_prediction)
   posterior_mean_der1_noisy = posterior_mean_der1_noisy+poly_func_der1(x_prediction)
   posterior_mean_der2_noisy = posterior_mean_der2_noisy+poly_func_der2(x_prediction)
   

   return posterior_mean,posterior_mean_der1,posterior_mean_der2, posterior_mean_noisy,posterior_mean_der1_noisy,posterior_mean_der2_noisy
   print 'Execution time: ',(datetime.now()-startTime)

################################################################################################################
################################################################################################################




########### 1-D OUTPUT #########################################################################################
################################################################################################################
def  GP_fit_func(x_prediction,x_training,t_training,Theta,Lambda,Sigma):
     return GP_fit(x_prediction,x_training,t_training,Theta,Lambda,Sigma)[0]

def  GP_fit_der1(x_prediction,x_training,t_training,Theta,Lambda,Sigma):
     return GP_fit(x_prediction,x_training,t_training,Theta,Lambda,Sigma)[1]

def  GP_fit_der2(x_prediction,x_training,t_training,Theta,Lambda,Sigma):
     return GP_fit(x_prediction,x_training,t_training,Theta,Lambda,Sigma)[2]

def  GP_fit_func_noisy(x_prediction,x_training,t_training,Theta,Lambda,Sigma):
     return GP_fit(x_prediction,x_training,t_training,Theta,Lambda,Sigma)[3]

def  GP_fit_der1_noisy(x_prediction,x_training,t_training,Theta,Lambda,Sigma):
     return GP_fit(x_prediction,x_training,t_training,Theta,Lambda,Sigma)[4]

def  GP_fit_der2_noisy(x_prediction,x_training,t_training,Theta,Lambda,Sigma):
     return GP_fit(x_prediction,x_training,t_training,Theta,Lambda,Sigma)[5]

################################################################################################################
################################################################################################################



######## CHECK CONVERGENCE HYPERPARAMETERS WHEN POSSIBLE #######################################################
################################################################################################################

def check_hyperparams_convergence(x_training,t_training,Theta,Lambda,Sigma):

  def covariance_DER00(x,y,amplitude,length_scale):      # Covariance function-function
    return amplitude**2*np.exp(-(x-y)**2/(4*length_scale**2))

  def Kov_train_train_noisy_func(x_training,t_training,Theta,Lambda,Sigma):
    Kov_train_train_noisy = []                                                                 
    for line in x_training:                                                                       
      for column in x_training:    
        if line==column:                                       
          Kov_train_train_noisy.append(covariance_DER00(line,column,Theta,Lambda)+(Sigma**2))
        else:
	  Kov_train_train_noisy.append(covariance_DER00(line,column,Theta,Lambda))
    return np.array(Kov_train_train_noisy).reshape(len(x_training),len(x_training))

  def logP(x_training,t_training,Theta,Lambda,Sigma):
    K=Kov_train_train_noisy_func(x_training,t_training,Theta,Lambda,Sigma)
    return -0.5*np.log(np.linalg.det( K )) - 0.5*np.dot(np.dot(t_training,np.linalg.inv(K)),t_training  )-len(x_training)/2.*np.log(np.pi*2)

  Theta_values  = np.arange(0.05,0.2,0.01)
  Lambda_values = np.arange(0.01,0.16,0.004)
  Parameters_function=[]

  for Theta_ in Theta_values:
    for Lambda_ in Lambda_values:
      Parameters_function.append(logP(x_training,t_training,Theta_,Lambda_,Sigma))

  Parameters_f_matrix=np.array(Parameters_function).reshape(len(Theta_values),len(Lambda_values))
  Parameters_f_matrix[Parameters_f_matrix==inf]=-8e8

  #from pylab import *
  from mpl_toolkits.mplot3d.axes3d import Axes3D
  X,Y=np.meshgrid(Lambda_values,Theta_values)
  fig,ax=plt.subplots()
  p=ax.pcolor(X,Y,Parameters_f_matrix,cmap=cm.coolwarm)
  cb=fig.colorbar(p,ax=ax)
  plt.ylabel(r'Amplitude $\theta$',fontsize=20)
  plt.xlabel(r'Length scale $\lambda$',fontsize=20)
  plt.tight_layout()
  show()
  plt.close()
  fig=plt.figure()
  ax=fig.add_subplot(1,1,1,projection='3d')
  p=ax.plot_surface(X,Y,Parameters_f_matrix*1e-8,rstride=1,cstride=1,cmap=cm.coolwarm,linewidth=0.)
  cb=fig.colorbar(p)
  plt.ylabel(r'Amplitude $\theta$',fontsize=20)
  plt.xlabel(r'Length scale $\lambda$',fontsize=20)
  plt.tight_layout()
  show()
  plt.close()

################################################################################################################
################################################################################################################

