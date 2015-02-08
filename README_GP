The Gaussian file allows to fit with accuracy any equation of state around a minimum. 
This is implemented as a python function that can be called in any other python script, such as the electronic_thermo.py script

The function GP_fit:
First, the training data is fit with a quadratic polynomial.
Second, the error on the fit is reduced with a Gaussian fit (since it works well with a distribution of points around 0).
The function return the best fit of different quantities:
  1- function with/withou error on the training data
  2- first derivative  """
  3- second derivative """
  
The function GP_fit_*** :
  Are used to output just a a single quantity in order to be used with other python built_in functions for minimization in 1-D
  
The function check_hyperparams_convergence:
  Is used to optimize the hyperparameters of the Gaussian Process. However numerical problems can be found 
  (the hyperparameters now are fixed by eye)
