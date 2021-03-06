import numpy as np
import matplotlib.pyplot as plt 
#import matplotlib.gridspec as gridspec
import pylab
import scipy
from matplotlib import rc     
from pylab import *
#from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
#from scipy import optimize
from scipy.integrate import simps
from scipy.optimize import bisect
import scipy.interpolate
from scipy.interpolate import interp1d
from scipy import interpolate
from glob import glob
import os.path


#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Calcola contributo termodinamico di elettroni - Si assume gas di elettroni non interagenti
# Daniele Dragoni 02/02/2015
print "--------------------------------------------------"
print "Always check: Number of electrons to deal with !!!"
print "Are you working in the limit of histogram bins << kb*T !?"
print "Are you integrating the whole DOS from core states up to a cutoff values above the Fermi energy?"
print "--------------------------------------------------"
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




#-------------------\ DEFINE functions & CONSTANTS /-----------------
#--------------------------------------------------------------------

Kb          =8.617332478e-5		# Boltzmann constant (eV/K)
eVtoRy      =0.073498618
BtoAng      =0.529177249
evtoJ       =1.60217733e-19
Navog       =6.02214078e23
N_el_valence=8 				# Dovrebbe essere 16 quando integro tutta la DOS !!!! DA CAMBIAREEEEE !!!
Delta       =20. 			# Range attorno Efermi entro cui cerco soluzione pot chimico
bins        =7600.			# Number of integration steps in integrating the DOS
check_convergence = True		# Perform check convergence wrt integration step 


def func(en,chem_pot,T):		# Fermi-Dirac occupation
    return 1/(np.exp((en-chem_pot)/(Kb*T))+1)

def zeri_pot(chem_pot,*args):
    en=args[0]
    T=args[1]
    num_electron=args[2]
    dos=args[3] 
    return num_electron-simps(dos*func(en,chem_pot,T),en)

def birchm(V,E0,V0,B0,B01):
    return E0+9*V0*B0/16*( B01*((V0/V)**(2/3.)-1)**3 + \
	   (((V0/V)**(2/3.)-1)**2) * (6-4*(V0/V)**(2/3.)) ) 

def birchm_1der(V,E0,V0,B0,B01):
    return -3/8.*B0*(V0/V)**(5/3.)*( (V0/V)**(2/3.)-1)*(16.-12*(V0/V)**(2/3.)+ \
	    3*B01*((V0/V)**(2/3.)-1))
#--------------------------------stop--------------------------------
#--------------------------------------------------------------------





#-----------------------\ LOAD FILES & CHECK /-----------------------
#--------------------------------------------------------------------

fnames  = sorted(glob('/home/dragoni/deneb.rsync/data/dragoni/DOS_BCC_Fe/Fe.dos.Cub-*.dat'))
Lattices= np.genfromtxt('/home/dragoni/deneb.rsync/data/dragoni/DOS_BCC_Fe/Lattices.dat')
Fermi_en= np.genfromtxt('/home/dragoni/deneb.rsync/data/dragoni/DOS_BCC_Fe/Fermi_energies.dat')

if Lattices.size==Fermi_en.size and Lattices.size==len(fnames):
    pass
else:
    raise ValueError('Arrays lengths are different !? Check better!! ')
#--------------------------------stop--------------------------------
#--------------------------------------------------------------------





#---------------\ LOAD ARRAYS & INTERPOLATE DATA /-------------------
#--------------------------------------------------------------------

Volumes_array=((Lattices*BtoAng)**3)/2.  		# In AA

dos          = []
enlev        = []
dos_spline   = []
enlev_spline = []
for index,filee in enumerate(fnames):
    dosfile        = np.genfromtxt(filee)
    dos.append(dosfile[:,1]+dosfile[:,2])
    #dosup.append(dosfile[:,1])
    #dosdown.append(dosfile[:,2])
    enlev.append(dosfile[:,0])
    enlev_spline.append(np.linspace(min(enlev[index]),max(enlev[index]),bins))
    tck=interpolate.splrep(enlev[index],dos[index],s=0)
    dos_spline.append(interpolate.splev(enlev_spline[index],tck,der=0))
#--------------------------------stop--------------------------------
#--------------------------------------------------------------------




# CONVERGENCE TESTS: NUMBER of electrons from DOS integration at 0K and fixed volume
#                    CHEMICAL POTENTIAL at 0K and fixed volume
#------------------------------\ /-----------------------------------
#--------------------------------------------------------------------
if check_convergence is True:

 f=interp1d(enlev[0],dos[0])
 tck=interpolate.splrep(enlev[0],dos[0],s=0)
 xnew=np.linspace(min(enlev[0]),max(enlev[0]),1900)
 ynew1=f(xnew)
 ynew2=interpolate.splev(xnew,tck,der=0)

 # Plot DOS with and without spline interpolation
 rc('text',usetex=True)
 plt.rcParams['xtick.labelsize']=20
 plt.rcParams['ytick.labelsize']=20
 plt.plot(enlev[0],dos[0],'ko-',label='QE points')
 plt.plot(xnew,ynew1,'b',label='spline_1')
 plt.plot(xnew,ynew2,'g',label='spline_2')
 plt.axvline(x=Fermi_en[0],ymin=0.,ymax=1,linewidth=1,color='r')
 plt.xlabel(r'Energy (eV)',fontsize=20)
 plt.ylabel(r'DOS',fontsize=20)
 legend(prop={"size":20},loc=0)
 plt.tight_layout()
 show()
 close()


 occupation_V0_T0=func(enlev[0],Fermi_en[0],0)		# Ferim-Dirac at 0K
 Nel_std=simps(dos[0]*occupation_V0_T0,enlev[0])  	# Integra la DOS cosi come fornita step       0.05  eV (~600K)
 Nel_interp_1=[]					# Integra la DOS con integration step piccoli 0.001 eV (~120K)
 Nel_interp_2=[]
 chem_pot1   =[]
 chem_pot2   =[]
 
 Max_number_bins=190000
 for binss in np.linspace(len(enlev[0]),Max_number_bins,50):
   xnew     = np.linspace(min(enlev[0]),max(enlev[0]),binss)
   ynew1    = f(xnew)
   ynew2    = interpolate.splev(xnew,tck,der=0)
   val1     = simps(ynew1*func(xnew,Fermi_en[0],0),xnew)
   val2     = simps(ynew2*func(xnew,Fermi_en[0],0),xnew)
   Nel_interp_1.append(val1)
   Nel_interp_2.append(val2)
   chem_pot1.append( bisect(zeri_pot,Fermi_en[0]-Delta,Fermi_en[0]+Delta,args=(xnew,0,N_el_valence,ynew1)) )
   chem_pot2.append( bisect(zeri_pot,Fermi_en[0]-Delta,Fermi_en[0]+Delta,args=(xnew,0,N_el_valence,ynew2)) ) 

 # Plot Number of electrons at 0K for the lowest volume as a function of bin width
 plt.plot(np.linspace(len(enlev[0]),Max_number_bins,50), Nel_interp_1,'k-',label='spline_1')
 plt.plot(np.linspace(len(enlev[0]),Max_number_bins,50), Nel_interp_2,'r-',label='spline_2')
 plt.plot(np.linspace(len(enlev[0]),Max_number_bins,50), np.repeat(Nel_std,50),label='QE-mesh!')
 plt.plot(np.linspace(len(enlev[0]),Max_number_bins,50), np.repeat(8,50),'k--',linewidth=2,label='Expected N_{el}')
 plt.fill_between(np.linspace(len(enlev[0]),Max_number_bins,50),np.repeat(8,50)+0.0002,np.repeat(8,50)-0.0002,alpha=0.2,facecolor='green') 
 plt.axvline(x=bins,ymin=0.,ymax=1,linewidth=1,color='r')
 plt.xlabel(r'\# Integration Step',fontsize=20)
 plt.ylabel(r'integrated N_{el}'  ,fontsize=20)
 plt.ticklabel_format(useOffset=False)
 legend(prop={"size":20},loc=0)
 plt.tight_layout()
 show()
 close()

 # Plot chemical potential at 0K for the lowest volume as a function of bin width
 plt.plot(np.linspace(len(enlev[0]),Max_number_bins,50), chem_pot1,'b-',label='spline_1')  
 plt.plot(np.linspace(len(enlev[0]),Max_number_bins,50), chem_pot2,'r-',label='spline_2')
 plt.plot(np.linspace(len(enlev[0]),Max_number_bins,50), np.repeat(Fermi_en[0],50),'k--',linewidth=2,label='Expected $\epsilon_F$')
 plt.fill_between(np.linspace(len(enlev[0]),Max_number_bins,50),np.repeat(Fermi_en[0],50)+0.0002,np.repeat(Fermi_en[0],50)-0.0002,alpha=0.2,facecolor='green') 
 plt.axvline(x=bins,ymin=0.,ymax=1,linewidth=2,color='r')
 plt.xlabel(r'Integration Step (eV)',fontsize=20)
 plt.ylabel(r'$\mu_{0K}$ chem pot (eV)'    ,fontsize=20)
 plt.ticklabel_format(useOffset=False)
 legend(prop={"size":20},loc=0)
 plt.tight_layout()
 show()
 close()
else:
 print
 print "Assuming you are converged wrt DOS integration step !  BE CAREFUL !!!"
 print
#--------------------------------stop--------------------------------
#--------------------------------------------------------------------




#------------------\ MAIN BODY - thermodynamics /--------------------
#--------------------------------------------------------------------
Temperatures_array	= np.linspace(0,1000,61)
Energy_V_T   		= np.zeros((Volumes_array.size,Temperatures_array.size))
Entropy_V_T   		= np.zeros((Volumes_array.size,Temperatures_array.size))
Nelectrons_V_T		= np.zeros((Volumes_array.size,Temperatures_array.size))
chemical_potential_V_T	= np.zeros((Volumes_array.size,Temperatures_array.size))
F_V_T         		= np.zeros((Volumes_array.size,Temperatures_array.size))

for vol_index,volume in enumerate(Volumes_array):
   for T_index,T in enumerate(Temperatures_array):
#     if T_index==0:
#	chemical_potential_V_T[vol_index,T_index]=  Fermi_en[vol_index]
#	occupation_V_T                   	 =  func(enlev[vol_index],chemical_potential_V_T[vol_index,T_index],T)
#	Energy_V_T[vol_index,T_index]     	 =  simps(enlev[vol_index]*dos[vol_index]*occupation_V_T,enlev[vol_index])
#	Entropy_V_T[vol_index,T_index]    	 =  0.		# Limite corretto x entropia
#	Nelectrons_V_T[vol_index,T_index] 	 =  simps(dos[vol_index]*occupation_V_T,enlev[vol_index])
#	F_V_T[vol_index,T_index]          	 =  Energy_V_T[vol_index,T_index]
#     else:
	chemical_potential_V_T[vol_index,T_index]=  bisect(zeri_pot,Fermi_en[vol_index]-Delta,Fermi_en[vol_index]+Delta,args=(enlev_spline[vol_index],T,N_el_valence,dos_spline[vol_index]))  		# Find Chemical potential at each V,T   >>   Bisect approach, slow but stable
	occupation_V_T                    	 =  func(enlev_spline[vol_index],chemical_potential_V_T[vol_index,T_index],T)
	Energy_V_T[vol_index,T_index]     	 =  simps(enlev_spline[vol_index]*dos_spline[vol_index]*occupation_V_T,enlev_spline[vol_index])
	tmp_V_T                           	 =  dos_spline[vol_index]*(occupation_V_T*np.log(occupation_V_T)+(1-occupation_V_T)*np.log(1-occupation_V_T))
	tmp_V_T[np.isnan(tmp_V_T)]       	 =  0.	 	# Put all nan values equal to zero due to presence of log(0) or log(0)*0  >> NB. However zero is the right limit to us
	Entropy_V_T[vol_index,T_index]   	 = -Kb*simps(tmp_V_T,enlev_spline[vol_index])	# Formula according to PRB 73, Sha-Choen (2006) gas of independent electrons  >>  DIVDED by 2 !!!!
	Nelectrons_V_T[vol_index,T_index] 	 =  simps(dos_spline[vol_index]*occupation_V_T,enlev_spline[vol_index])
	F_V_T[vol_index,T_index]          	 =  Energy_V_T[vol_index,T_index]-T*Entropy_V_T[vol_index,T_index]


# Plot data
rc('text',usetex=True)
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20

# Plot ____ as a function of T for all volumes
for vol_index,volume in enumerate(Volumes_array):
 #plt.plot(Temperatures_array,np.array(Energy_V_T[vol_index,:])*eVtoRy*1000,'-')
 plt.plot(Temperatures_array,np.array(Entropy_V_T[vol_index,:])*eVtoRy*1000,'-',label='Entropy')
 #plt.plot(Temperatures_array,np.array(F_V_T[vol_index,:])*eVtoRy*1000,'-',label='Free')
plt.xlabel(r'Temperature (K)',fontsize=19)
plt.ylabel(r'Energy (mRy)',fontsize=19)
legend(prop={"size":19},loc=0)
plt.tight_layout()
#savefig('./electronic_thermocontrib.pdf')
#plt.show()
plt.close()

# Plot chemical potential as a function of T for various volumes
for vol_index in np.array([0,5,30,33,34,35,36,37,38,39,40]):
    plt.plot(Temperatures_array,chemical_potential_V_T[vol_index,:]/Fermi_en[vol_index],label=str(('%.3f' % (Volumes_array[vol_index]))) )   
    #plt.plot(Temperatures_array,np.repeat(Fermi_en[vol_index],len(Temperatures_array,)),'k--')
plt.plot(Temperatures_array,np.repeat(1,Temperatures_array.size),'k--')
plt.xlabel(r'Temperature (K)',fontsize=19)
plt.ylabel(r'$\mu/\epsilon_F$',fontsize=19)
plt.ticklabel_format(useOffset=False)
legend(prop={"size":18},loc=0)
plt.tight_layout()
show()
close()


# Plot 3D surfaces of Free energy as a function of volume and temperature
fig   = plt.figure(figsize=(12,8))
ax    = fig.add_subplot(111, projection='3d')
y     = Volumes_array#[::-1]
x     = Temperatures_array
X,Y   = np.meshgrid(x, y)
Z     = F_V_T*eVtoRy*1000
ax.set_ylabel(r' Volume (\AA^3)',fontsize=24)
ax.set_xlabel(r' Temperature (K)',fontsize=24)
ax.set_zlabel(r'  $\mathcal{F}$ (Ry)',fontsize=24)

ax.plot_surface(X, Y, Z, rstride=1,cstride=1,alpha=0.8, cmap=cm.coolwarm,antialiased=True,linewidth=0)
ax.view_init(elev=28,azim=64)
#ax.set_ylim3d(10.67,12.15 );
ax.set_xlim3d(0, 1100);
#ax.set_zlim3d(-3, 2);
fig.tight_layout()
plt.show()
##plt.savefig('Free_electronic.pdf')
##print opt_volume
plt.close()


# Plot 3D surfaces of the Free energy as a function of volume and temperature removed the 0K contribution !!!
DeltaF= np.zeros((Volumes_array.size,Temperatures_array.size))
for nv in np.arange(len(Volumes_array)):
    for nt in np.arange(len(Temperatures_array)):
        DeltaF[nv,nt]=F_V_T[nv,nt]-F_V_T[nv,1]
DeltaF[:,0]=DeltaF[:,1]                   # put the 0K data equal to the first non-zero temperature for stability


fig   = plt.figure(figsize=(12,8))
ax    = fig.add_subplot(111, projection='3d')
y     = Volumes_array#[::-1]
x     = Temperatures_array
X,Y   = np.meshgrid(x, y)
Z     = DeltaF*eVtoRy*1000
ax.set_ylabel(r' Volume (\AA^3)',fontsize=24)
ax.set_xlabel(r' Temperature (K)',fontsize=24)
ax.set_zlabel(r'  $\Delta \mathcal{F}$ (mRy)',fontsize=24)

ax.plot_surface(X, Y, Z, rstride=1,cstride=1,alpha=0.8, cmap=cm.coolwarm,antialiased=True,linewidth=0)
##cset = ax.contourf(X, Y, Z, zdir='z', offset=-254.4, cmap=cm.coolwarm)
##cset = ax.contour(X, Y, Z, zdir='x', offset=10.67, cmap=cm.coolwarm,linewidth=5)
##cset = ax.contour(X, Y, Z, zdir='y', offset=-80, cmap=cm.coolwarm_r)
ax.view_init(elev=28,azim=64)
#ax.set_ylim3d(10.67,12.15 );
ax.set_xlim3d(0, 1100);
ax.set_zlim3d(-1.2,0.2);
fig.tight_layout()
plt.show()
##plt.savefig('Delta_Free_electronic.pdf')
plt.close()
#--------------------------------stop--------------------------------
#--------------------------------------------------------------------






#---------------\ Thermal expansion EL + VIBRATIONAL /---------------
#--------------------------------------------------------------------
print 'Always CHECK sorting of data loaded !!!!'
fnames_vib  = sorted(glob('/home/dragoni/bellatrix.rsync/data/dragoni/elastic-DAN-FM/iso/REdo/curve_freeenergy_at_t_*.dat'))

Theta = 0.0002		# GP hyperparameter - Amplitude oscillations
Lambda  = 0.08		# GP hyperparameter - Correlation length
Sigma  = 0.00008	# GP hyperparameter - Error on training data
F_vib_T     = []
V_vib_T     = []
coeff_BM_T  = []
cov_BM_T    = []
DeltaF_interpolated = []
opt_V_vib = []
opt_V_sum = []
opt_V_sum_GP      = []
opt_V_vib_GP      = []
Bulk_isoth_sum    = []
Bulk_isoth_vib    = []
for T_index,filee in enumerate(fnames_vib[::10]): 			# the number of temperatures considered must be the same 
  ##
  F_vib_T.append((np.genfromtxt(filee))[:,1])				# Units Ry
  V_vib_T.append((np.genfromtxt(filee))[:,0]*0.529177249**3)  		# To AA^3
  ##
  tck1 = interpolate.splrep(Volumes_array,DeltaF[:,T_index]*eVtoRy,s=0)	# Ry
  DeltaF_interpolated.append(interpolate.splev(V_vib_T[T_index],tck1,der=0))
  ##
  coeff_vib,cov_vib = scipy.optimize.curve_fit(birchm,xdata=V_vib_T[T_index],ydata=F_vib_T[T_index],p0=[-254,11.,0.1,4.])
  coeff_sum,cov_sum = scipy.optimize.curve_fit(birchm,xdata=V_vib_T[T_index],ydata=(F_vib_T[T_index]+DeltaF_interpolated[T_index]),p0=[-254,11.,0.1,4.])
  coeff_BM_T.append(coeff_sum)
  cov_BM_T.append(cov_sum)
  opt_V_vib.append(scipy.optimize.fmin(birchm, 11., args=(),xtol=1e-9,ftol=1e-7,maxiter=2000,disp=0)[0])
  opt_V_sum.append(scipy.optimize.fmin(birchm, 11., args=(coeff_sum[0],coeff_sum[1],coeff_sum[2],coeff_sum[3]),xtol=1e-9,ftol=1e-7,maxiter=2000,disp=0)[0])
  opt_V_vib_GP.append(scipy.optimize.fmin(GP_fit_func_noisy,x0=11.5,args=(V_vib_T[T_index],F_vib_T[T_index],Theta,Lambda,Sigma),xtol=1e-9,ftol=1e-7,maxiter=100,full_output=True,disp=0)[0][0] )
  opt_V_sum_GP.append(scipy.optimize.fmin(GP_fit_func_noisy,x0=11.5,args=(V_vib_T[T_index],(F_vib_T[T_index]+DeltaF_interpolated[T_index]),Theta,Lambda,Sigma),xtol=1e-9,ftol=1e-7,maxiter=100,full_output=True,disp=0)[0][0] )
  Bulk_isoth_sum.append(coeff_sum[2])					# Units Ry/AA^3
  Bulk_isoth_vib.append(coeff_vib[2])					# Units Ry/AA^3

Bulk_isoth_vib_GP=[]
Bulk_isoth_sum_GP=[]
for T_index,filee in enumerate(fnames_vib[::10]): 
 Bulk_isoth_vib_GP.append(GP_fit_der2_noisy(np.array([opt_V_sum_GP[T_index],opt_V_sum_GP[T_index]+0.1]),V_vib_T[T_index], F_vib_T[T_index],Theta,Lambda,Sigma)[0])
 Bulk_isoth_sum_GP.append(GP_fit_der2_noisy(np.array([opt_V_sum_GP[T_index],opt_V_sum_GP[T_index]+0.1]),V_vib_T[T_index],(F_vib_T[T_index]+DeltaF_interpolated[T_index]),Theta,Lambda,Sigma)[0])


if os.path.isfile('/home/dragoni/bellatrix.rsync/data/dragoni/elastic-DAN-FM/iso/REdo/v_vs_t-5.290_5.301_5.312_5.323_5.333_5.344_5.355_5.365_5.376_5.387_5.398_5.408_5.419_5.440_5.46.dat') is True:
   vibr_expansion=np.genfromtxt('/home/dragoni/bellatrix.rsync/data/dragoni/elastic-DAN-FM/iso/REdo/v_vs_t-5.290_5.301_5.312_5.323_5.333_5.344_5.355_5.365_5.376_5.387_5.398_5.408_5.419_5.440_5.46.dat')
   vol_vibr  =vibr_expansion[:,1]*BtoAng**3
   T_vol_vibr=vibr_expansion[:,0]
else:
   raise ValueError('Not existing Path, check it !! ')

# Plot & CHECK old results
plt.plot(Temperatures_array,opt_V_sum,linewidth=2,label='Thermal exp. vibr+el')  
plt.plot(Temperatures_array,opt_V_vib,'r--',linewidth=2,label='Thermal exp. vibr')
plt.fill_between(Temperatures_array, opt_V_vib,opt_V_sum,facecolor='g',alpha=0.2 )
plt.plot(T_vol_vibr,vol_vibr,label='Thermal exp. vibr-old-data')
plt.xlabel(r'Temperature (K)',fontsize=19)
plt.ylabel(r'$V(T)$ (\AA$^3$)',fontsize=19)
plt.ticklabel_format(useOffset=False)
legend(prop={"size":18},loc=0)
plt.tight_layout()
show()
close()


with open('VIB+ELEC_thermal_expansion.dat', 'w') as f:
   print >> f, "# Temp (K) Vol (\AA^3)"
   for T_index,T in enumerate(Temperatures_array):
       print >> f, T, opt_V_sum[T_index]



#--------------------------------stop--------------------------------
#--------------------------------------------------------------------



#-----------------------------\ C_V /--------------------------------  #RIVEDOOOOOOOOOOOOOOOOOOOOO
#--------------------------------------------------------------------
tckA             = interpolate.splrep(Temperatures_array,Entropy_V_T[0,:],s=0)		# eV
tckB             = interpolate.splrep(Temperatures_array,Energy_V_T[0,:],s=0)		# eV
C_V_electronic_A = interpolate.splev(Temperatures_array,tckA,der=1)*Temperatures_array	# eV/K
C_V_electronic_B = interpolate.splev(Temperatures_array,tckB,der=1) 			# better from Entropy cause is more stable at low T
plt.plot(Temperatures_array,C_V_electronic_A*evtoJ*Navog,label='Specific Heat$_{el}$')	# For 8 electrons !!!!!
#plt.plot(Temperatures_array,C_V_electronic_B,label='Specific Heat$_{el}$')
plt.xlabel(r'Temperature (K)',fontsize=19)
plt.ylabel(r'$C_V$ (J mol$^{-1}$ K$^{-1}$)',fontsize=19)
plt.ticklabel_format(useOffset=False)
legend(prop={"size":18},loc=0)
plt.tight_layout()
show()
close()



tckC                  = interpolate.splrep(Temperatures_array,opt_V_sum,s=0)							# Sparse results !!!
alpha_Vol_el          = interpolate.splev(Temperatures_array,tckC,der=1)/interpolate.splev(Temperatures_array,tckC,der=0) 	# 1/K
alpha_Lat_el          = alpha_Vol_el/3.												# 1/K
C_P_el_vib_correction = opt_V_sum*Temperatures_array*alpha_Vol_el**2*(np.array(Bulk_iosth_sum)/eVtoRy)				# eV/K 
#C_P_sum               = (C_V_electronic_A+C_V_vibrational[::10]/eVtoRy)+C_P_el_vib_correction					# eV/K
Bulk_adia_sum         = Bulk_iosth_sum * C_P_sum/(C_V_electronic_A+C_V_vibrational[::10]/eVtoRy)/4.5874e-4			# GPa from eV/AA^3 




with open('ELEC_Cv.dat', 'w') as f:
    print >> f, "# Temp (K)  Cv (J/mol K)"
    for T_index,T in enumerate(Temperatures_array):
       print >> f, T, C_V_electronic_A[T_index]*evtoJ*Navog
#--------------------------------stop--------------------------------
#--------------------------------------------------------------------
