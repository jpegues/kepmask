# pykepmask
**pykepmask** is a package that predicts the spatial location of emission in disks undergoing Keplerian rotation, particularly protoplanetary disks (e.g., [Rosenfeld et al. 2013](https://ui.adsabs.harvard.edu/abs/2013ApJ...774...16R/abstract); [Yen et al. 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...832..204Y/abstract)).  These masks can be used to extract the molecular line emission, such as C18O emission, and to decrease the noise incorporated into final image products (such as mom0s and spectra).  

**Check out our tutorial (tutorial_pykepmask.ipynb) for installation instructions and a walkthrough of how to use this package.**  But briefly, the full package workflow is given in the coding block below.  This block would generate masks and plot image products for the molecular line H2CO 303-202, observed toward the protoplanetary disk J1604-2130 (which is included with the tutorial):

```
###IMPORT NECESSARY PACKAGES AND ASTRONOMY CONSTANTS
#Import the kepmask module as km
from pykepmask import kepmask as km
#Import constants from astropy
import astropy.constants as astconst
msun = astconst.M_sun.value #Solar mass [kg]
pc0 = astconst.pc.value #Parsec [m]
au0 = astconst.au.value #AU [m]
#Import pi from numpy
import numpy as np
pi = np.pi


###ESTABLISH STELLAR/DISK CHARACTERISTICS FOR THE DISK J1604-2130
mstar = 1.1 *msun #[kg]; Stellar mass
vsys = 4.6 *1E3 #[m/s]; Systemic velocity
dist = 149 *pc0 #[m]; Distance to source
PA = (195) *pi/180.0 #[radians]; Position angle
inc = (6.2) *pi/180.0 #[radians]; Inclination angle
midx = 254 #[index]; Location along x-axis of midpoint
midy = 240 #[index]; Location along y-axis of midpoint


###CREATE AN INSTANCE OF THE KepMask CLASS FOR THE H$_2$CO OBSERVED TOWARD J1604-2130
kmset_j1604 = km.KepMask(mstar=mstar, vsys=vsys, dist=dist, PA=PA, inc=inc, midx=midx, midy=midy)
#Load in the molecular line data
fitsname = "J1604-2130_H2CO_303-202.fits" #This should point to the location+filename of the image cube .fits file
kmset_j1604.extract(fitsname=fitsname)


###SET THE PARAMETERS FOR THE KEPLERIAN MASK DIMENSIONS
Rmax = 350 *au0 #[m] #Set the radial boundary of the masks
whichchans = np.arange(16, 30+1, 1) #Set the channel indices that will actually have masks.
#Set the velocity broadening parameters
V0_ms = 0.15 *1E3 #Prefactor [m/s]
R0_AU = 100.0 #Characteristic radius [AU] - note that this is the sole parameter not in S.I. units!
q0 = 0.2 #Power law - note that this value should be positive (negation is already within the code)
#Generate the Keplerian masks
kmset_j1604.generate(mask_Rmax=Rmax, whichchans=whichchans, V0_ms=V0_ms, R0_AU=R0_AU, q0=q0)
masks = kmset_j1604.get_product("masks") #Now the masks variable contains the masks!


###PLOT THE MASKED IMAGE PRODUCTS
vrange_kms = [2.2, 6.8]
kmset_j1604.plot_product("channels", vrange_kms=vrange_kms, ncol=5) #Channel map
kmset_j1604.plot_product("mom0") #mom0
kmset_j1604.plot_product("spectrum", ylim=[-20, 1400]) #Spectrum
```

If you end up using our package (and we hope you do!), please cite our hard work!  You can cite us using Zenodo (linked via the shiny DOI release badge below):

![DOI button](https://zenodo.org/badge/176531775.svg)

And if you have any questions/comments/concerns/issues, please do let us know!

