###FILE: _utilsKep.py
###PURPOSE: Script that contains (background, user-unfriendly) functions for generating Keplerian masks.
###NOTE: The functions in this script are NOT meant to be accessed directly by package users!  Users should only need to use the class contained within kepmask.
###github.com/jpegues/kepmask



##BELOW SECTION: Imports necessary functions
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
try:
    import astropy.constants as astconst
    cconst = astconst.c.value #m/s
    G0 = astconst.G.value #m^3/kg/s^2
    au0 = astconst.au.value #m
    pc0 = astconst.pc.value #m
    msun = astconst.M_sun.value #kg
    kB = astconst.k_B.value #J/K
except ImportError:
    cconst = 299792458 #m/s
    G0 = 6.67384e-11 #m^3/kg/s^2
    au0 = 1.49597871e+11 #m
    pc0 = 3.08567758e+16 #m
    msun = 1.9891e+30 #kg
    kB = 1.38064852e-23 #J/K
plt.close()
pi = np.pi
#



##FUNCTION: calc_Kepvelmask
##PURPOSE: Generates masks that predict emission as a function of line-of-sight Keplerian velocity for a given disk.
##INPUTS:
##    Dimension Parameters
##    - xlen [int; required]: Length of x-axis for each mask
##    - midx [int; required]: Index along the x-axis of the mask center
##    - rawidth [int/float [radians]; required]: X-axis (R.A.) width in [radians]...
##      ...of each pixel
##    - ylen [int; required]: Length of y-axis for each mask
##    - midy [int; required]: Index along the y-axis of the mask center
##    - decwidth [int/float [radians]; required]: Y-axis (DEC.) width in [radians]...
##      ...of each pixel
##    - vel_arr [array; required]: Array of velocities for each mask
##    - velwidth [int/float [m/s]; required]: Channel width in [m/s]...
##      ...of the velocity; assumed to be the same between each channel
##
##    Mask Extent Parameters
##    - emsummask [2D array; default=None]: If given, will use an unbiased mask...
##      ...(such as a continuum mask) to decide the edges of the Keplerian masks;...
##      ...if None given, then will not use emsummask to set the mask edges
##    - rmax [int/float [m]; default=None]: If given, will use as a radius cuttoff...
##      ...to decide the edges of the Keplerian masks; should be in meters;...
##      ...if None given, then will not use rmax to set the mask edges
##
##    Stellar and Geometric Parameters
##    - mstar [int/float [kg]; required]: Stellar mass in [kg]
##    - posang [int/float [radians]; required]: Position Angle in [radians]
##    - incang [int/float [radians]; required]: Inclination Angle in [radians]
##    - dist [int/float [m]; required]: Distance to source in [m]
##    - sysvel [int/float [m/s]; required]: Systemic velocity of the source in [m/s]
##
##    Hyperfine Parameters (optional)
##    - freqlist [list of floats [Hz]; default=None]: If hyperfine masking (i.e.,...
##    - ...combined masking) is desired, then set freqlist as the list of rest...
##    - ...frequencies for all lines to consider.  The first value in freqlist...
##    - ...should be the main transition (all other transitions will be shifted...
##    - ...from the main transition)
##
##    Beam and Beam Smearing Parameters
##    - bmaj [int/float [radians]; required]: Full (not half) width in [radians]...
##      ...of the beam major axis
##    - bmin [int/float [radians]; required]: Full (not half) width in [radians]...
##      ...of the beam minor axis
##    - bpa [int/float [radians]; required]: Angle of the beam
##    - beamfactor [int/float; default=3.0]: See conv_circle() function in this script
##    - ...Controls beam convolution approximation.
##    - beamcut [int/float; default=0.3]: Controls beam convolution normalization.
##
##    Broadening Parameters
##    Yen et al. 2016 Broadening Parameters
##    - broadyen_pre0 [float [m/s]; default=None]: pre-factor for Yen et al. 2016...
##    - ...profile at distance given by broadyen_r0
##    - broadyen_r0 [float [AU]; default=None]: radius corresponding to...
##    - ...broadyen_pre0
##    - broadyen_qval [float; default=None]: power-law index for Yen et al. 2016...
##    - ...profile
##
##    Testing Parameters
##    - showtests [bool; default=False]: If True, will plot tests of masks
##    - cmap [matplotlib colorbar instance; required]: Colormap to use for...
##      ...colorbar of any tests plotted
##NOTES:
##    - Units of radians, kg, and m/s as applicable, EXCEPT FOR BROADENING PARAMETER R0_AU, WHICH IS IN AU.
def calc_Kepvelmask(xlen, ylen, vel_arr, bmaj, bmin, bpa, midx, midy,
            rawidth, decwidth, velwidth, mstar, posang, incang, dist,
            sysvel, whichchans, freqlist=None,
            broadyen_pre0=None, broadyen_r0=None, broadyen_qval=None,
            beamfactor=3.0, beamcut=0.03,
            showtests=False, radeltarr=None, decdeltarr=None,
            emsummask=None, rmax=None, cmap=plt.cm.hot_r):
    ##Below Section: Raises warning if conflicts in desired cutoffs
    if (emsummask is not None) and (rmax is not None): #Warning if both given
        print("Both mask-override and max. R specified! Choosing max. R.")
        emsummask = None


    ##Below Section: If hyperfine lines, prepares velocity shifts of multiple masks
    if freqlist is not None: #If hyperfine/combined masks desired
        #Below calculates velocity shifts
        sysvelarr = [(sysvel + conv_freqtovel(
                                        freq=freqlist[ai], restfreq=freqlist[0]))
                for ai in range(0, len(freqlist))] #m/s
    else: #If no hyperfine lines given, will just calculate 1 mask set as normal
        sysvelarr = [sysvel]
    sepmasklist = [None]*len(sysvelarr) #Relevant only if hyperfine masks desired


    ##Below Section: Calculates expected velocity field for the given disk
    ##NOTE: For hyperfine/combined masks, calculates each mask separately;...
    ##    technically they could be calculated simultaneously, but for now are...
    ##    calculated separately (and thus more slowly) to allow for cleaner code
    for ai in range(0, len(sysvelarr)):
        dictres = calc_Kepvelfield(xlen=xlen, ylen=ylen,
            rawidth=rawidth, decwidth=decwidth, mstar=mstar,
            radeltarr=radeltarr, decdeltarr=decdeltarr,
            posang=posang, incang=incang, dist=dist, showtests=showtests,
            rmax=rmax, midx=midx, midy=midy, sysvel=sysvelarr[ai])
        velmatr = dictres["velmatr"] #Deprojected velocities in [m/s]
        rmatr = dictres["rmatr"] #Deprojected radii in [m]
        yoverr = dictres["y/r"] #y-axis over r [unitless ratio]


        ##Below Section: Incorporates broadening due to Yen+2016
        vdeltyen = calc_broad_yen(rmatr=rmatr, turbpreval=broadyen_pre0,
                    r0=broadyen_r0, qval=broadyen_qval)
        quadwidth = np.sqrt((vdeltyen**2) + (velwidth**2))
        #Tests, if so desired
        if showtests:
            #Thermal broadening width plot
            extarr = np.array([radeltarr[0], radeltarr[-1],
                            decdeltarr[-1], decdeltarr[0]])*180.0/pi*3600
            phere = plt.imshow(vdeltyen/1.0E3,
                        extent=extarr, cmap=cmap)
            cbar = plt.colorbar(phere)
            plt.suptitle("TEST PLOT: Broadening via Yen+2016 scheme: "
                    +"M_star = {0:.2f} M_Sun".format(mstar/1.0/msun))
            plt.title("PA = %.2f deg" % (posang*180.0/pi)
                    +(", Inc = %.2f deg" % (incang*180.0/pi))
                    +(", Dist = %.2f pc" % (dist/1.0/pc0)))
            plt.xlabel("RA [\"]")
            plt.ylabel("DEC [\"]")
            cbar.set_label(r"km s$^{-1}$", rotation=270, labelpad=20)
            plt.show()


        #Below Section: Calculates velocity masks (when velocities in ranges)
        if (whichchans is not None):
            tmpinds = whichchans
        else:
            tmpinds = np.arange(0, len(vel_arr), 1)
        #
        #curmasklist = [] #To hold current mask set
        curmasklist = np.zeros(
            shape=(len(vel_arr), velmatr.shape[0], velmatr.shape[1])
        ) #To hold current mask set
        #for vi in tmpinds: #range(0, len(vel_arr)):
        for vi in range(0, len(vel_arr)):
            #Skip this channel if not a requested channel
            if (vi not in tmpinds):
                continue

            #Below adds thermal broadening and channel width in quadrature
            velhere = vel_arr[vi]
            maskhere = ((velmatr >= velhere - (quadwidth/2.0))
                    *(velmatr <= velhere + (quadwidth/2.0))
                                    ).astype(float)

            #Below converts beam from radian units to pixel (pts) units
            beamsetptshere = change_beamradtopts(bmaj=bmaj, bmin=bmin,
                    bpa=bpa, rawidth=rawidth, decwidth=decwidth)
            bmajpts = beamsetptshere[0] #[pts]
            bminpts = beamsetptshere[1] #[pts]

            #Below applies approx. beam conv. (doesn't account for direction)
            maskhere = conv_circle(matr=maskhere, bmaj=bmajpts, bmin=bminpts,
                        factor=beamfactor)
            if maskhere.max() != 0: #If actual mask there; avoids 0/0
                maskhere = maskhere/1.0/maskhere.max() #Scales so max=1
            maskhere[maskhere < beamcut] = 0 #Chops mask at given norm. cut

            #Below applies radial cutoffs, if so desired
            if emsummask is not None: #If overall mask of emission given
                maskhere[~emsummask] = 0
            if rmax is not None: #If outer edge of disk known
                maskhere[rmatr > rmax] = 0
            maskhere = maskhere.astype(bool)
            curmasklist[vi] = maskhere #.append(maskhere)


        #Below Section: "Interpolates" for masks not shown due to low pixel res.
        sysloc = np.argmin(np.abs(vel_arr - sysvelarr[ai])) #Ind. nearest sys vel
        #Below checks masks to the left of the systemic velocity (sys vel)
        for vi in range(np.min(tmpinds), sysloc+1-1):
            if (vi == 0): #If at start of channels
                continue
            #Copies over previous mask if doesn't show up for this channel
            if ((curmasklist[vi].max() == False)
                        and (curmasklist[vi-1].max() == True)):
                curmasklist[vi] = curmasklist[vi-1] #.copy() #Copy mask

        #Below checks masks at AND to the right of the systemic velocity
        for vi in range(sysloc, (np.max(tmpinds)+1))[::-1]: #Reversed direction
            if (vi == len(vel_arr) - 1): #If at end of channels
                continue
            #Copies over previous mask if doesn't show up for this channel
            if ((curmasklist[vi].max() == False)
                        and (curmasklist[vi+1].max() == True)):
                curmasklist[vi] = curmasklist[vi+1] #.copy() #Copy mask

        #Below records this mask set within the overall mask list
        sepmasklist[ai] = curmasklist


    ##Below Section: Adds mask sets (relevant only if hyperfine masks desired)
    finalmasklist = np.asarray(sepmasklist[0]).copy()
    for ai in range(1, len(sysvelarr)):
        finalmasklist = finalmasklist + np.asarray(sepmasklist[ai])


    ##Below Section: Returns calculated mask set
    return finalmasklist.astype(bool)
#



##FUNCTION: calc_Kepvelfield
##PURPOSE: Calculates a model of the Keplerian velocity (along the line of sight) of a disk based on that disk's angles, solar mass, and beam.
##SOURCE OF TECHNIQUE: (but using my own radial projection code)
##   Yen et al. 2016: http://iopscience.iop.org/article/10.3847/0004-637X/832/2/204/pdf
##NOTES:
##    -
def calc_Kepvelfield(xlen, ylen, midx, midy, rawidth, decwidth, mstar, posang, incang,
            dist, sysvel, rmax=None, showtests=False,
            radeltarr=None, decdeltarr=None):
    ##Below Section: Calculates deprojected radius matrix and velocity matrix
    #x,y in angular space
    indmatrs = np.indices(np.zeros(shape=(ylen, xlen)).shape)
    xangmatr = (indmatrs[1] - midx)*rawidth #radians
    yangmatr = (indmatrs[0] - midy)*decwidth #radians

    #Below rotates angular indices (clockwise!) by position angle
    #In different coordinate system
    #xrotangmatr = (xangmatr*np.cos(posang)) + (yangmatr*np.sin(posang)) #radians
    #yrotangmatr = (yangmatr*np.cos(posang)) - (xangmatr*np.sin(posang)) #radians
    #In typical astronomy coordinate system - see Yen+2016
    xrotangmatr = (xangmatr*np.sin(posang)) + (yangmatr*np.cos(posang)) #radians
    yrotangmatr = (yangmatr*np.sin(posang)) - (xangmatr*np.cos(posang)) #radians

    #Below calculates deprojected radius matrix in physical units
    rangmatr = np.sqrt((xrotangmatr**2)
                + ((yrotangmatr/1.0/np.cos(incang))**2)) #Radius in [rad]
    rposmatr = np.tan(rangmatr)*dist #Radii in physical units


    ##Below Section: Calculates deprojected velocity matrix
    velmatr = ((xrotangmatr/1.0/rangmatr)
                *np.sin(incang)*np.sqrt(G0*mstar/1.0/rposmatr))
    velmatr = velmatr + sysvel #Adds in systemic velocity
    velmatr[rposmatr == 0] = sysvel #Sets central value with 0 (since nan where r=0)

    #Tests, if so desired
    if showtests:
            #Plot the line-of-sight velocity
            extarr = np.array([radeltarr[0], radeltarr[-1],
                            decdeltarr[-1], decdeltarr[0]])*180.0/pi*3600
            phere = plt.imshow(velmatr/1.0E3,
                        extent=extarr, cmap=plt.cm.RdBu_r)
            cbar = plt.colorbar(phere)
            plt.suptitle("TEST PLOT: Line-of-sight velocity: "
                    +"M_star = {0:.2f} M_Sun".format(mstar/1.0/msun))
            plt.title("PA = %.2f deg" % (posang*180.0/pi)
                    +(", Inc = %.2f deg" % (incang*180.0/pi))
                    +(", Dist = %.2f pc" % (dist/1.0/pc0)))
            plt.xlabel("RA [\"]")
            plt.ylabel("DEC [\"]")
            cbar.set_label(r"km s$^{-1}$", rotation=270, labelpad=20)
            plt.show()


    ##Below Section: Returns the results
    return {"velmatr":velmatr, "rmatr":rposmatr, "y/r":(yrotangmatr/1.0/rangmatr)}
#



##FUNCTION: calc_broad_yen
##PURPOSE: Calculates thermal/non-thermal velocity broadening using the equation in Yen et al. 2016.
##NOTES:
#    - Inputs should be m/s and AU, as applicable
def calc_broad_yen(rmatr, r0=100, turbpreval=0.1*1E3, qval=0.2):
    deltvmatr = turbpreval*((rmatr/1.0/(r0*au0))**(-1*qval)) #m/s
    return 4*deltvmatr
#



##FUNCTION: conv_circle
##PURPOSE: Convolves a given matrix with a Gaussian, truncated at a circle of the given radii.
##NOTES:
##    - Units of radians as applicable
def conv_circle(matr, bmaj, bmin, factor):
    #Below applies and returns matrix smoothed over given area
    #NOTE: For now, turns ovular beam into a circle
    trunc = np.sqrt(bmaj*bmin) #uses diameter, not radius ####/(4.0*np.log(2))) #circular radius approx. of gaus. beam
    return ndi.filters.gaussian_filter(matr, sigma=(trunc/1.0/factor))
#



###FUNCTION: conv_freqtovel (8-17-18)
###PURPOSE: This function is meant to convert from frequency to velocity
def conv_freqtovel(freq, restfreq):
    return cconst*(1.0 - (freq/1.0/restfreq))
#



##FUNCTION: change_beamradtopts
##PURPOSE: Converts a beam in radian units to pixel (point) units.
##NOTES:
##    - Units of radians as applicable
def change_beamradtopts(bmaj, bmin, bpa, rawidth, decwidth):
    #Below first projects bmaj to x[rad] and y[rad] coordinates
    #Note that x-sin and y-cos; since for bmaj, PA measured North to East
    bmaj_xrad = bmaj*np.sin(bpa)
    bmaj_yrad = bmaj*np.cos(bpa)
    bmaj_xpts = bmaj_xrad/1.0/rawidth #Converts to pts
    bmaj_ypts = bmaj_yrad/1.0/decwidth #Converts to pts

    #Next projects bmin to x[rad] and y[rad] coordinates
    #Note that x-cos and y-sin; since for bmin, PA measured East to South
    bmin_xrad = bmin*np.cos(bpa)
    bmin_yrad = bmin*np.sin(bpa)
    bmin_xpts = bmin_xrad/1.0/rawidth #Converts to pts
    bmin_ypts = bmin_yrad/1.0/decwidth #Converts to pts

    #Finally calculates the length for bmaj and bmin in points
    bmajpts = np.sqrt((bmaj_xpts * bmaj_xpts) + (bmaj_ypts * bmaj_ypts)) #[pts]
    bminpts = np.sqrt((bmin_xpts * bmin_xpts) + (bmin_ypts * bmin_ypts)) #[pts]

    #Below returns bmaj and bmin in pts units
    return [bmajpts, bminpts]
#
