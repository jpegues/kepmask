###FILE: demoKep.py
###PURPOSE: Script that demonstrates how to generate Keplerian masks and plot channel maps.




##Below Section: Imports any necessary functions
import numpy as np
import modAstro as astmod
import modKep as kepmod
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.close()
#Useful constants
try:
    import astropy.constants as astconst
    pc0 = astconst.pc.value #m
    au0 = astconst.au.value #m
    msun = astconst.M_sun.value #kg
except ImportError:
    pc0 = 3.08567758e+16 #m
    au0 = 1.49597871e+11 #m
    msun = 1.98847542e+30 #kg
pi = np.pi
#



#----------------------------------------------------------------------------------
##USER INPUTS:
#File, Loading, and Saving Parameters
fitsname = "J1604_H2CO_303-202_image.fits" #Path and name of .fits file containing emission channels
linename = "j1604_h2co303202" #A name that will be appended to any saved files (e.g., a channel map or spectrum); set linename="" to not append anything

#Mask Parameters
doloadmask = False #If doloadmask=True, will load masks from maskloadname; if doloadmask=False, will generate Keplerian masks from scratch
maskloadname = "chanmaskold_"+linename+".npy" #.npy file; name of Keplerian masks file
masksavename = "chanmasknew_"+linename+".npy" #Only used if doloadmask=False.  If masksavename=None, then the generated Keplerian masks will not be saved.  If masksavename=<a valid string name>, then the newly-generated Keplerian masks will be saved as the given masksavename

#Plot Parameters
doplots = True #If True, will plot a channel map and a spectrum
if doplots: #The rest of this block is activated only if doplots=True
    #For channel map
    chansavename = "chanmap_"+linename+".png" #If chansavename=None, then the channel map will be displayed instead of saved.  If chansavename=<a valid string name>, then the channel map will be saved as chansavename rather than displayed
    whichchan_list = np.arange(17, 30+1, 1).astype(int) #Array of integer indices or None; index range of channels to plot in the channel map; set to None to plot all available channels
    #
    #For spectrum
    specsavename = "spec_"+linename+".png" #If specsavename=None, then the spectrum will be displayed instead of saved.  If specsavename=<a valid string name>, then the spectrum will be saved as specsavename rather than displayed


#Stellar and Disk Parameters
mstar = 0.95 *msun #[kg]; Stellar mass
sysvel = 4.6 *1E3 #[m/s]; Systemic velocity
dist = 150.1 *pc0 #[m]; Distance to source
posang = (189.7) *pi/180.0 #[radians]; Position angle
incang = (10.0) *pi/180.0 #[radians]; Inclination angle
midx = 254 #[index]; Location along x-axis of midpoint
midy = 240 #[index]; Location along y-axis of midpoint
rmax = 400 *au0 #[m]; Desired boundary (radial distance in meters) of Keplerian masks


#Hyperfine Parameters
freqlist = None #array of floats [Hz]; set freqlist=None if no hyperfine fitting is desired.  If hyperfine masking (i.e., combined masking) is desired, then set freqlist as the list of rest frequencies for all lines to consider.  The first value in freqlist should be the main transition (all other transitions will be shifted from the main transition)


#Broadening Parameters
whichbroad = "yen" #str; Type of broadening to use.  Can be either "yen" for Yen et al. 2016 broadening scheme, or "thermal" for thermal+turbulent broadening scheme

#For thermal broadening scheme (required only if whichbroad="therm")
broadtherm_umol = None #float or None; mean molecular weight
broadtherm_mmol = None #float [kg] or None; molecular mass
broadtherm_alpha = None #float or None; alpha-parameter (related to viscosity)
broadtherm_Tqval = None #float or None; power-law index for temperature profile
broadtherm_T0 = None #float [K] or None; midplane temperature at distance given by broadtherm_r0
broadtherm_r0 = None #float [AU] or None; midplane distance corresponding to broadtherm_T0

#For Yen et al. 2016 broadening scheme (required only if whichbroad="yen")
broadyen_pre0 = 0.175 *1E3 #float [m/s] or None; pre-factor for Yen et al. 2016 profile at distance given by broadyen_r0
broadyen_r0 = 100 #float [AU] or None; radius corresponding to broadyen_pre0
broadyen_qval = 0.2 #float or None power-law index for Yen et al. 2016 profile


#General Plot Parameters
#For channel map
plotscale = 1 #float that is > 0; scales the size of the overall channel map
rowscale = 2 #float that is > 2; scales how the size of the channel map increases with the number of rows
cmap = plt.cm.bone_r #Matplotlib colormap
ncol = 10 #Number of columns in the channel map
vmin = 0 #int/float or None; minimum value for the plot's colorbar; if None, will take the minimum value across the plotting range of the channel map to be vmin
vmax = None #int/float or None; maximum value for the plot's colorbar; if None, will take the maximum value across the plotting range of the channel map to be vmax
limchan = [-3, 3] #2-number list [arcsec] or None; the x-axis and y-axis range to be plotted for channel map; plots full range if None
#For spectrum
speccolor = "purple" #Color of the spectrum
specwidth = 4.0 #Line width of the spectrum
specalpha = 0.75 #Translucence of the spectrum
specstyle = "-" #Line style of the spectrum
limspec = [-1.5, 1.5] #2-number list [km/s] or None; the x-axis and y-axis range to be plotted for spectrum; plots full range if None


#Channel Map Value Scaling Parameters
emscale = 1E3 #Converts emission from [Jy/beam] -> [mJy/beam]
xscale = 180.0/pi*3600 #Converts x-axis (RA) from [radians] -> [arcsec]
yscale = 180.0/pi*3600 #Converts y-axis (DEC) from [radians] -> [arcsec]
vscale = 1.0/1E3 #Converts velocities from [m/s] -> [km/s]
xlabel = r"$\Delta$R.A."+" [\"]" #X-axis label
ylabel = r"$\Delta$Dec."+" [\"]" #Y-axis label


#Spectrum Value Scaling Parameters
specscale = 1.0E3 #Converts y-axis from [Jy] -> [mJy]
velscale = 1.0/1.0E3 #Converts x-axis from [m/s] -> [km/s]
speclabel = "mJy" #Label for the y-axis of the spectrum
vellabel = r"(v - v$_{sys}$) [km/s]" #Label for the x-axis of the spectrum


#Tick Label, Axis Label, and Font Parameters
tickwidth = 3 #Width of plot ticks
tickheight = 5 #Height of plot ticks
ticksize = 14 #Font size for tick labels
textsize = 16 #Font size for in-graph labels 
titlesize = 16 #Font size for titles


#Channel Map Colorbar Parameters
docbar = True #If True, will plot a colorbar for the channel map
cbarlabel = "[mJy/beam]" #Title of colorbar


#Channel Map Beam Parameters
plotbeam = True #If True, will plot a beam at the bottom-left of the plot
beamx = 0.80 #float in interval [0, 1]; scaled x-axis location of plotted beam, where 0 will put the beam all the way to the left of the channel, and 1 will put the beam all the way to the right
beamy = 0.20 #float in interval [0, 1]; scaled y-axis location, of plotted beam, where 0 will put the beam all the way to the bottom of the channel, and 1 will put the beam all the way to the top


#Channel Map Velocity In-Box Label Parameters
velx = 0.10 #float in interval [0, 1]; scaled x-axis location of velocity in-graph label for each channel, where 0 will put the label all the way to the left of the channel, and 1 will put the label all the way to the right
vely = 0.85 #float in interval [0, 1]; scaled y-axis location of velocity in-graph label for each channel, where 0 will put the label all the way to the bottom of the channel, and 1 will put the label all the way to the top
velalpha = 0.6 #Opacity of velocity box
velboxcolor = "white" #Color of velocity box
velboxedge = "white" #Color of edge of velocity box


#Channel Map User-Defined In-Box Label Parameters
boxtexts = ["J16042165"] #list of str; strings that can be plotted as in-graph labels (useful, for example, for plotting a source's name); if None, will not plot any such in-graph labels
boxxs = [0.05] #list of floats in interval [0, 1]; scaled x-axis location of any given user-defined in-graph labels, where 0 will put the label all the way to the left of the channel, and 1 will put the label all the way to the right
boxys = [0.05] #list of floats in interval [0, 1]; default=0.90: Scaled y-axis location of any given user-defined in-graph labels, where 0 will put the label all the way to the bottom of the channel, and 1 will put the label all the way to the top
boxind = 0 #Index of the channel where user would like to plot any given user-defined labels


#Channel Map Mask Contour Parameters
cstyle = "--" #Line style for mask contours
cwidth = 2 #Line width for mask contours
ccolor = "black" #Color of mask contours
calpha = 0.7 #Opacity of mask contours



mpl.rcParams["axes.linewidth"] = 2.5 #Width of subplot border




##NOTE: USER DOES NOT NEED TO MODIFY ANYTHING BELOW THIS LINE
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#

#----------------------------------------------------------------------------------
##PROCESS THE FITS FILE:
##Below Section: Reads in and processes the given .fits file
chandict = astmod.extract_fits(filename=fitsname, filetype="chan")
chanmatr = chandict["emmatr"] #[Jy/beam], list of channel matrices of emission
#For channel beam
bmaj = chandict["bmaj"]*pi/180.0 #[rad], full (not half) axis
bmin = chandict["bmin"]*pi/180.0 #[rad], full (not half) axis
bpa = chandict["bpa"]*pi/180.0 #[rad]
#For channel velocities
velarr = chandict["velarr"] #[m/s]
velwidth = chandict["velwidth"] #[m/s]
veldeltarr = velarr - sysvel #m/s
#For RA
raarr = chandict["raarr"]*pi/180.0 #[rad]
rawidth = chandict["rawidth"]*pi/180.0 #[rad]
radeltarr = raarr - raarr[midx] #[rad]
#For DEC
decarr = chandict["decarr"]*pi/180.0 #[rad]
decwidth = chandict["decwidth"]*pi/180.0 #[rad]
decdeltarr = decarr - decarr[midy] #rad
#




#----------------------------------------------------------------------------------
##GENERATE SET OF KEPLERIAN MASKS:
#Below loads a previously-generated Keplerian mask, if so desired
if doloadmask:
    mask_list = np.load(maskloadname)

#Otherwise, below generates a list of Keplerian masks from scratch
else:
    #Below generates the masks
    mask_list = np.asarray(kepmod.calc_Kepvelmask(
        xlen=len(chanmatr[0][0]), ylen=len(chanmatr[0]),
        vel_arr=velarr, bmaj=bmaj, bmin=bmin, bpa=bpa, midx=midx, midy=midy,
        rawidth=rawidth, decwidth=decwidth, velwidth=velwidth, mstar=mstar,
        posang=posang, incang=incang, dist=dist, sysvel=sysvel,
        beamfactor=3.0, beamcut=0.03,
        whichbroad=whichbroad, freqlist=freqlist,
        broadtherm_umol=broadtherm_umol, broadtherm_mmol=broadtherm_mmol,
        broadtherm_alpha=broadtherm_alpha, broadtherm_Tqval=broadtherm_Tqval,
        broadtherm_T0=broadtherm_T0, broadtherm_r0=broadtherm_r0,
        broadyen_pre0=broadyen_pre0, broadyen_r0=broadyen_r0,
        broadyen_qval=broadyen_qval,
        showtests=False, testsavename="testing.png",
        showmidtests=False, midtestsavename="testinginner.png",
        emsummask=None, rmax=rmax, cmap=cmap))
    #Below saves the masks, if so desired
    if masksavename is not None:
        np.save(masksavename, mask_list)
#




#----------------------------------------------------------------------------------
##PLOT A CHANNEL MAP:
if doplots:
    ##For channel map
    #Plot a channel map, overplotted with the Keplerian masks
    astmod.plot_channels(
        chanall_list=chanmatr*emscale,
        maskall_list=mask_list, whichchan_list=whichchan_list, x_arr=radeltarr*xscale,
        y_arr=decdeltarr*yscale, bmaj=bmaj, bmin=bmin, bpa=bpa,
        velall_arr=velarr*vscale,
        cmap=cmap, velx=velx, vely=vely, ncol=ncol, velalpha=velalpha,
        velboxcolor=velboxcolor, velboxedge=velboxedge, vmin=vmin, vmax=vmax,
        plotscale=plotscale, spec=rowscale, cstyle=cstyle, cwidth=cwidth, ccolor=ccolor,
        calpha=calpha, xlabel=xlabel, ylabel=ylabel, tickwidth=tickwidth,
        tickheight=tickheight, boxtexts=boxtexts, boxxs=boxxs, boxys=boxys,
        boxind=boxind, docbar=docbar, cbarlabel=cbarlabel,
        xlim=limchan, ylim=limchan,
        beamx=beamx, beamy=beamy, ticksize=ticksize, textsize=textsize,
        titlesize=titlesize, plotbeam=plotbeam)
    #Save the channel map, if so desired
    if chansavename is not None:
        plt.savefig(chansavename)
        plt.close()
    #Otherwise, will display the map
    else:
        plt.show()
    #
    #
    ##Plot a spectrum
    #Calculate beam area
    brarea_raw = pi/1.0/np.log(2)*(bmaj/2.0*bmin/2.0) #Beam area [rad^2]
    pixrarea_raw = np.abs(rawidth*decwidth) #Pixel area [rad^2]
    beamarea = brarea_raw/1.0/pixrarea_raw #Desired beam area [pix]
    #
    #Calculate spectrum
    specarr_raw = np.array([chanmatr[ai][mask_list[ai]].sum() for ai in range(0, len(chanmatr))]) #Sum up emission in each mask for each channel
    specarr = specarr_raw/1.0/beamarea*specscale #Unit conversion from Jy/beam -> mJy
    #
    #Plot spectrum
    plt.plot((velarr-sysvel)*velscale, specarr, drawstyle="steps-mid", color=speccolor, linewidth=specwidth, alpha=specalpha, linestyle=specstyle) #Plot the spectrum
	if limspec is not None:
        plt.xlim(limspec) #Change the x-axis range of the plot
    plt.ylabel(speclabel)
    plt.xlabel(vellabel)
    if specsavename is not None:
        plt.savefig(specsavename) #Save the figure as "testing.png"
        plt.close() #Close the figure
    else:
        plt.show()



