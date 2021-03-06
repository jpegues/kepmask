###FILE: _utilsAstro.py
###PURPOSE: Script that contains (background, user-unfriendly) functions for performing general astronomy calculations and actions (e.g., reading in .fits files).
###NOTE: The functions in this script are NOT meant to be accessed directly by package users!  Users should only need to use the class contained within kepmask.
###github.com/jpegues/kepmask



###BELOW SECTION: Imports necessary functions
import numpy as np
import matplotlib.pyplot as plt
import math as calc
import matplotlib.gridspec as gridder
import matplotlib.patches as patch #For Ellipse package
try:
    import astropy.io.fits as fitter
except ImportError:
    import pyfits as fitter
except ImportError:
    raise ImportError("Doesn't seem like you have a recognized module installed that will read in .fits files...  This file recognizes either astropy.io.fits or pyfits as such modules.  Please either install one of these modules or change the import settings within the share_modAstro.py script to import a fits file reader that you use.")
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



##FUNCTION: extract_fits
##PURPOSE: Extracts information from a continuum or channel emission .fits file.
##INPUTS:
##    - filename [str; required]: Filename (including its path from...
##       ...current directory); should be a .fits file
##    - filetype ["cont" or "chan"; required]: "cont" for continuum .fits...
##       ...or "chan" for channel map .fits
##    - restfreq [int/float; default=None]: If "chan" .fits files are in frequency...
##       ...units, then restfreq must be set to the line's rest frequency
##NOTES:
def extract_fits(filename, restfreq=None):
    #Below Section: Extracts info from the file
    openfit = fitter.open(filename)
    fitdict = {} #To hold extracted info
    
    #For emission and associated lengths
    emmatr = openfit[0].data[0]
    fitdict["emmatr"] = emmatr
    ralen = len(emmatr[0,0]) #Length of RA
    declen = len(emmatr[0]) #Length of DEC
    nchan = len(emmatr) #Number of channels
    fitdict["nchan"] = nchan
    fitdict["ralen"] = ralen
    fitdict["declen"] = declen
    
    #For emission-specific info, such as array and beam dimensions
    #For RA and DEC
    ra0 = openfit[0].header["CRVAL1"]*pi/180.0 #[deg] -> [rad]
    rawidth = openfit[0].header["CDELT1"]*pi/180.0 #[deg] -> [rad]
    dec0 = openfit[0].header["CRVAL2"]*pi/180.0 #[deg] -> [rad]
    decwidth = openfit[0].header["CDELT2"]*pi/180.0 #[deg] -> [rad]
    fitdict["raarr"] = np.array([(ra0 + (ei*rawidth))
                    for ei in range(0, ralen)])
    fitdict["rawidth"] = rawidth
    fitdict["decarr"] = np.array([(dec0 + (ei*decwidth))
                    for ei in range(0, declen)])
    fitdict["decwidth"] = decwidth
    
    #For beam
    try: #If same beam for each channel
        fitdict["bmaj"] = openfit[0].header["BMAJ"]*pi/180.0 #[deg] -> [rad], full axis
        fitdict["bmin"] = openfit[0].header["BMIN"]*pi/180.0 #[deg] -> [rad], full axis
        fitdict["bpa"] = openfit[0].header["BPA"]*pi/180.0 #[deg] -> [rad]
    except KeyError: #If different beams for each channel
        #Make sure beam data is where it is expected to be
        if ((openfit[1].header["TTYPE1"] != "BMAJ")
                or (openfit[1].header["TTYPE2"] != "BMIN")
                or (openfit[1].header["TTYPE3"] != "BPA")
                or (openfit[1].header["TUNIT1"] != "arcsec")
                or (openfit[1].header["TUNIT2"] != "arcsec")
                or (openfit[1].header["TUNIT3"] != "deg")):
            raise ValueError("Whoa! Beam data in weird format! Contact"
                                +" your code provider!")
        #Extract beam data across channels
        tempnc = openfit[1].header["NCHAN"]
        tempbmajs = [openfit[1].data[bi][0] for bi in range(0, tempnc)]
        tempbmins = [openfit[1].data[bi][1] for bi in range(0, tempnc)]
        tempbpas = [openfit[1].data[bi][2] for bi in range(0, tempnc)]
        #Determine median beam values
        tempbmajmean = np.mean(tempbmajs)*1.0 #arcsec
        tempbminmean = np.mean(tempbmins)*1.0 #arcsec
        tempbpamean = np.mean(tempbpas)*1.0 #deg
        #Make sure beam variation is not too huge across channels
        bcheck = 0.02
        if (((np.std(tempbmajs)/tempbmajmean) > bcheck)
                or ((np.std(tempbmins)/tempbminmean) > bcheck)
                or ((np.std(tempbpas)/tempbpamean) > bcheck)):
            print(np.std(tempbmajs)/tempbmajmean)
            print(np.std(tempbmins)/tempbminmean)
            print(np.std(tempbpas)/tempbpamean)
            raise ValueError("Whoa! >{0}% diff. in beam over channels!"
                    .format((bcheck*100)))
        #Record median beam values
        fitdict["bmaj"] = tempbmajmean/3600.0*pi/180 #arcsec->rad, full axis
        fitdict["bmin"] = tempbminmean/3600.0*pi/180 #arcsec->rad, full axis
        fitdict["bpa"] = tempbpamean*pi/180.0 #[deg] -> [rad]
    
    #For velocity
    vel0 = openfit[0].header["CRVAL3"] #m/s
    velwidth = openfit[0].header["CDELT3"] #m/s
    velarr = np.array([(vel0 + (ei*velwidth)) for ei in range(0, nchan)])
    if openfit[0].header["CUNIT3"] == "m/s":
        fitdict["velwidth"] = velwidth #m/s
        fitdict["velarr"] = velarr #m/s
    elif openfit[0].header["CUNIT3"] == "Hz": #Converts from freq.
        actvelarr = conv_freqtovel(freq=velarr, restfreq=restfreq)
        fitdict["velarr"] = actvelarr
        fitdict["velwidth"] = actvelarr[1] - actvelarr[0]
    
    #Below closes fits file out of politeness
    openfit.close()
    #Below Section: Returns the extracted information in dictionary form
    return fitdict #Degrees and m/s as applicable
#



##FUNCTION: plot_channels
##PURPOSE: Generates a series of channel maps.
##INPUTS:
##    Channel Parameters
##    - chanall_list [list of 2D arrays; required]: List of all channels (including...
##       ...any the user doesn't want to plot)
##    - maskall_list [list of 2D arrays; required]: List of all channel masks...
##       ...(including any the user doesn't want to plot); if None, will not plot masks
##    - whichchan_list [list of indices; required]: List of channel indices...
##       ...(and masks, if masks are given) that the user would like to plot
##    - velall_arr [1D array; required]: Array of velocities for each channel...
##       ...(including any channels the user doesn't want to plot)
##    - x_arr [1D array; required]: Array of x-axis values for the plot
##    - y_arr [1D array; required]: Array of y-axis values for the plot
##    - vmin [int/float; default=None]: Minimum value for the plot's colorbar;...
##       ...if None, will take the minimum value across the plotting range...
##       ...of the channel map to be vmin
##    - vmax [int/float; default=None]: Maximum value for the plot's colorbar;...
##       ...if None, will take the maximum value across the plotting range...
##       ...of the channel map to be vmax
##
##    General Plot Parameters
##    - plotscale [float that is > 0; default=10]: Scales the size of the overall...
##       ...channel map
##    - spec [float that is > 2; default=2]: Scales how the size of the channel map...
##       ...increases with the number of rows
##    - ncol [int; default=7]: Number of columns in the figure
##    - cmap [matplotlib colorbar instance; required]: Colormap to use for the colorbar
##    - xlim [2-number list; default=[-5,5]]: The x-axis range to be plotted
##    - ylim [2-number list; default=[-5,5]]: The y-axis range to be plotted
##
##    Beam Parameters
##    - plotbeam [bool; default=False]: If True, will plot a beam on the...
##       ...bottom-leftmost channel
##    - beamx [float in interval [0, 1]; default=0.80]: Scaled x-axis location...
##       ...of velocity in-graph label for each channel...
##       ...where 0 will put the label all the way to the left of the channel...
##       ...and 1 will put the label all the way to the right
##    - beamy [float in interval [0, 1]; default=0.80]: Scaled y-axis location...
##       ...of velocity in-graph label for each channel...
##       ...where 0 will put the label all the way to the bottom of the channel...
##       ...and 1 will put the label all the way to the top
##    - bmaj [int/float; required]: Full (not half) major axis of the beam to plot;...
##       ...can be set to None if plotbeam is False
##    - bmin [int/float; required]: Full (not half) minor axis of the beam to plot;...
##       ...can be set to None if plotbeam is False
##    - bpa [int/float [radians]; required]: Angle in [radians] of the beam to plot;...
##       ...can be set to None if plotbeam is False
##
##    Velocity Label Parameters
##    - velx [float in interval [0, 1]; default=0.10]: Scaled x-axis location of...
##       ...velocity in-graph label for each channel...
##       ...where 0 will put the label all the way to the left of the channel...
##       ...and 1 will put the label all the way to the right
##    - vely [float in interval [0, 1]; default=0.85]: Scaled y-axis location...
##       ...of velocity in-graph label for each channel...
##       ...where 0 will put the label all the way to the bottom of the channel...
##       ...and 1 will put the label all the way to the top
##    - velalpha [float in interval [0, 1]; default=0.6]: alpha value...
##       ...for velocity in-graph label
##    - velboxcolor [str; default="white"]: color for the box of...
##       ...the velocity in-graph label
##    - velboxedge [str; default="white"]: color for the box edge of...
##       ...the velocity in-graph label
##
##    Mask Contour Parameters
##    - cstyle [str; default="--"]: Line style for the mask contours
##    - cwidth [float; default=2]: Line width for the mask contours
##    - ccolor [str; default="black"]: Line color for the mask contours
##    - calpha [float in interval [0, 1]; default=0.7]: Alpha for the mask contours
##
##    Colorbar, Axis Label, and Font Parameters
##    - docbar [bool; default=True]: If True, will show the colorbar of this plot
##    - cbarlabel [str; default=""]: Colorbar title
##    - xlabel [str; default=""]: X-axis label
##    - ylabel [str; default=""]: Y-axis label
##    - ticksize [int/float; default=14]: Font size for plot ticks
##    - textsize [int/float; default=14]: Font size for plot axis labels and suptitle
##    - titlesize [int/float; default=14]: Font size for plot title
##    - tickwidth [int/float; default=3] #Width of ticks
##    - tickheight [int/float; default=4] #Height of ticks
##    
##    User-Defined Box Parameters
##    - boxtexts [list of str; default=None]: Strings that can be plotted as...
##       ...in-graph labels (useful, for example, for plotting a source's name);...
##       ...if None, will not plot any such in-graph labels
##    - boxind [int; default=0]: Index of the channel where user would like to...
##       ...plot any given user-defined labels
##    - boxxs [list of floats in interval [0, 1]; default=0.05]: Scaled x-axis...
##       ...location of any given user-defined in-graph labels...
##       ...where 0 will put the label all the way to the left of the channel...
##       ...and 1 will put the label all the way to the right
##    - boxys [list of floats in interval [0, 1]; default=0.90]: Scaled y-axis...
##       ...location of any given user-defined in-graph labels...
##       ...where 0 will put the label all the way to the bottom of the channel...
##       ...and 1 will put the label all the way to the top
##NOTES:
##    - 
def plot_channels(chanall_list, maskall_list, whichchan_list, x_arr, y_arr,
            bmaj, bmin, bpa, velall_arr, cmap, velx=0.10, vely=0.85, ncol=7,
            velalpha=0.6, velboxcolor="white", velboxedge="white",
            vmin=0, vmax=None, plotscale=7, spec=2,
            cstyle="--", cwidth=2, ccolor="black", calpha=0.7,
            xlabel="", ylabel="", tickwidth=3, tickheight=5,
            boxtexts=None, boxxs=[0.05], boxys=[0.90], boxind=0,
            docbar=False, cbarlabel="", xlim=[-5,5], ylim=[-5,5],
            beamx=0.80, beamy=0.80, ticksize=14, textsize=14, titlesize=14,
            plotbeam=False):
    ##Below Section: Extracts desired channels, velocities, and masks for the disk
    if whichchan_list is not None: #If particular channels desired
        chanplot_list = chanall_list[whichchan_list].copy() #Desired channels
        velplot_arr = velall_arr[whichchan_list].copy() #Desired velocities
    else: #Extracts all channels
        chanplot_list = chanall_list.copy() #Desired channels
        velplot_arr = velall_arr.copy() #Desired velocities
    #Below extracts masks, if given
    maskplot_list = None
    if maskall_list is not None and whichchan_list is not None:
        maskplot_list = maskall_list[whichchan_list].astype(float) #Desired masks
    elif maskall_list is not None:
        maskplot_list = maskall_list.astype(float) #Desired masks, all channels
    
    #Below determines extent of rows across channel map, given the number of columns
    nrow = ((len(velplot_arr) - 1)//ncol) + 1
    
    
    ##Below Section: Determines where in channels to measure vmin and vmax...
    #... if vmin and/or vmax not given
    if (vmin is None) or (vmax is None):
        #Below determines x-axis range for measuring vmin and vmax
        #If no x-axis plotting range (xlim) given, will take vmin and/or vmax...
        #...over entire x-axis span for the channels
        if xlim is None:
            lhere = 0 #Starts at leftmost point of channels
            rhere = len(x_arr) - 1 #Extends to rightmost point of channels
        #Otherwise, if xlim is given, will take vmin and/or vmax over the...
        #...given x-axis span for the channels
        else:
            lhere = int(np.abs(x_arr - xlim[0]).argmin())
            rhere = int(np.abs(x_arr - xlim[1]).argmin())
        if rhere < lhere: #If x_arr has decreasing values (like descending RA)
            rtemp = rhere
            rhere = lhere
            lhere = rtemp
        
        #Below determines y-axis range for measuring vmin and vmax
        #If no y-axis plotting range (ylim) given, will take vmin and/or vmax...
        #...over entire y-axis span for the channels
        if ylim is None:
            there = 0 #Starts at topmost point of channels
            bhere = len(y_arr) - 1 #Extends to bottommost of channels
        #Otherwise, if ylim is given, will take vmin and/or vmax over the...
        #...given y-axis span for the channels
        else:
            there = int(np.abs(y_arr - ylim[0]).argmin())
            bhere = int(np.abs(y_arr - ylim[1]).argmin())
        if bhere < there: #If y_arr has decreasing values (like descending DEC)
            btemp = bhere
            bhere = there
            there = btemp
    
    #Below calculates vmin and vmax values for overall channel map, if not given
    if vmin is None: #Takes min. emission across channel map to be vmin
        #NOTE: Measures vmin within xlim and ylim, if xlim and/or ylim given
        vmin = calc.floor(min([chere[there:bhere+1,lhere:rhere+1].min()
                    for chere in chanplot_list]))
    if vmax is None: #Takes max. emission across channel map to be vmax
        #NOTE: Measures vmax within xlim and ylim, if xlim and/or ylim given
        vmax = calc.ceil(max([chere[there:bhere+1,lhere:rhere+1].max()
                    for chere in chanplot_list]))
    
    
    ##Below Section: Generates subplots on a snazzy grid
    #Below determines parameters of grid
    #Left-right parameters
    sublength = 0.3 #Length of subplot side
    lrmargin = 0.22 #Margin at each left and right side of overall figure
    lrtot = ((sublength*ncol) + 2.0*lrmargin) #Total width of overall figure
    cbarmargin = (lrtot - lrmargin)/1.0/lrtot #Equal to (1 - <margin of colorbar>)
    #Top-bottom parameters
    subwidth = sublength #Equal-sized plots (length = width)
    tbmargin = 0.12 #Margin at each top and bottom side of overall figure
    tbtot = ((subwidth*nrow) + (2.0*tbmargin)) #Total height of full figure    
    
    #Below determines the grid locations themselves for the subplots
    gridlist = [] #List to hold grid locations
    for vi in range(0, len(velplot_arr)): #Iterates through needed subplots
        #Below calculates current column and row location
        ri = vi // ncol #Row index
        ci = vi % ncol #Column index
    
        #Below calculates location of current subplot within the grid
        tophere = (1 - (tbmargin/tbtot) - (ri*subwidth/tbtot)) #Top location
        bothere = (1 - (tbmargin/tbtot) - ((ri+1)*subwidth/tbtot)) #Bottom loc.
        lefthere = ((lrmargin/lrtot) + (ci*sublength/lrtot)) #Left loc.
        righthere = ((lrmargin/lrtot) + ((ci+1)*sublength/lrtot)) #Right loc.
        
        #Below records current subplot grid location
        gridlist.append(gridder.GridSpec(1,1))
        gridlist[-1].update(top=tophere, bottom=bothere,
                    left=lefthere, right=righthere)
    
    
    #Below Section: Plots the subplots at determined grid locations
    #Below generates overall plot and scales the plot uniformly
    plotxlen = (plotscale+(nrow*spec))*(lrtot/1.0/tbtot) #Size of overall plot x-axis
    plotylen = (plotscale+(nrow*spec)) #Size of overall plot y-axis
    fig = plt.figure(figsize=(plotxlen, plotylen)) #Overall figure
    
    #Below iterates through and plots each desired channel of the channel map
    for bi in range(0, len(velplot_arr)): #Iterate through channels
        #Below sets up the base subplot and its ticks for this current channel
        plothere = plt.subplot(gridlist[bi][0,0], aspect=1)
        plothere.tick_params(width=tickwidth, size=tickheight,
                labelsize=ticksize, direction="in") #Axis tick sizes
        plt.locator_params(nbins=6) #Number of ticks
        tickdict = calc_cbarticks(vmin=vmin, vmax=vmax) #Ticks for colorbar
        
        #Below plots current channel
        maphere = plothere.contourf(x_arr, y_arr,
                chanplot_list[bi], tickdict["tickhids"],
                vmin=vmin, vmax=vmax,
                cmap=cmap)
        
        #Below plots current mask contour over current channel, if desired
        if maskplot_list is not None:
            plothere.contour(x_arr, y_arr,
                maskplot_list[bi],
                [-1,0,1,2], alpha=calpha, linewidths=cwidth,
                colors=ccolor, linestyles=cstyle)
        
        #Below puts in an in-graph label of velocity
        plt.text(velx, vely, "{:.2f}km/s".format(velplot_arr[bi]),
            bbox=dict(facecolor=velboxcolor, alpha=velalpha,
                            edgecolor=velboxedge),
            fontsize=textsize, transform=plothere.transAxes)
        
        #Below handles this channel's ticks and tick labels; removes if need be
        if bi == (ncol*(nrow - 1)): #Bottom-most channel keeps its tick labels
            plothere.set_xlabel(xlabel, fontsize=titlesize)
            plothere.set_ylabel(ylabel, fontsize=titlesize)
        else: #Otherwises, strips away tick labels
            plothere.tick_params(labelbottom=False, labelleft=False)
        
        #Below zooms in on plot, if so desired
        if xlim is not None: #For x-axis range
            plothere.set_xlim(xlim)
        if ylim is not None: #For y-axis range
            plothere.set_ylim(ylim)
        plothere.set_xlim(maphere.ax.get_xlim()[::-1]) #Makes x-axis descend
        
        #Below adds in any in-graph labels to this channel, if so desired
        if (boxtexts is not None) and (bi == boxind):
            for ci in range(0, len(boxtexts)):
                plothere.text(boxxs[ci], boxys[ci],
                    boxtexts[ci],
                    bbox=dict(facecolor="white", alpha=0.5,
                            edgecolor="white"),
                    fontsize=textsize,
                    transform=plothere.transAxes)
        
        #Below plots the beam, if so desired
        if (bi == (ncol*(nrow - 1))) and plotbeam: #Plots on bottommost-left
            yspan = plothere.get_ylim() #ybounds of this subplot
            xspan = plothere.get_xlim() #xbounds of this subplot
            #Below scales beam x-y location on plot
            bloc = [(((xspan[1] - xspan[0])*beamx) + xspan[0]),
                (((yspan[1] - yspan[0])*beamy) + yspan[0])]

            #Below plots the beam
            ellhere = patch.Ellipse(xy=bloc, width=bmin*180.0/pi*3600,
                    height=bmaj*180.0/pi*3600,
                    angle=bpa*-1*180.0/pi, zorder=300)
            ellhere.set_facecolor("gray")
            ellhere.set_edgecolor("black")
            plothere.add_artist(ellhere)
    
    
    ##Below Section: Generates a colorbar, if so desired
    if docbar:
        #Below prepares ticks for the colorbar
        tickdict = calc_cbarticks(vmin=vmin, vmax=vmax) #Prepare colorbar ticks
        plottickvals = tickdict["tickhids"] #Plot colors
        cbartickvals = tickdict["tickshows"] #Colorbar tick values
        cbarticknames = tickdict["ticknames"] #Colorbar tick labels
        
        #Below generates the colorbar
        if len(velplot_arr) >= ncol: #If more channels than column size
            cbarright = ((lrmargin/lrtot) + (ncol*sublength/lrtot))
        else: #If fewer channels than column size
            cbarright = (((lrmargin/lrtot)
                    + ((len(velplot_arr))*sublength/lrtot)))
        cbar_ax = fig.add_axes([cbarright, (tbtot-tbmargin-subwidth)/tbtot,
                    (1-cbarmargin)/4.0,
                    subwidth/1.0/tbtot]) #left, bot, width, height
        cbar = plt.colorbar(maphere, ticks=cbartickvals,
                        cax=cbar_ax) #Colorbar itself
        cbar.ax.set_yticklabels(cbarticknames) #Labels colorbar ticks
        cbar.set_label(cbarlabel, fontsize=titlesize,
                        rotation=270, labelpad=20) #Cbar. title
        cbar.ax.tick_params(labelsize=ticksize) #Cbar. font size
#



##FUNCTION: calc_cbarticks
##PURPOSE: Determines colorbar ticks, as well as nice spacing for colorbar tick labels.
##INPUTS:
##    - vmin [int/float]: Minimum value to show on the plot's colorbar
##    - vmax [int/float]: Maximum value to show on the plot's colorbar
##NOTES:
##    - 
def calc_cbarticks(vmin, vmax):
    #Below calculates total tick range
    numticks = int(calc.ceil(vmax - vmin))
    if numticks < 5: #If weird (such as 0-1) tick range given
        numticks = 10
    
    #Below determines interval of ticks
    if numticks < 15:
        tickinv = 2
    elif numticks < 40:
        tickinv = 5
    elif numticks < 80:
        tickinv = 10
    elif numticks < 200:
        tickinv = 25
    elif numticks < 500:
        tickinv = 50
    else:
        tickinv = 100
    
    #Below determines tick values and labels
    tickhids = np.linspace(vmin,
                calc.ceil(vmax)+1, 20) #Values per tick
    tickshows = np.arange((calc.ceil(vmin/1.0/tickinv)*tickinv),
                calc.ceil(vmax)+1, tickinv).astype(int)
    ticknames = tickshows.astype(str) #Tick labels
    
    #Below returns the determined tick characteristics
    return {"tickinv":tickinv, "tickhids":tickhids,
                "tickshows":tickshows, "ticknames":ticknames}
#



##FUNCTION: calc_beamarea
##PURPOSE: Calculates the beam area (in pixels, so to speak) for a given beam.
##INPUTS:
##    - bmaj [float; required]: Full major axis of beam [rad]
##    - bmin [float; required]: Full minor axis of beam [rad]
##    - rawidth [float; required]: Delta-R.A. of R.A. axis [rad]
##    - decwidth [float; required]: Delta-Decl. of Decl. axis [rad]
##NOTES:
def calc_beamarea(bmaj, bmin, rawidth, decwidth):
    #Calculate beam area
    brarea_raw = pi/1.0/np.log(2)*(bmaj/2.0*bmin/2.0) #Beam area [rad^2]
    pixrarea_raw = np.abs(rawidth*decwidth) #Pixel area [rad^2]
    beamarea = brarea_raw/1.0/pixrarea_raw #Desired beam area [pix]
    #Return the beam area
    return beamarea
#




