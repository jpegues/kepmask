###FILE: kepmask.py
###PURPOSE:
###github.com/jpegues/kepmask


##Below Section: Imports necessary functions
from . import _utilsAstro as astmod
from . import _utilsKep as kepmod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch #For Ellipse package
plt.close()
pi = np.pi

class KepMask():
    def __init__(self, **kwargs):
        """
        METHOD: __init__
        PURPOSE: Initialize an instance of the KepMask class.
        INPUTS: At minimum, requires the following input parameters:
          - midx [index]: Location along the x-axis (R.A. axis) of the midpoint.
          - midy [index]: Location along the y-axis (Dec. axis) of the midpoint.
          - mstar [kg]: Mass of the host star.
          - PA [radian]: Position angle.
          - inc [radian]: Inclination angle.
          - dist [m]: Distance to the source.
          - vsys [m/s]: Systemic velocity of the source.
        OUTPUTS:
          - Initialized instance of the KepMask class.
        NOTES:
          - N/A
        """
        #Initialize a dictionary to hold input parameters.
        self._valdict = {}
        self._imdict = {} #Placeholder for future image cube data.
        self._maskdict = {} #Placeholder for future mask generation.

        #Fill the dictionary with all given values.
        for key, val in kwargs.items():
            self._valdict[key] = val

        #Record the names of all mask parameters that will ever be used.
        self._paramkeys = np.sort(["midx", "midy", "mstar", "PA", "inc",
                            "dist", "vsys"])

        #Record the names of all image cube data that will ever be used.
        self._imkeys = np.sort(["ralen", "declen", "velarr", "bmaj", "bmin",
                    "bpa", "rawidth", "decwidth", "velwidth", "emmatr",
                    "raarr", "decarr"])

        #Record the names of all masked products that will ever be generated.
        self._prod_list = ["masks", "spectrum", "mom0",
                                "channels", "intflux"]

        #Record the names of all plots that will ever be generated.
        self._plot_list = ["spectrum", "mom0", "channels"]
    #

    def get_parameter(self, param, _inner=False, _keyset=None):
        """
        METHOD: get_parameter
        PURPOSE: Return the value of the given parameter (if it exists).
          If the parameter does not exist, then a helpful error will be thrown
          that lists all of the available parameters.
        INPUTS:
          - param [str]: The name of the desired parameter.
          - _inner [bool]: NOT for the user to use!  This is an internal boolean
              for determining why this function is being called.
          - __keyset [list, None]: NOT for the user to use!  This is an internal
              list to give all parameters needed to run a method.
        OUTPUTS:
          - Value of the desired parameter, if it exists, OR an error message.
        NOTES:
          - N/A
        """
        #The following protocol is called when a user uses this method.
        if not _inner:
            #Attempt to fetch the desired parameter from input storage.
            try:
                return self._valdict[param]
            #
            #If that doesn't work, attempt to fetch from image storage.
            except KeyError:
                try:
                    return self._imdict[param]
            #If the parameter does not exist, print available parameters.
                except KeyError:
                    errstr = ("Looks like the parameter you requested "
                        +"({0}) doesn't ".format(param)
                        +"exist.  Here are all of the parameters you can "
                        +"actually access via the get_parameter() method: "
                        +"{0}".format((np.sort([key for key in self._valdict]
                                        +[key for key in self._imdict]))))
                raise KeyError(errstr)

        #The following protocol is for internal code use only (NOT users!).
        else:
            #Attempt to fetch the desired parameter from storage.
            try:
                return self._valdict[param]
            #If the parameter does not exist, print all required parameters.
            except KeyError:
                errstr = ("Whoa!  Looks like you're missing a parameter "
                    +"value for ({0}).  ".format(param)
                    +"In order to run this method, you need to set the "
                    +"following parameters: "+"{0}".format(_keyset)
                    +" These can be passed in when you create a class "
                    +"instance of KepMask.  OR, you can set missing "
                    +"parameter values using the "
                    +".set_parameter(param, value) method.")
                raise KeyError(errstr)
    #

    def _get_imdata(self, param):
        """
        METHOD: _get_imdata
        PURPOSE: NOT meant for user use!  This is a background function for
          looking up image cube data.
        """
        #Attempt to fetch the desired parameter from image cube storage.
        try:
            return self._imdict[param]
        #If the parameter does not exist, ask the reader to extract the data.
        except KeyError:
            errstr = ("Looks like you haven't loaded any image cube data "
                    +"yet.  Please use the "
                    +".extract(fitsname, restfreq=None) "
                    +"method to do so first, BEFORE trying to generate "
                    +"masks or generate/plot any products!")
            raise KeyError(errstr)
    #

    def _get_maskdata(self, param):
        """
        METHOD: _get_maskdata
        PURPOSE: NOT meant for user use!  This is a background function for
          looking up masks and masked products.
        """
        #Attempt to fetch the desired parameter from mask storage.
        try:
            return self._maskdict[param]
        #If the parameter does not exist, ask the reader to generate masks.
        except KeyError:
            errstr = ("Looks like you haven't generated any masks yet.  "
                    +"Please use the "
                    +".generate() "
                    +"method to do so first, BEFORE trying to generate or "
                    +"plot any products!")
            raise KeyError(errstr)
    #

    def set_parameter(self, param, value):
        """
        METHOD: set_parameter
        PURPOSE: Set the value of the given parameter.
          If the parameter did not previously exist, then it will be created.
          If the parameter DID previously exist, then its previous value will
          be overwritten.
        INPUTS:
          - param [str]: The name of the given parameter.
          - value [anything]: The value of the given parameter.
        OUTPUTS:
          - N/A
        NOTES:
          - N/A
        """
        #Set the value of the given parameter.
        self._valdict[param] = value
    #

    def extract(self, fitsname, restfreq=None):
        """
        METHOD: extract
        PURPOSE: Load in the data from a .fits file image cube, containing
          imaged channel data.
        INPUTS:
          - fitsname [str filename]: filepath+filename of the .fits image cube.
          - restfreq [(optional) float, Hz]: Rest frequency of molecular line.
              Only necessary if the .fits file's channel units are frequencies
              [Hz] instead of velocities [m/s].
        OUTPUTS:
          - N/A
        NOTES:
          - The .fits file should have 4 dimensions:
            - 0th dimension = Polarization (ASSUMED TO BE SINGLE POLARIZATION, size=1)
            - 1th dimension = Channels
            - 2th dimension = Declination
            - 3th dimension = R.A.
        """
        #Call on a utils method to extract the .fits information.
        self._imdict = astmod.extract_fits(
                                filename=fitsname, restfreq=restfreq)
    #

    def generate(self, whichchans=None, hypfreqs=None, showtests=False,
                    V0_ms=300.0, R0_AU=100.0, q0=0.3, mask_override=None,
                    mask_Rmax=None, beamfactor=3.0, beamcut=0.03,
                    do_generate_products=True):
        """
        METHOD: generate
        PURPOSE: Generate a Keplerian mask for the previously-stored image cube.
        INPUTS:
          - beamcut [float, default=0.03]: Factor that controls beam convolution
              approximation.  Provides cutoff for the normalized approximation.
              See NOTES below.
          - beamfactor [float, default=3.0]: Factor that controls beam
              convolution approximation.  Ensures convolution is more circular.
              See NOTES below.
          - hypfreqs [None OR list of floats [Hz], default=None]: List of
              hyperfine frequencies [Hz] for generating Keplerian masks for
              molecular lines with hyperfine structure.  The main hyperfine
              transition should be first in the list.
              If hypfreqs=None, then no hyperfine masks will be generated.
          - mask_override [2D bool array OR None, default=None]: Regions in
              the channels where Keplerian masks are allowed to take place.
              For instance, if you are looking to extract emission within
              only a certain annulus, then mask_override could be set to a
              2D bool array that highlights only that annulus region.
          - mask_Rmax [float [m] OR None, default=None]: Radial extent of the
              masks in meters.  If None, no such boundary is applied.
          - R0_AU [float [AU], default=100.0]: Characteristic radius (IN AU!)
              for the broadening scheme described by Yen+2016.  See NOTES below.
          - q0 [float, default=0.3]: Power law index for the broadening scheme
              described by Yen+2016.  See NOTES below.
          - showtests [bool, default=False]: If True, will plot the velocity
              fields calculated during the generation of the Keplerian masks.
          - V0_ms [float [m/s], default=300.0]: Prefactor for the broadening
              scheme described by Yen+2016.  See NOTES below.
          - whichchans [None OR list of indices, default=None]: List of indices
              corresponding to the desired channels to have masks.
        OUTPUTS:
          - N/A
        NOTES:
          - We advise that the beamfactor and beamcut values be left at their
              defaults, as they are currently scaled so that a point source would
              be "smeared" to the size of the given beam.
          - Note also that the beam convolution is only approximated, via
              a Gaussian filter, and uses a circle of size max([bmin, bmaj]).
              and bmin).
          - We use the velocity broadening scheme of Yen+2016, which looks like:
              Delta_V = 4 * V0*(R/R0)^(-q0)
              This broadening (Delta_V) is added in quadrature to the velocity
              width of the channels.
          - If both mask_override and mask_Rmax are not None, then the value
              of mask_Rmax will be used.
        """
        #Set up a dictionary to hold masks and related products.
        maskdict = {}

        #Gather mask parameters, which should have been previously loaded.
        paramdict = {} #Temporary dictionary to hold looked-up parameters.
        for key in self._paramkeys:
            paramdict[key] = self.get_parameter(param=key, _inner=True,
                                            _keyset=self._paramkeys)

        #Gather image data, which should have been previously extracted.
        imdict = {} #Temporary dictionary to hold looked-up image cube data.
        for key in self._imkeys:
            imdict[key] = self._get_imdata(param=key)
        emmatr = imdict["emmatr"] #Record channel data under a variable name
        maskdict["channels"] = emmatr #Record channel data in mask dictionary

        #If no specific channels requested, then use all channels.
        if whichchans is None:
            whichchans = np.arange(0, len(emmatr), 1) #All channels included
        maskdict["whichchans"] = whichchans

        #Determine the max. value in [bmaj, bmin]
        #NOTE: We take the max. since the beam convolution approximation uses
        #       the circular approximation of the beam anyway.  Better to
        #       have masks that are too big than masks that are too small.
        bmax = max([imdict["bmaj"], imdict["bmin"]])

        #Determine R.A. and Decl. axes, offset from midpoint.
        radeltarr = imdict["raarr"] - imdict["raarr"][paramdict["midx"]]
        decdeltarr = imdict["decarr"] - imdict["decarr"][paramdict["midy"]]

        #Call on a utils method to calculate the Keplerian masks.
        maskall = kepmod.calc_Kepvelmask(
            xlen=imdict["ralen"], ylen=imdict["declen"], bmin=bmax,
            vel_arr=imdict["velarr"], bmaj=bmax, bpa=imdict["bpa"],
            midx=paramdict["midx"], midy=paramdict["midy"],
            rawidth=imdict["rawidth"], decwidth=imdict["decwidth"],
            velwidth=imdict["velwidth"], mstar=paramdict["mstar"],
            posang=paramdict["PA"], incang=paramdict["inc"],
            dist=paramdict["dist"], sysvel=paramdict["vsys"],
            radeltarr=radeltarr, decdeltarr=decdeltarr,
            beamfactor=beamfactor, beamcut=beamcut, freqlist=hypfreqs,
            broadyen_pre0=V0_ms, broadyen_r0=R0_AU, broadyen_qval=q0,
            showtests=showtests, emsummask=mask_override, rmax=mask_Rmax,
            whichchans=whichchans)
        #Nullify masks in undesired channels
        for ai in range(0, len(maskall)):
            if ai not in whichchans:
                maskall[ai] = False

        #Go ahead and calculate relevant masked products, since we're here.
        #For the masks themselves:
        maskdict["masks"] = maskall

        #Generate masked products, if so requested
        if do_generate_products:
            #For the spectrum (array of summed masked emission):
            #Calculate and record the *raw* spectrum by summing desired channels
            #specarr_raw = np.array([emmatr[ai][maskall[ai]].sum()
            #                        for ai in range(0, len(maskall))]) #[Jy/beam]
            specarr_raw = np.nansum(np.where(maskall, emmatr, 0), axis=(1,2))#Jy/beam
            #Calculate the beam area [pix]
            beamarea = astmod.calc_beamarea(bmaj=imdict["bmaj"],
                            bmin=imdict["bmin"], rawidth=imdict["rawidth"],
                            decwidth=imdict["decwidth"])
            #Convert the raw spectrum from [Jy/beam] to [Jy]
            specarr = specarr_raw/1.0/beamarea #[Jy/beam] -> [Jy]
            #Record the converted spectrum
            maskdict["spectrum"] = specarr

            #For the velocity-integrated flux:
            maskdict["intflux"] = specarr.sum()*imdict["velwidth"]

            #For the mom0 (pixel-by-pixel sum of masked emission):
            kepmom0 = np.zeros(shape=(imdict["declen"],
                                        imdict["ralen"])) #Initialize mom0 matrix
            #Iterate through desired channels with emission
            for ihere in whichchans: #Iterate through desired channels
                kepmatrhere = emmatr[ihere].copy() #Current channel
                kepmatrhere[~maskall[ihere]] = 0.0 #Remove data beyond mask
                kepmom0 += kepmatrhere #Add channel to overall mom0
            #Store the complete mom0
            maskdict["mom0"] = kepmom0*imdict["velwidth"]

        #Store the masks and related products.
        self._maskdict = maskdict
        #
    #

    def get_product(self, prod):
        """
        METHOD: get_product
        PURPOSE: Return the desired image product (e.g., spectrum or mom0).
        INPUTS:
          - prod [str]: The name of the desired product.
        OUTPUTS:
          - The desired product (if it exists), or a helpful error that lists
              the available types of products.
        NOTES:
          - The available products are:
            - "masks": The generated Keplerian masks.
            - "channels": The (unmasked) channel map.
            - "mom0": The velocity-integrated emission map [flux units / beam * m/s].
            - "spectrum": The computed spectrum [flux units].
            - "intflux": The velocity-integrated flux [flux units * m/s].
        """
        #Raise an error if the desired product is not allowed.
        if prod not in self._prod_list:
            errstr = ("Whoa! Looks like the product you requested ({0}) "
                        .format(prod)
                        +"is not a valid product.  Valid products are: "
                        +"{0}".format(self._prod_list))
            raise KeyError(errstr)

        #If the requested product are channels, fetch from image storage.
        if prod == "channels":
            return self._get_imdata("emmatr")

        #Otherwise, attempt to fetch the desired product from mask storage.
        try:
            return self._maskdict[prod]
        #If the parameter does not exist, ask the reader to generate masks.
        except KeyError:
            errstr = ("Looks like you haven't generated any masks and/or their "
                    +"products yet.  Please use the "
                    +".generate(do_generate_products) "
                    +"method to do so first, BEFORE trying to get masks "
                    +"or any masked products!\n"
                    +"In particular, do_generate_products must be set to True "
                    +"in order to generate any emission products (spectra, etc.)")
            raise KeyError(errstr)
        #
    #

    def plot_product(self, prod, dosave=False, savename="testing.png",
                        figsize=(8,8), title="", ncol=7,
                        xlim=[-3.0, 3.0], ylim=[-3.0, 3.0],
                        color="black", cmap=plt.cm.bone_r,
                        vmin=0.0, vmax=None, vrange_kms=None,
                        textsize=14, titlesize=16,
                        linewidth=3.0, linestyle="-", alpha=0.5,
                        beamx=0.85, beamy=0.15):
        """
        METHOD: plot_product
        PURPOSE: Return a plot of the desired image product (e.g., spectrum
          or mom0).
        INPUTS:
          - alpha [float, default=0.5]: Line transparency.  Valid values are from
              0 (transparent) to 1 (opaque).  Used only for spectra.
          - beamx [float, default=0.8]: Relative x-axis location of beam (scaled
              from 0 to 1).  Used only for mom0s and channel maps.
          - beamy [float, default=0.8]: Relative y-axis location of beam (scaled
              from 0 to 1).  Used only for mom0s and channel maps.
          - cmap [<matplotlib colormap>, default=plt.cm.bone_r]: Colormap.  Used
              only for mom0s and channel maps.
          - color [str, default="black"]: Line color.  All matplotlib values
              are valid.  Used only for spectra.
          - dosave [bool, default=False]: If True, will save the figure under
              savename.  If False, will display the figure instead.
          - figsize [tuple, default=(8,8)]: Size of the final figure.
          - linestyle [str, default="-"]: Line style.  All matplotlib values
              are valid.  Used only for spectra.
          - linewidth [float, default=3.0]: Line thickness.  Used only for
              spectra.
          - ncol [int, default=7]: Number of columns.  Used only for channel
              maps.
          - prod [str]: The name of the desired product to plot.
          - savename [str, default="testing.png"]: Name for the saved figure.
              Only used if dosave=True.
          - textsize [int/float, default=14]: Font size for axis-label text.
          - title [str, default=""]: Title of the final figure.
          - titlesize [int/float, default=16]: Font size for title text.
          - vmax [float OR None, default=None]: Max. value for colorbar.
              If None, will use the max. value of emission within xlim,ylim
              range. Used only mom0s and channel maps.
          - vmin [float OR None, default=0]: Min. value for colorbar.  If None,
              will use the min. value of emission within xlim,ylim range. Used
              only mom0s and channel maps.
          - vrange_kms [list [km/s], default=None]: Velocity range of
              channels to plot.  If None, will plot all channels.
          - xlim [list [arcsec], default=[-3.0,3.0]: x-axis bounds of figure.
          - ylim [list [arcsec], default=[-3.0,3.0]: y-axis bounds of figure.
        OUTPUTS:
          - The desired plot (if it exists), or a helpful error that lists
              the available types of plots.  Can be saved or displayed.
        NOTES:
          - The available plots are:
            - "channels": The channel map with masks overplotted.
            - "mom0": The velocity-integrated emission map.
            - "spectrum": The spectrum.
        """
        #Raise an error if the desired plot is not allowed.
        if prod not in self._plot_list:
            errstr = ("Whoa! Looks like the plot you requested ({0}) "
                        .format(prod)
                        +"is not a supported plot. Supported plots are: {0}"
                        .format(self._plot_list))
            raise KeyError(errstr)

        #Gather mask parameters.  Should have been previously loaded.
        paramdict = {} #Temporary dictionary to hold looked-up parameters.
        for key in self._paramkeys:
            paramdict[key] = self.get_parameter(param=key, _inner=True,
                                            _keyset=self._paramkeys)

        #Gather image data for plots.  Should have been previously extracted.
        imdict = {} #Temporary dictionary to hold looked-up image cube data.
        for key in self._imkeys:
            imdict[key] = self._get_imdata(param=key)

        #Determine R.A. and Decl. axes, offset from midpoint.
        radeltarr = imdict["raarr"] - imdict["raarr"][paramdict["midx"]]
        decdeltarr = imdict["decarr"] - imdict["decarr"][paramdict["midy"]]
        #Convert to arcsec
        radeltarr *= 180.0/pi*3600 #[rad] -> [arcsec]
        decdeltarr *= 180.0/pi*3600 #[rad] -> [arcsec]

        #Plot the desired product.
        fig = plt.figure(figsize=figsize) #Base figure
        #For a channel map, if so desired.  Uses a utils method to do so.
        if prod == "channels":
            #Extract the channels, masks, and desired channel indices
            data = self._get_maskdata(param=prod)*1E3 #[Jy/beam]->[mJy/beam]
            if vmin is None: #Set min. colorbar value for 2D plots, if not given
                vmin = data.min()
            if vmax is None: #Set max. colorbar value for 2D plots, if not given
                vmax = data.max()
            masks = self._get_maskdata(param="masks")
            varr = imdict["velarr"]/1.0E3 #[m/s] -> [km/s]
            if vrange_kms is not None:
                chanrange = np.where(((varr <= vrange_kms[1])
                                        & (varr >= vrange_kms[0])))[0]
            else:
                chanrange = np.arange(0, len(masks), 1)
            #Plot the channel maps
            astmod.plot_channels(chanall_list=data, maskall_list=masks,
                whichchan_list=chanrange, x_arr=radeltarr, y_arr=decdeltarr,
                bmaj=imdict["bmaj"], bmin=imdict["bmin"], bpa=imdict["bpa"],
                velall_arr=varr, cmap=cmap, ncol=ncol,
                vmin=vmin, vmax=vmax, xlim=xlim, ylim=ylim,
                xlabel=r"\Delta R.A. [\"]", ylabel=r"\Delta Dec. [\"]",
                docbar=True, cbarlabel=r"mJy beam$^{-1}$",
                textsize=textsize, ticksize=textsize, titlesize=titlesize,
                beamx=beamx, beamy=beamy, plotbeam=True)
        #For a spectrum, if so desired.
        elif prod == "spectrum":
            #Extract the desired product data to plot.
            data = self._get_maskdata(param=prod)*1E3 #Change Jy to mJy

            #Generate the spectrum plot
            plt.plot((imdict["velarr"]-paramdict["vsys"])/1.0E3, #m/s -> km/s
            data, drawstyle="steps-mid", color=color, linewidth=linewidth,
            alpha=alpha, linestyle=linestyle) #Plot the spectrum
            #Scale the axes
            plt.xlim(xlim)
            plt.ylim(ylim)
            #Label the axes
            plt.xlabel(r"(v - v$_{sys}$), km s$^{-1}$", fontsize=textsize)
            plt.ylabel("mJy", fontsize=textsize)
        #For a mom0, if so desired.
        elif prod == "mom0":
            #Extract the desired product data to plot.
            data = self._get_maskdata(param=prod)*1E3/1.0E3
            #Above converts [mJy/beam*m/s] -> [mJy/beam*m/s]
            if vmin is None: #Set min. colorbar value for 2D plots, if not given
                vmin = data.min()
            if vmax is None: #Set max. colorbar value for 2D plots, if not given
                vmax = data.max()

            #Plot the mom0 and a colorbar
            axhere = fig.add_subplot(111)
            phere = plt.imshow(data, extent=[radeltarr[0], radeltarr[-1],
                                            decdeltarr[-1], decdeltarr[0]],
                                    origin="upper",
                                    vmin=vmin, vmax=vmax, cmap=cmap)
            cbar = plt.colorbar(phere)
            #Scale the axes
            plt.xlim(xlim)
            plt.ylim(ylim)
            axhere.set_xlim(axhere.get_xlim()[::-1]) #Makes x-axis descend
            #Label the axes
            plt.xlabel(r"$\Delta$ R.A. "+"[\"]", fontsize=textsize)
            plt.ylabel(r"$\Delta$ Dec. "+"[\"]", fontsize=textsize)
            cbar.set_label(r"mJy beam$^{-1}$ km s$^{-1}$",
                            fontsize=titlesize,
                            rotation=270, labelpad=20) #Cbar. title
            cbar.ax.tick_params(labelsize=textsize) #Cbar. font size
            #Plot the beam
            yspan = axhere.get_ylim() #ybounds of this subplot
            xspan = axhere.get_xlim() #xbounds of this subplot
            bloc = [(((xspan[1] - xspan[0])*beamx) + xspan[0]),
                (((yspan[1] - yspan[0])*beamy) + yspan[0])] #Scale beam
            ellhere = patch.Ellipse(xy=bloc,
                    width=imdict["bmin"]*180.0/pi*3600,
                    height=imdict["bmaj"]*180.0/pi*3600,
                    angle=imdict["bpa"]*-1*180.0/pi, zorder=300) #Plot beam
            ellhere.set_facecolor("gray")
            ellhere.set_edgecolor("black")
            axhere.add_artist(ellhere)

        #Title and save the figure, if so desired.
        plt.title(title, fontsize=titlesize)
        if dosave:
            plt.savefig(savename)
            plt.close()
        #Otherwise, display the figure.
        else:
            plt.show()
        return
    #
#
