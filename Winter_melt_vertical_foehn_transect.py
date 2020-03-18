''' Script to plot vertical transect through the Antarctic Peninsula mountains during foehn conditions.

Author: Ella Gilbert, adapted from Andy Elvidge, 2018. Updated March 2020.'''

''' Script to plot vertical transect through the Antarctic Peninsula mountains during foehn conditions.

Author: Ella Gilbert, adapted from Andy Elvidge, 2018. Updated March 2020.'''

os.chdir('/users/ellgil82/scripts/Wintertime_melt/')
import iris
import numpy as np
from tools import rotate_data, find_gridbox
import matplotlib.pyplot as plt
import fnmatch
import os
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import sys
reload(sys)
sys.getdefaultencoding()
from matplotlib import rcParams
import iris.plot as iplt
import matplotlib
sys.path.append('/users/ellgil82/scripts/Tools/')
from tools import compose_date, compose_time
from rotate_data import rotate_data
from divg_temp_colourmap import shiftedColorMap
import numpy.ma as ma
import scipy
import pandas as pd
from tools import rotate_data, find_gridbox, compose_date, compose_time
import pandas as pd
import datetime

## Set-up cases
case_study = 'CS1' # string of case study in the format 'CS' + number, e.g. 'CS1'
#res = 'km4p0' # string of resolution to match filename, e.g. 'km4p0'

# Make sure Python is looking in the right place for files
if case_study == 'CS1':
    os.chdir('/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS1/') # path to data
    filepath = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS1/'
    slice_time = '20160510T0000Z'
    case_start = '2016-05-08' # Must be in datetime format as a string, e.g. '2016-05-08'
    case_end = '2016-05-15'
    res_list = [ 'km1p5', 'km4p0'] # List of model resolutions you want to process
elif case_study == 'CS2':
    os.chdir('/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/')
    filepath = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/'
    case_start = '2016-05-23' # Must be in datetime format as a string, e.g. '2016-05-08'
    case_end = '2016-05-30'
    slice_time = '20160525T1200Z'
    res_list = ['km1p5', 'km4p0']

def load_surf(res):
    '''Load surface meteorological variables.

    Inputs:

        - res: string indicating resolution of model output loaded, either 'km1p5' or 'km4p0'.

    Outputs:

        - var_dict: dictionary of variables for plotting.
    '''
    surf = []
    vert = []
    for file in os.listdir(filepath):
        if fnmatch.fnmatch(file, slice_time+'_Peninsula*%(res)s_*_pa000.pp' % locals()):
            surf.append(file)
        elif fnmatch.fnmatch(file, slice_time+'_Peninsula*%(res)s_*_pc000.pp' % locals()):
            vert.append(file)
    print ('\n importing cubes...')
    os.chdir(filepath)
    theta = iris.load_cube(vert, 'air_potential_temperature')
    orog = iris.load_cube(surf, 'surface_altitude')
    zonal = iris.load_cube(vert, 'x_wind')
    ## Rotate projection
    print ('\n rotating pole...')
    for i in [theta, orog, zonal]:
        real_lon, real_lat = rotate_data(i, np.ndim(i)-2, np.ndim(i)-1)
    theta.convert_units('celsius')
    # Take orography data and use it to create hybrid height factory instance
    auxcoord=iris.coords.AuxCoord(orog.data,standard_name=str(orog.standard_name),long_name="orography",var_name="orog",units=orog.units)
    for x in [theta, zonal]:
        x.add_aux_coord(auxcoord,(np.ndim(x)-2, np.ndim(x)-1))
        factory=iris.aux_factory.HybridHeightFactory(sigma=x.coord("sigma"), delta = x.coord("level_height"), orography=x.coord("surface_altitude"))
        x.add_aux_factory(factory) # this should produce a 'derived coordinate', 'altitude' (test this with >>> print theta)
    var_dict = {'theta': theta,
    'zonal': zonal,
    'lon': real_lon,
    'lat': real_lat}
    return var_dict

surf_1p5 = load_surf('km1p5')
surf_4p0 = load_surf('km4p0')

#latitude to take transect at = 235

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans',
                               'Verdana']

#transect, profile = np.meshgrid(surf_1p5['theta'].coord('longitude').points, surf_1p5['theta'].coord('level_height').points)

def plot_transect(res):
    ''''Plot transect through mountains during foehn conditions. Thesis Figure 4.7.

    Inputs:

        - res: string indicating resolution of MetUM model output.

    Outputs:

        - image files in .eps or .png format

    '''
    fig, ax = plt.subplots(1,1, figsize = (12,10))
    ax.set_facecolor('#222222')
    bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-10., max_val=40., name='bwr_zero',
                               var=res['theta'].data, start=0., stop=1.)
    if res == surf_4p0:
        slice_idx = 163
        res_str = '4km'
    elif res == surf_1p5:
        slice_idx = 235
        res_str = '1p5km'
    cf = iplt.contourf(res['theta'][12, :, slice_idx, :], cmap=bwr_zero, vmin=-10.1, vmax=40.,
                       levels=[-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 13, 16, 19, 21, 25, 30, 35,
                               40])  #np.linspace(-10., 40., 21))
    c = iplt.contour(res['zonal'][:,slice_idx,:], levels = [0,8,16,24,32,40], colors = 'k')
    if case_study == 'CS1':
        plt.clabel(c, [8,16,24,32], rotation = 0, fontsize = 28, color = 'k', inline = True, fmt = '%.0f')
    elif case_study == 'CS2':
        plt.clabel(c, [8, 24, 32], rotation=0, fontsize=28, color='k', inline=True, fmt='%.0f')
    plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    PlotLonMin, PlotLonMax = -68, -60
    ax.set_ylim(0,5000)
    ax.set_yticks([0, 2500, 5000])
    ax.tick_params(axis='both', which='both', labelsize=32, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax.set_ylabel('Altitude\n(m)', rotation=0, fontsize=32, labelpad=70, color='dimgrey')
    [l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
    XTicks = np.linspace(PlotLonMin, PlotLonMax, 3)
    XTickLabels = [None] * len(XTicks)
    for i, XTick in enumerate(XTicks):
        if XTick < 0:
            XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
        else:
            XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')
    plt.sca(ax)
    plt.xticks(XTicks, XTickLabels)
    ax.set_xlim(PlotLonMin, PlotLonMax)
    CBarXTicks = [-10, 0, 10, 20, 30, 40]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.4, 0.15, 0.45, 0.03])
    CBar = plt.colorbar(cf, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks)  #
    CBar.set_label('Air potential temperature ($^{\circ}$C)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=32, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                        tick2On=False)
    CBar.outline.set_linewidth(2)
    plt.subplots_adjust(left = 0.3, right = 0.95, bottom = 0.27, top = 0.97)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/transects_theta_u_wind_'+case_study+'_'+res_str+'.png')
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/transects_theta_u_wind_'+case_study+'_'+res_str+'.eps')
    plt.show()

plot_transect(surf_4p0)

# Calculate Froude number during each case.

def Froude_number(u_wind):
    # At latitude of Cabinet Inlet: take mean u wind at 1 rossby wave of deformation upstream =~ 150 km, and average
    # over 200-2000 m to get flow conditions representative of those impinging on the peninsula (after Elvidge et al., 2016)
    mean_u = np.mean(u_wind)
    N = 0.01 # s-1 = Brunt-Vaisala frequency
    h = 2000 # m = height of AP mountains
    Fr = mean_u/(N*h)
    h_hat = (N*h)/mean_u
    return Fr, h_hat

Fr, h_hat = Froude_number(surf_4p0['zonal'][7:23,160:167, 95:102].data)
Fr, h_hat = Froude_number(u_ERA.data)
