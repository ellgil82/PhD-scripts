# coding= ASCII
""" Script for calculating differences between one MetUM run and another, using different ancillary files (land-sea mask
and orography) and plotting snapshot maps of near-surface temperature and wind vectors to visualise this.

Dependencies:
- iris 1.11.0
- matplotlib 1.5.1
- numpy 1.10.4

Author: Ella Gilbert, 2018, updated January 2019.

"""

# Import modules
import iris
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import sys
reload(sys)
sys.getdefaultencoding()
from matplotlib import rcParams
import matplotlib
import numpy.ma as ma
import pandas as pd
sys.path.append('/users/ellgil82/scripts/Tools/')
from tools import compose_date, compose_time, find_gridbox
import datetime

## Define functions
def rotate_data(var, lat_dim, lon_dim):
    ## Rotate projection
    #create numpy arrays of coordinates
    rotated_lat = var.coord('grid_latitude').points
    rotated_lon = var.coord('grid_longitude').points
    ## set up parameters for rotated projection
    pole_lon = var.coord('grid_longitude').coord_system.grid_north_pole_longitude
    pole_lat = var.coord('grid_latitude').coord_system.grid_north_pole_latitude
    #rotate projection
    real_lon, real_lat = iris.analysis.cartography.unrotate_pole(rotated_lon,rotated_lat, pole_lon, pole_lat)
    print ('\nunrotating pole...')
    lat = var.coord('grid_latitude')
    lon = var.coord('grid_longitude')
    lat = iris.coords.DimCoord(real_lat, standard_name='latitude',long_name="grid_latitude",var_name="lat",units=lat.units)
    lon= iris.coords.DimCoord(real_lon, standard_name='longitude',long_name="grid_longitude",var_name="lon",units=lon.units)
    var.remove_coord('grid_latitude')
    var.add_dim_coord(lat, data_dim=lat_dim)
    var.remove_coord('grid_longitude')
    var.add_dim_coord(lon, data_dim=lon_dim)
    return real_lon, real_lat

## Set-up cases
case = 'CS2' # string of case study in the format 'CS' + number, e.g. 'CS1'

# Make sure Python is looking in the right place for files
if case == 'CS1':
    os.chdir('/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS1/') # path to data
    filepath = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/'
    f_old = '/data/clivarm/wip/ellgil82/May_2016/Compare/CS1/km1p5/'
    f_new = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/'
    case_start = '2016-05-08'  # Must be in datetime format as a string, e.g. '2016-05-08'
    case_end = '2016-05-15'
    i = np.arange(16)
elif case == 'CS2':
    os.chdir('/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/')
    filepath = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/'
    f_old = '/data/clivarm/wip/ellgil82/May_2016/Compare/CS2_new/'
    f_new = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/'
    case_start = '2016-05-23'  # Must be in datetime format as a string, e.g. '2016-05-08'
    case_end = '2016-05-30'
    i = np.arange(16)


def find_gridbox(x, y, real_lat, real_lon): # Finds the indices of the inputted lat/lon coordinates
    global lon_index, lat_index
    lat_index = np.argmin((real_lat - x) ** 2)  # take whole array and subtract lat you want from
    lon_index = np.argmin((real_lon - y) ** 2)  # each point, then find the smallest difference
    return lon_index, lat_index

def construct_srs(var_name):
    i = np.arange(16)
    k = var_name
    series = []
    for j in i:
        a = k[:12, j]
        a = np.array(a)
        series = np.append(series, a)
    return series

def construct_Timesrs(var_name):
    i = np.arange(16)
    m = var_name
    x = []
    for j in i:
        b = m[j, :12]
        x = np.append(x, b)
    return x

def load_surf(which): # 'which' can be either 'old' or 'new'
    '''Load time series of MetUM model output.

    Inputs:

    'which': either 'old' for time series using original MetUM model orography and coastlines, or 'new' for updated MetUM
             model orography and coastlines.

    Outputs:

    surf_var: Dictionary of surface meteorological variables.

    '''
    surf = []
    if which == 'old':
        os.chdir(f_old)
        for file in os.listdir(f_old):
            if fnmatch.fnmatch(file, '*km1p5_smoothed_pa000.pp'):
                surf.append(file)
    elif which == 'new':
        os.chdir(f_new)
        for file in os.listdir(f_new):
            if fnmatch.fnmatch(file,  '*km1p5_ctrl_pa012.pp'):
                surf.append(file)
    print ('\n importing cubes...')
    T_surf = iris.load_cube(surf, 'surface_temperature')
    T_air = iris.load_cube(surf, 'air_temperature')
    orog = iris.load_cube('/data/clivarm/wip/ellgil82/May_2016/Re-runs/km1p5_orog.pp', 'surface_altitude') # This assumes your orography and land-sea mask are stored separately,
    lsm = iris.load_cube('/data/clivarm/wip/ellgil82/May_2016/Re-runs/km1p5_lsm.pp', 'land_binary_mask') # but can be adapted to read in from one of your file streams.
    T_surf.convert_units('celsius')
    T_air.convert_units('celsius')
    RH = iris.load_cube(surf, 'relative_humidity')
    ## Iris v1.11 version
    u_wind = iris.load_cube(surf, 'x_wind')
    v_wind = iris.load_cube(surf, 'y_wind')
    if which == 'old':
        v_wind = v_wind[:,:,1:,:]
    elif which == 'new':
        v_wind = v_wind[:, 1:, :]
    Var = [T_surf, T_air, RH, u_wind, v_wind]
    ## Rotate projection
    print ('\n rotating pole...')
    for var in Var:
        if which == 'old':
            real_lon, real_lat = rotate_data(var, 2,3)
        elif which == 'new':
            real_lon, real_lat = rotate_data(var, 1, 2)
    ## Find the nearest grid box to the latitude of interest
    print ('\n finding AWS...')
    lon_index, lat_index = find_gridbox(-66.48272, -63.37105, real_lat, real_lon)
    print('\n converting time units...')
    #convert units within iris
    Time = T_surf.coord('time')
    Time_srs = Time.units.num2date(Time.points)
    print ('\n calculating wind speed...')
    ##convert u and v wind to wind speed
    #convert to numpy array
    v_CI = (v_wind.data)
    u_CI = (u_wind.data)
    sp_srs = np.sqrt((u_CI**2)+(v_CI**2))
    # Create Larsen mask !! Is this for 4 km or 1.5 ??
    a = np.ones((orog.shape))
    orog = orog[:270,95:240].data
    lsm = lsm[:270,95:240].data
    b =  np.zeros((270,95))
    c = np.zeros((270,160))
    d = np.zeros((130,400))
    orog = np.hstack((b, orog))
    orog = np.hstack((orog, c))
    orog = np.vstack((orog,d))
    lsm = np.hstack((b, lsm))
    lsm = np.hstack((lsm, c))
    lsm = np.vstack((lsm,d))
    mask2d = np.ma.masked_where(orog.data > 15, a)
    Larsen_mask = np.ma.masked_where(lsm.data == 0, mask2d)
    Larsen_mask = np.broadcast_to(Larsen_mask == 1, T_surf.shape, subok =True)
    T_surf = np.ma.masked_array(T_surf.data, Larsen_mask.mask)
    RH = np.ma.masked_array(RH.data, Larsen_mask.mask)
    T_air = np.ma.masked_array(T_air.data, Larsen_mask.mask)
    sp_srs = np.ma.masked_array(sp_srs, Larsen_mask.mask)
    print('\n extracting time series from cubes...')
    # just one grid box
    if which == 'new':
        T_surf = T_surf[:,lat_index,lon_index].data
        T_air = T_air[:,lat_index,lon_index].data
        RH = RH[:,lat_index,lon_index].data
        sp_srs = sp_srs[:,lat_index, lon_index]
    elif which == 'old':
        T_surf = T_surf[:,:,lat_index,lon_index].data
        T_air = T_air[:,:,lat_index,lon_index].data
        RH = RH[:,:, lat_index,lon_index].data
        sp_srs = sp_srs[:,:,lat_index, lon_index]
        T_surf = construct_srs(T_surf)
        T_air = construct_srs(T_air)
        RH = construct_srs(RH)
        sp_srs = construct_srs(sp_srs)
        Time_srs = construct_Timesrs(Time_srs)
    RH[RH > 100] = 100
    print('\n constructing series...')
    var_dict = {
    'sp_srs': sp_srs,
    'Ts': T_surf,
    'T_air': T_air,
    'RH': RH,
    'Time_srs': Time_srs}
    return var_dict

old_surf = load_surf(which = 'old')
new_surf = load_surf(which = 'new')

def load_AWS():
    '''Load AWS observations from case study.'''
    print('\nimporting AWS observations...')
    # Load AWS data
    AWS_srs = np.genfromtxt ('/data/clivarm/wip/ellgil82/AWS/iWS18_SEB_hourly_untilnov17.txt', names = True)
    AWS_srs = pd.DataFrame(AWS_srs) # Convert to pandas DataFrame this way because it loads in incorrectly using pd.from_csv
    # Calculate date, given list of years and day of year
    date_list = compose_date(AWS_srs['year'], days=AWS_srs['day'])
    AWS_srs['Date'] = date_list
    # Set date as index
    AWS_srs.index = AWS_srs['Date']
    # Calculate actual time from decimal DOY (seriously, what even IS that format?)
    AWS_srs['time'] = 24*(AWS_srs['Time'] - AWS_srs['day'])
    # Trim to case study
    print('\nsubsetting for Case Study...')
    case = AWS_srs.loc[case_start:case_end]
    print('\nconverting times...')
    # Convert times so that they can be plotted
    time_list = []
    for i in case['time']:
        hrs = int(i)                 # will now be 1 (hour)
        mins = int((i-hrs)*60)       # will now be 4 minutes
        secs = int(0 - hrs*60*60 + mins*60) # will now be 30
        j = datetime.time(hour = hrs, minute=mins)
        time_list.append(j)
    case['Time'] = time_list
    case['datetime'] = case.apply(lambda r : pd.datetime.combine(r['Date'],r['Time']),1)
    case['E'] = case['LWnet_corr'] + case['SWnet_corr'] + case['Hlat'] + case['Hsen'] - case['Gs']
    return case

AWS_var = load_AWS()

## ============================================ PLOTTING ================================================== ##

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans',
                               'Verdana']

def surf_plot():
    '''Plot time series of surface meteorological variables in two MetUM runs. Thesis Figure 4.9.'''
    fig, ax = plt.subplots(2,2,sharex= True, figsize=(22,12))
    ax = ax.flatten()
    days = mdates.DayLocator(interval=1)
    dayfmt = mdates.DateFormatter('%d %b')
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    plot = 0
    for j, k in zip(['RH', 'FF_10m', 'Tair_2m', 'Tsobs'], ['RH', 'sp_srs', 'T_air', 'Ts']):
        limits = {'RH': (0, 100), 'FF_10m': (0, 30), 'Tair_2m': (-25, 15), 'Tsobs': (-25, 15)}
        titles = {'RH': 'Relative \nhumidity (%)',
                  'FF_10m': 'Wind speed \n(m s$^{-1}$)',
                  'Tair_2m': '2 m air \ntemperature ($^{\circ}$C)',
                  'Tsobs': 'Surface \ntemperature ($^{\circ}$C)'}
        obs = ax[plot].plot(AWS_var['datetime'], AWS_var[j], color='k', linewidth=2.5, label="Cabinet Inlet AWS")
        ax2 = ax[plot].twiny()
        ax2.plot(old_surf['Time_srs'], old_surf[k], linewidth=2.5, color='#54278f', label='Default UM set-up', zorder=5)
        ax2.plot(new_surf['Time_srs'], new_surf[k], linewidth=2.5, color='#1c9099',label='Updated UM set-up', zorder=4)
        ax2.axis('off')
        ax2.set_xlim(new_surf['Time_srs'][1], new_surf['Time_srs'][-1])
        ax[plot].set_xlim(new_surf['Time_srs'][1], new_surf['Time_srs'][-1])
        ax2.tick_params(axis='both', which='both', labelsize=24, color = 'dimgrey', length = 8, width = 2, labelcolor='dimgrey', pad=10)
        ax2.set_ylim(limits[j])
        ax[plot].set_ylim(limits[j])  # [floor(np.floor(np.min(AWS_var[j])),5),ceil(np.ceil( np.max(AWS_var[j])),5)])
        ax[plot].set_ylabel(titles[j], rotation=0, fontsize=24, color='dimgrey', labelpad=80)
        ax[plot].tick_params(axis='both', which='both', labelsize=24, color = 'dimgrey', length = 8, width = 2, labelcolor='dimgrey', pad=10)
        lab = ax[plot].text(0.08, 0.85, zorder=100, transform=ax[plot].transAxes, s=lab_dict[plot], fontsize=32,fontweight='bold', color='dimgrey')
        plot = plot + 1
    for axs in [ax[0], ax[2]]:
        axs.yaxis.set_label_coords(-0.3, 0.5)
        axs.spines['right'].set_visible(False)
    for axs in [ax[1], ax[3]]:
        axs.yaxis.set_label_coords(1.27, 0.5)
        axs.yaxis.set_ticks_position('right')
        axs.tick_params(axis='y', tick1On=False)
        axs.spines['left'].set_visible(False)
    for axs in [ax[2], ax[3]]:
        plt.setp(axs.get_yticklabels()[-2], visible=False)
        axs.xaxis.set_major_formatter(dayfmt)
        # plt.setp(axs.get_xticklabels()[])
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        axs.set_xlim(AWS_var['datetime'][1], AWS_var['datetime'][-1])
        axs.tick_params(axis='both', which='both', labelsize=24, color = 'dimgrey', length = 8, width = 2, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    # Legend
    lns = [Line2D([0], [0], color='k', linewidth=2.5)]
    labs = ['Observations from Cabinet Inlet']
    col_dict = {'Default': '#54278f', 'Updated': '#1c9099'}
    for r in ['Default', 'Updated']:  #
        lns.append(Line2D([0], [0], color=col_dict[r],linewidth=2.5))
        labs.append(r + ' UM set-up')  # ('1.5 km output for Cabinet Inlet')
    lgd = ax[1].legend(lns, labs, bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=20)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.05, hspace = 0.05, top = 0.95, right = 0.83, left = 0.18, bottom = 0.08)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/'+case+'_old_v_new_srs.pdf', transparent = True)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/'+case+'_old_v_new_srs.eps', transparent = True)
    plt.show()

surf_plot()