""" Script for calculating and plotting time series of meteorological variables and surface energy fluxes from model
data and automatic weather station observations.

This can be adapted to be include observations from any location within the model domain, and data can be manipulated
to include moving time-window averaging, mean statistics etc.

Dependencies:
- iris 1.11.0
- matplotlib 1.5.1
- numpy 1.10.4
- rotate_data.py and divg_temp_colourmap scripts in Tools folder (or update path to match the directory it is stored in)

Author: Ella Gilbert, 2017. Updated April 2018 and March 2020.

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
from matplotlib.ticker import FormatStrFormatter
import numpy.ma as ma
import scipy
import pandas as pd
import datetime
sys.path.append('/users/ellgil82/scripts/Tools/')
from tools import compose_date, compose_time, find_gridbox, rotate_data
from rotate_data import rotate_data
from divg_temp_colourmap import shiftedColorMap
from sklearn.metrics import mean_squared_error

## Set-up cases
case_study = 'CS1' # string of case study in the format 'CS' + number, e.g. 'CS1'
#res = 'km4p0' # string of resolution to match filename, e.g. 'km4p0'

# Make sure Python is looking in the right place for files
if case_study == 'CS1':
    os.chdir('/data/clivarm/wip/ellgil82/May_2016/Re-runs/') # path to data
    filepath = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/'
    case_start = '2016-05-08' # Must be in datetime format as a string, e.g. '2016-05-08'
    case_end = '2016-05-15'
    #AWS_idx = (25404,25812) # Indices of corresponding times in AWS data (hacky workaround)
    res_list = [ 'km1p5', 'km4p0'] # List of model resolutions you want to process
elif case_study == 'CS2':
    os.chdir('/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/4 km/')
    filepath = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/4 km/'
    case_start = '2016-05-23' # Must be in datetime format as a string, e.g. '2016-05-08'
    case_end = '2016-05-30'
    #AWS_idx = (26124,26508)
    res_list = ['km1p5', 'km4p0']

def load_AWS():
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

def load_SEB(res):
    '''Function to load in SEB data from the MetUM. Make sure the file names point to the correct file stream where your variables are stored.
    This can be adapted to use other formats, e.g. NetCDF, GRIB etc. (see Iris docs for further information: https://scitools.org.uk/iris/docs/latest/#).'''
    SEB = []
    surf = []
    for file in os.listdir(filepath):
        if fnmatch.fnmatch(file, '*%(res)s_*_pa012.pp' % locals()):
            surf.append(file)
        elif fnmatch.fnmatch(file, '*%(res)s_*_pb012.pp' % locals()):
            SEB.append(file)
    print('\n importing cubes at %(res)s resolution...' % locals())
    os.chdir(filepath)
    print ('\n Downwelling shortwave...')
    SW_d = iris.load_cube(SEB, 'surface_downwelling_shortwave_flux_in_air')
    print('\n Downwelling longwave...')
    LW_d = iris.load_cube(SEB, 'surface_downwelling_longwave_flux')
    print('\n Net shortwave...')
    SW_n = iris.load_cube(SEB, 'surface_net_downward_shortwave_flux')
    print('\n Net longwave...')
    LW_n = iris.load_cube(SEB, 'surface_net_downward_longwave_flux')
    print('\n Latent heat...')
    LH = iris.load_cube(SEB, 'surface_upward_latent_heat_flux')
    print('\n Sensible heat...')
    SH = iris.load_cube(SEB, 'surface_upward_sensible_heat_flux')
    print('\n Surface temperature...')
    T_surf = iris.load_cube(surf, 'surface_temperature')
    T_surf.convert_units('celsius')
    Var = [SH, LH, LW_d, SW_d, LW_n, SW_n, T_surf]
    ## Rotate projection
    print ('\n rotating pole...')
    for var in Var:
        real_lon, real_lat = rotate_data(var, 1, 2)
    ## Find the nearest grid box to the latitude of interest
    print ('\n finding AWS...')
    lon_index, lat_index = find_gridbox(-66.48272, -63.37105, real_lat, real_lon)
    print('\n converting time units...')
    #convert units within iris
    Time = SH.coord('time')
    Time_srs = Time.units.num2date(Time.points)
    # Create Larsen mask to return values only a) on the ice shelf, b) orography is < 50 m
    orog = iris.load_cube(filepath + res + '_orog.pp', 'surface_altitude')
    lsm = iris.load_cube(filepath + res + '_lsm.pp', 'land_binary_mask')
    a = np.ones((orog.shape))
    orog = orog[:270, 95:240].data
    lsm = lsm[:270, 95:240].data
    b = np.zeros((270, 95))
    c = np.zeros((270, 160))
    d = np.zeros((130, 400))
    orog = np.hstack((b, orog))
    orog = np.hstack((orog, c))
    orog = np.vstack((orog, d))
    lsm = np.hstack((b, lsm))
    lsm = np.hstack((lsm, c))
    lsm = np.vstack((lsm, d))
    mask2d = np.ma.masked_where(orog.data > 50, a)
    Larsen_mask = np.ma.masked_where(lsm.data == 0, mask2d)
    Larsen_mask = np.broadcast_to(Larsen_mask == 1, T_surf.shape, subok=True)
    SH = np.ma.masked_array(SH.data, Larsen_mask.mask)
    LH = np.ma.masked_array(LH.data, Larsen_mask.mask)
    SW_d = np.ma.masked_array(SW_d.data, Larsen_mask.mask)
    LW_d = np.ma.masked_array(LW_d.data, Larsen_mask.mask)
    SW_n = np.ma.masked_array(SW_n.data, Larsen_mask.mask)
    LW_n = np.ma.masked_array(LW_n.data, Larsen_mask.mask)
    # Flip turbulent fluxes to match convention (positive = down)
    LH = 0 - LH
    SH = 0 - SH
    # Calculate 5th and 95th percentiles to give estimate of variability in time series
    print('\n calculating percentiles...')
    percentiles = []
    for each_var in [SH, LH, SW_d, SW_n, LW_d, LW_n]:
        p95 = np.percentile(each_var, 95, axis=(1, 2))
        p5 = np.percentile(each_var, 5, axis=(1, 2))
        percentiles.append(p5)
        percentiles.append(p95)
    print('\n extracting time series from cubes...')
    # Just one grid box
    SW_d = SW_d[:,lat_index,lon_index].data
    LW_d = LW_d[:,lat_index,lon_index].data
    SW_n = SW_n[:,lat_index,lon_index].data
    LW_n = LW_n[:,lat_index,lon_index].data
    LH = LH[:,lat_index,lon_index].data
    SH = SH[:,lat_index,lon_index].data
    T_surf = T_surf[:,lat_index, lon_index].data
    print('\n constructing %(res)s series...' % locals())
    print ('\n making melt variable...')
    # Calculate total SEB (without Gs, which is unavailable in the UM)
    E = SW_n + LW_n + LH + SH
    # Create melt variable
    # Create masked array when Ts<0
    melt = np.ma.masked_where(T_surf<-0.025, E)
    melt = melt.data - melt.data*(np.ma.getmask(melt))
    melt_forced = np.ma.masked_where(AWS_var['Tsobs']<-0.025, E)
    melt_forced = melt_forced.data - melt_forced.data * (np.ma.getmask(melt_forced))
    melt_forced[melt_forced<0] = 0
    var_dict = {
    'Time_srs': Time_srs,
    'Ts': T_surf,
    'SW_n': SW_n,
    'SW_d': SW_d,
    'LW_n': LW_n,
    'LW_d': LW_d,
    'SH': SH,
    'LH': LH,
    'melt': melt,
    'melt_forced': melt_forced,
    'percentiles': percentiles,
    'E': E}
    return var_dict

def load_surf(res):
    '''Function to load in surface meteorological data from the MetUM. Make sure the file names point to the correct file stream where your variables are stored.
        This can be adapted to use other formats, e.g. NetCDF, GRIB etc. (see Iris docs for further information: https://scitools.org.uk/iris/docs/latest/#).'''
    surf = []
    for file in os.listdir(filepath):
        if fnmatch.fnmatch(file, '*%(res)s_*_pa012.pp' % locals()):
            surf.append(file)
    print('\n importing cubes...')
    os.chdir(filepath)
    T_surf = iris.load_cube(surf, 'surface_temperature')
    T_air = iris.load_cube(surf, 'air_temperature')
    orog = iris.load_cube(filepath+res+'_orog.pp', 'surface_altitude') # This assumes your orography and land-sea mask are stored separately,
    lsm = iris.load_cube(filepath + res + '_lsm.pp', 'land_binary_mask') # but can be adapted to read in from one of your file streams.
    T_surf.convert_units('celsius')
    T_air.convert_units('celsius')
    RH = iris.load_cube(surf, 'relative_humidity')
    ## Iris v1.11 version
    u_wind = iris.load_cube(surf, 'x_wind')
    v_wind = iris.load_cube(surf, 'y_wind')
    v_wind = v_wind[:,1:,:]
    Var = [T_surf, T_air, RH, u_wind, v_wind]
    ## Rotate projection
    print('\n rotating pole...')
    for var in Var:
        real_lon, real_lat = rotate_data(var, 1,2)
    ## Find the nearest grid box to the latitude of interest
    print('\n finding AWS...')
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
    # Calculate 5th and 95th percentiles to give estimate of variability in time series
    print ('\n calculating percentiles... (this may take some time)')
    percentiles = []
    for each_var in [T_surf, T_air, RH, sp_srs]:
        p95 = np.percentile(each_var, 95, axis = (1,2))
        p5 = np.percentile(each_var, 5, axis = (1,2))
        percentiles.append(p5)
        percentiles.append(p95)
    print('\n extracting time series from cubes...')
    # just one grid box
    T_surf = T_surf[:,lat_index,lon_index].data
    T_air = T_air[:,lat_index,lon_index].data
    RH = RH[:,lat_index,lon_index].data
    RH[RH>100] = 100
    sp_srs = sp_srs[:,lat_index, lon_index]
    print('\n constructing %(res)s series...' % locals())
    var_dict = {
    'sp_srs': sp_srs,
    'Ts': T_surf,
    'T_air': T_air,
    'RH': RH,
    'Time_srs': Time_srs,
    'percentiles': percentiles}
    return var_dict

# Load model data
SEB_1p5 = load_SEB('km1p5')
surf_1p5 = load_surf('km1p5')
SEB_4p0 = load_SEB('km4p0')
surf_4p0 = load_surf('km4p0')

## =========================================== SENSITIVITY TESTING ================================================== ##

## Does the temperature bias improve if we choose a more representative grid point?

def sens_test(res):
    surf = []
    for file in os.listdir(filepath):
        if fnmatch.fnmatch(file, '*%(res)s_*_pa012.pp' % locals()):
            surf.append(file)
    print ('\n importing cubes...')
    os.chdir(filepath)
    T_surf = iris.load_cube(surf, 'surface_temperature')
    T_air = iris.load_cube(surf, 'air_temperature')
    orog = iris.load_cube(filepath+res+'_orog.pp', 'surface_altitude')
    lsm = iris.load_cube(filepath + res + '_lsm.pp', 'land_binary_mask')
    T_surf.convert_units('celsius')
    T_air.convert_units('celsius')
    RH = iris.load_cube(surf, 'relative_humidity')
    ## Iris v1.11 version
    u_wind = iris.load_cube(surf, 'x_wind')
    v_wind = iris.load_cube(surf, 'y_wind')
    v_wind = v_wind[:,1:,:]
    Var = [T_surf, T_air, RH, u_wind, v_wind]
    ## Rotate projection
    print ('\n rotating pole...')
    #create numpy arrays of coordinates
    rotated_lat = RH.coord('grid_latitude').points
    rotated_lon = RH.coord('grid_longitude').points
    ## set up parameters for rotated projection
    pole_lon = 298.5
    pole_lat = 22.99
    #rotate projection
    real_lon, real_lat = iris.analysis.cartography.unrotate_pole(rotated_lon,rotated_lat, pole_lon, pole_lat)
    print ('\nunrotating pole...')
    lat = RH.coord('grid_latitude')
    lon = RH.coord('grid_longitude')
    lat = iris.coords.DimCoord(real_lat, standard_name='latitude',long_name="grid_latitude",var_name="lat",units=lat.units)
    lon = iris.coords.DimCoord(real_lon, standard_name='longitude',long_name="grid_longitude",var_name="lon",units=lon.units)
    for var in Var:
        var.remove_coord('grid_latitude')
        var.add_dim_coord(lat, data_dim=1)
        var.remove_coord('grid_longitude')
        var.add_dim_coord(lon, data_dim=2)
    ## Find the nearest grid box to the latitude of interest
    print ('\n finding AWS...')
    lon_index, lat_index = find_gridbox(-66.48272, -63.37105, real_lat, real_lon)
    print('\n converting time units...')
    #convert units within iris
    Time = T_surf.coord('time')
    Time_srs = Time.units.num2date(Time.points)
    #convert to numpy array
    v_CI = (v_wind.data)
    u_CI = (u_wind.data)
    sp_srs = np.sqrt((u_CI**2)+(v_CI**2))
    Ts_subset = T_surf[:, lat_index, 140 ]
    Ta_subset = T_air[:, lat_index, 140 ]
    wind_subset = sp_srs[:, lat_index, 140 ]
    return Ts_subset, Ta_subset, wind_subset

#Ts_subset, Ta_subset, wind_subset = sens_test('km1p5')

## ============================================ COMPUTE STATISTICS ================================================== ##

def calc_bias():
    ''' Calculate bias of modelled time series.'''
    # Forecast error
    surf_met_obs = [AWS_var['Tsobs'], AWS_var['Tair_2m'], AWS_var['RH'], AWS_var['FF_10m'], AWS_var['SWin_corr'], AWS_var['LWin'], AWS_var['SWnet_corr'], AWS_var['LWnet_corr'], AWS_var['Hsen'], AWS_var['Hlat'], AWS_var['E'], AWS_var['melt_energy'], AWS_var['melt_energy']]
    surf_mod = [surf_1p5['Ts'], surf_1p5['T_air'], surf_1p5['RH'], surf_1p5['sp_srs'], SEB_1p5['SW_d'], SEB_1p5['LW_d'], SEB_1p5['SW_n'], SEB_1p5['LW_n'],  SEB_1p5['SH'],  SEB_1p5['LH'], SEB_1p5['E'], SEB_1p5['melt'], SEB_1p5['melt_forced']]
    mean_obs = []
    mean_mod = []
    bias = []
    errors = []
    r2s = []
    rmses = []
    for i in np.arange(len(surf_met_obs)):
        b = surf_mod[i] - surf_met_obs[i]
        errors.append(b)
        mean_obs.append(np.mean(surf_met_obs[i]))
        mean_mod.append(np.mean(surf_mod[i]))
        bias.append(mean_mod[i] - mean_obs[i])
        slope, intercept, r2, p, sterr = scipy.stats.linregress(surf_met_obs[i], surf_mod[i])
        r2s.append(r2)
        mse = mean_squared_error(y_true=surf_met_obs[i], y_pred=surf_mod[i])
        rmses.append(np.sqrt(mse))
        idx = ['Ts', 'Tair', 'RH', 'wind', 'SWd', 'LWd', 'SWn', 'LWn', 'SH', 'LH', 'total', 'melt', 'melt forced']
    df = pd.DataFrame(index = idx)
    df['obs mean'] = pd.Series(mean_obs, index = idx)
    df['mod mean'] = pd.Series(mean_mod, index = idx)
    df['bias'] =pd.Series(bias, index=idx)
    df['rmse'] = pd.Series(rmses, index = idx)
    df['% RMSE'] = ( df['rmse']/df['obs mean'] ) * 100
    df['correl'] = pd.Series(r2s, index = idx)
    for i in range(len(surf_mod)):
        slope, intercept, r2, p, sterr = scipy.stats.linregress(surf_met_obs[i], surf_mod[i])
        print(idx[i])
        print('\nr2 = %s\n' % r2)
    print('RMSE/bias = \n\n\n')
    df.to_csv('/data/clivarm/wip/ellgil82/May_2016/Bias_and_RMSE_'+case_study+'.csv')
    print(df)

#calc_bias()

## Compare mean melt fluxes between observations and model
# Create masked array to compare melt only in periods of melting (i.e. a mean of non-zero melt)
obs_melt_nonzero = np.ma.masked_where(AWS_var['melt_energy']== 0,AWS_var['melt_energy'] ) # observed
melt_nonzero = np.ma.masked_where(SEB_1p5['melt']== 0,SEB_1p5['melt']) # modelled
melt_forced_nonzero = np.ma.masked_where(SEB_1p5['melt_forced']== 0,SEB_1p5['melt_forced'])

# Convert melt energy into mm w.e
# Define constants
Lf = 334000 # J kg-1
rho_H2O = 999.7 # kg m-3
melt_mmwe = ((AWS_var['melt_energy']/(Lf*rho_H2O))*1800)*1000 # multiply by (60 seconds * 30 mins) to get flux per second & convert to mm
cmv_melt = np.cumsum(melt_mmwe, axis = 0)
print(cmv_melt[-1]) # Observed meltwater production during case study

melt_mmwe_mod = ((SEB_1p5['melt']/(Lf*rho_H2O))*1800)*1000
cmv_melt_mod = np.cumsum(melt_mmwe_mod)
print(cmv_melt_mod[-1]) # Modelled meltwater production during case study

melt_mmwe_mod_F = ((SEB_1p5['melt_forced']/(Lf*rho_H2O))*1800)*1000
cmv_melt_mod_F = np.cumsum(melt_mmwe_mod_F)
print(cmv_melt_mod_F[-1]) # Modelled meltwater production, observed Ts prescribed during case study

def load_whole_dataset():
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
    case = AWS_srs.loc['2014-11-25':'2017-11-14'] #'2015-01-01':'2015-12-31'
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

AWS_total = load_whole_dataset()

melt_m_per_30min_obs = (AWS_total['melt_energy']/(Lf*rho_H2O)) # in mm per 30 min
melt_m_per_s_obs = melt_m_per_30min_obs*1800 # multiply by (60 seconds * 30 mins) to get flux per second
total_melt_mmwe = melt_m_per_s_obs*1000
total_melt_cmv = np.cumsum(total_melt_mmwe, axis = 0)
print(total_melt_cmv[-1]) # Total observed meltwater production, entire AWS series.

## ================================================ PLOTTING ======================================================== ##

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans',
                               'Verdana']

def SEB_plot():
    '''Plot time series of observed and modelled surface energy fluxes, including model output at two resolutions.
    Thesis Figures 4.10 and 4.11.'''
    fig, ax = plt.subplots(2,2,sharex= True, sharey = True, figsize=(22, 12))
    ax = ax.flatten()
    col_dict = {'0.5 km': '#33a02c', '1.5 km': '#f68080', '4.4 km': '#1f78b4'}
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    days = mdates.DayLocator(interval=1)
    dayfmt = mdates.DateFormatter('%d %b')
    plot = 0
    for r in ['1.5 km']: # can increase the number of res
        for j, k in zip(['SWin', 'LWin', 'Hsen', 'Hlat'], ['SW_d', 'LW_d', 'SH', 'LH']):
            limits = {'SWin': (-200,400), 'LWin': (-200,400), 'Tair_2m': (-25, 15), 'Tsurf': (-25, 15)}
            titles = {'SWin': 'SW$_{\downarrow}$ \n(W m$^{-2}$)', 'LWin': 'LW$_{\downarrow}$ \n(W m$^{-2}$)',
                      'Hsen': 'H$_L$ \n(W m$^{-2}$)', 'Hlat': 'H$_S$ \n(W m$^{-2}$)'}
            ax2 = ax[plot].twiny()
            obs = ax[plot].plot(AWS_var['datetime'], AWS_var[j], color='k', linewidth=2.5, label="Cabinet Inlet AWS", zorder = 3)
            ax2.plot(SEB_1p5['Time_srs'], SEB_1p5[k], linewidth=2.5, color=col_dict[r], label='*%(r)s UM output for Cabinet Inlet' % locals(), zorder = 5)
            ax2.plot(SEB_4p0['Time_srs'], SEB_4p0[k], linewidth=2.5, color='#1f78b4', label='*%(r)s UM output for Cabinet Inlet' % locals(), zorder=4)
            ax2.axis('off')
            ax2.set_xlim(SEB_1p5['Time_srs'][1], SEB_1p5['Time_srs'][-1])
            ax2.tick_params(axis='both', which='both', color = 'dimgrey', labelsize=24, width = 2, length = 5, labelcolor='dimgrey', pad=10)
            ax[plot].tick_params(axis='both', which='both', color='dimgrey', labelsize=24, width=2, length=5,labelcolor='dimgrey', pad=10)
            ax2.set_ylim([-200,400])
            ax[plot].set_ylabel(titles[j], rotation=0, fontsize=24, color='dimgrey', labelpad=80)
            lab = ax[plot].text(0.08, 0.85, zorder = 6, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
            plot = plot + 1
    for axs in [ax[0], ax[2]]:
        axs.yaxis.set_label_coords(-0.3, 0.5)
        axs.spines['right'].set_visible(False)
    for axs in [ax[1], ax[3]]:
        axs.yaxis.set_label_coords(1.3, 0.5)
        axs.yaxis.set_ticks_position('right')
        axs.spines['left'].set_visible(False)
    for axs in [ax[2], ax[3]]:
        #plt.setp(axs.get_yticklabels()[-2], visible=False)
        axs.xaxis.set_major_formatter(dayfmt)
    for axs in ax:
        axs.spines['top'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        axs.set_xlim(AWS_var['datetime'][1], AWS_var['datetime'][-1])
        axs.tick_params(axis='both', which='both', color = 'dimgrey', labelsize=24, width = 2, length = 5, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    #Legend
    lns = [Line2D([0],[0], color='k', linewidth = 2.5)]
    labs = ['Observations from Cabinet Inlet']
    for r in ['1.5 km', '4.4 km']:
        lns.append(Line2D([0],[0], color=col_dict[r], linewidth = 2.5))
        labs.append(r[0]+'.'+r[2]+' km UM output for Cabinet Inlet')#'1.5 km UM output for Cabinet Inlet')#
    lgd = ax[1].legend(lns, labs, bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=20)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.05, hspace = 0.1, top = 0.95, right = 0.79, left = 0.21, bottom = 0.08)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/SEB_'+case_study+'_both_res_no_range.png', transparent = True)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/SEB_'+case_study+'_both_res_no_range.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/SEB_'+case_study+'_both_res_no_range.pdf', transparent = True)
    plt.show()

def surf_plot():
    '''Plot time series of observed and modelled surface meteorological variables, including model output at two resolutions.
    Thesis Figures 4.2 and 4.3.'''
    fig, ax = plt.subplots(2,2,sharex= True, figsize=(22, 12))
    ax = ax.flatten()
    col_dict = {'0.5 km': '#33a02c', '1.5 km': '#f68080', '4.4 km': '#1f78b4'}
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l' }
    days = mdates.DayLocator(interval=1)
    dayfmt = mdates.DateFormatter('%d %b')
    plot = 0
    # Plot each variable in turn for 1.5 km resolution
    for r in ['1.5 km']: # can increase the number of res, '4.0 km'
        for j, k in zip(['RH', 'FF_10m', 'Tair_2m', 'Tsobs'], ['RH', 'sp_srs', 'T_air', 'Ts']):
            limits = {'RH': (0,100), 'FF_10m': (0,30), 'Tair_2m': (-25, 15), 'Tsobs': (-25, 15)}
            titles = {'RH': 'Relative \nhumidity (%)',
                      'FF_10m': 'Wind speed \n(m s$^{-1}$)',
                      'Tair_2m': 'T$_{air}$ ($^{\circ}$C)',
                      'Tsobs': 'T$_{S}$ ($^{\circ}$C)'}
            obs = ax[plot].plot(AWS_var['datetime'], AWS_var[j], color='k', linewidth=2.5, label="Cabinet Inlet AWS")
            ax2 = ax[plot].twiny()
            ax2.plot(surf_1p5['Time_srs'], surf_1p5[k], linewidth=2.5, color=col_dict[r], label='*%(r)s UM output for Cabinet Inlet' % locals(), zorder = 5)
            ax2.plot(surf_4p0['Time_srs'], surf_4p0[k], linewidth=2.5, color='#1f78b4', label='4.0 km UM output for Cabinet Inlet' , zorder = 4)
            ax2.axis('off')
            ax2.set_xlim(surf_1p5['Time_srs'][1], surf_1p5['Time_srs'][-1])
            ax[plot].set_xlim(surf_1p5['Time_srs'][1], surf_1p5['Time_srs'][-1])
            ax2.tick_params(axis='both', which='both', color='dimgrey', labelsize=24, width=2, length=5, labelcolor='dimgrey', pad=10)
            ax[plot].tick_params(axis='both', which='both', color='dimgrey', labelsize=24, width=2, length=5,labelcolor='dimgrey', pad=10)
            ax2.set_ylim(limits[j])
            ax[plot].set_ylim(limits[j])#[floor(np.floor(np.min(AWS_var[j])),5),ceil(np.ceil( np.max(AWS_var[j])),5)])
            ax[plot].set_ylabel(titles[j], rotation=0, fontsize=24, color = 'dimgrey', labelpad = 80)
            lab = ax[plot].text(0.08, 0.85, zorder = 100, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
            plot = plot + 1
        for axs in [ax[0], ax[2]]:
            axs.yaxis.set_label_coords(-0.3, 0.5)
            axs.spines['right'].set_visible(False)
        for axs in [ax[1], ax[3]]:
            axs.yaxis.set_label_coords(1.27, 0.5)
            axs.tick_params(axis='y', tick1On = True)
            axs.yaxis.tick_right()
            axs.spines['left'].set_visible(False)
        for axs in [ax[2], ax[3]]:
            #plt.setp(axs.get_yticklabels()[-2], visible=False)
            axs.xaxis.set_major_formatter(dayfmt)
            #plt.setp(axs.get_xticklabels()[])
        for axs in [ax[0], ax[1]]:
            [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        for axs in ax:
            axs.spines['top'].set_visible(False)
            plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
            axs.set_xlim(AWS_var['datetime'][1], AWS_var['datetime'][-1])
            axs.tick_params(axis='both', which='both', color = 'dimgrey', labelsize=24, width = 2, length = 5,  labelcolor='dimgrey', pad=10)
            [l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
    # Legend
    lns = [Line2D([0],[0], color='k', linewidth = 2.5)]
    labs = ['Observations from Cabinet Inlet']
    for r in ['1.5 km', '4.4 km']:#
        lns.append(Line2D([0],[0], color=col_dict[r], linewidth = 2.5))
        labs.append(r[0]+'.'+r[2]+' km UM output for Cabinet Inlet')#('1.5 km output for Cabinet Inlet')
    lgd = ax[1].legend(lns, labs, bbox_to_anchor=(0.55, 1.1), loc=2, fontsize=20)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    plt.subplots_adjust(wspace = 0.05, hspace = 0.05, top = 0.95, right = 0.85, left = 0.16, bottom = 0.08)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/Foehn_melt_surface_met_'+case_study+'_both_res_no_range.png', transparent = True)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/Foehn_melt_surface_met_'+case_study+'_both_res_no_range.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/Foehn_melt_surface_met_'+case_study+'_both_res_no_range.pdf', transparent = True)
    plt.show()

print('\nplotting surface vars....')
#surf_plot()

print('\nplotting SEBs....')
#SEB_plot()

def correl_plot():
    '''Plot correlation scatterplots between observed and modelled surface variables, including correlation coefficients
    from a Pearson correlation test. Thesis Figures 4.5 and 4.6.'''
    R_net = SEB_1p5['SW_n'] + SEB_1p5['LW_n']
    fig, ax = plt.subplots(4,2, figsize = (16,28))
    ax2 = ax[:,1]
    ax = ax.flatten()
    ax2.flatten()
    lab_dict = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    for axs in ax:
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey', )
        axs.set(adjustable='box-forced', aspect='equal')
        axs.tick_params(axis='both', which='both', color = 'dimgrey', labelsize=24, width = 2, length = 8, labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        #[l.set_visible(False) for (w, l) in enumerate(axs.xaxis.get_ticklabels()) if w % 2 != 0]
        axs.yaxis.set_label_coords(-0.4, 0.5)
    for axs in ax2:
        axs.spines['left'].set_visible(False)
        axs.spines['right'].set_visible(True)
        axs.yaxis.set_label_position('right')
        axs.yaxis.set_ticks_position('right')
        axs.tick_params(axis='both', which='both', color = 'dimgrey', labelsize=24, width = 2, length = 8,  labelcolor='dimgrey', pad=10)
        [l.set_visible(False) for (w, l) in enumerate(axs.yaxis.get_ticklabels()) if w % 2 != 0]
        axs.yaxis.set_label_coords(1.45, 0.57)
    plot = 0
    surf_met_mod = [surf_1p5['Ts'], surf_1p5['T_air'], surf_1p5['RH'], surf_1p5['sp_srs'], SEB_1p5['SW_d'], SEB_1p5['LW_d'], R_net, SEB_1p5['melt'] ]
    surf_met_obs = [AWS_var['Tsobs'], AWS_var['Tair_2m'], AWS_var['RH'], AWS_var['FF_10m'], AWS_var['SWin_corr'], AWS_var['LWin'], AWS_var['Rnet_corr'], AWS_var['melt_energy']]
    titles = ['$T_S$ \n($^{\circ}$C)', '$T_{air}$ \n($^{\circ}$C)', '\nRelative \nHumidity \n(%)', '\nWind speed \n(m s$^{-1}$)', '$SW_\downarrow$ \n(W m$^{-2}$)',  '$LW_\downarrow$ \n(W m$^{-2}$)', '$R_{net}$ \n(W m$^{-2}$)', 'E$_{melt}$ \n(W m$^{-2}$)']
    from itertools import chain
    for i in range(len(surf_met_mod)):
        slope, intercept, r2, p, sterr = scipy.stats.linregress(surf_met_obs[i], surf_met_mod[i])
        if p <= 0.01:
            ax[plot].text(0.9, 0.15, horizontalalignment='right', verticalalignment='top',
                          s='r = %s' % np.round(r2, decimals=2), fontweight = 'bold', transform=ax[plot].transAxes, size=24,
                          color='dimgrey')
        else:
            ax[plot].text(0.9, 0.15, horizontalalignment='right', verticalalignment='top',
                          s='r = %s' % np.round(r2, decimals=2), transform=ax[plot].transAxes, size=24,
                          color='dimgrey')
        ax[plot].scatter(surf_met_obs[i], surf_met_mod[i], color = '#f68080', s = 50)
        ax[plot].set_xlim(min(chain(surf_met_obs[i],surf_met_mod[i])), max(chain(surf_met_obs[i], surf_met_mod[i])))
        ax[plot].set_ylim(min(chain(surf_met_obs[i], surf_met_mod[i])), max(chain(surf_met_obs[i], surf_met_mod[i])))
        ax[plot].plot(ax[plot].get_xlim(), ax[plot].get_ylim(), ls="--", c = 'k', alpha = 0.8)
         #'r$^{2}$ = %s' % r2,
        #ax[plot].text(0.5, 1.1, s = titles[i], transform = ax[plot].transAxes)
        ax[plot].set_xlabel('Observed %s' % titles[i], size = 24, color = 'dimgrey', rotation = 0, labelpad = 10)
        ax[plot].set_ylabel('Modelled %s' % titles[i], size = 24, color = 'dimgrey', rotation =0, labelpad= 80)
        lab = ax[plot].text(0.1, 0.85, transform = ax[plot].transAxes, s=lab_dict[plot], fontsize=32, fontweight='bold', color='dimgrey')
        plot = plot +1
    plt.subplots_adjust(top = 0.98, hspace = 0.1, bottom = 0.05, wspace = 0.15, left = 0.2, right = 0.8)
    plt.setp(ax[2].get_xticklabels()[-2], visible=False)
    plt.setp(ax[0].get_yticklabels()[-2], visible=True)
    plt.setp(ax[0].get_yticklabels()[0], visible=True)
    #plt.setp(ax[5].get_xticklabels()[-2], visible=False)
    plt.setp(ax[7].get_yticklabels()[-1], visible=True)
    plt.setp(ax[7].get_yticklabels()[1], visible=True)
    plt.setp(ax[2].get_xticklabels()[-4], visible=False)
    plt.setp(ax[2].get_yticklabels()[-1], visible=False)
    plt.setp(ax[1].get_yticklabels()[-3], visible=False)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/'+ case_study + '_correlations.png', transparent=True )
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/'+ case_study + '_correlations.eps', transparent=True)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/'+ case_study + '_correlations.pdf', transparent=True)
    plt.show()

#correl_plot()

def total_SEB():
    '''Plot complete surface energy balance in observations, model output and model output, with melt calculated by prescribing
    observed Ts. Thesis Figures 4.12 and 4.13.'''
    fig, axs = plt.subplots(3, 1, figsize=(18, 27),  sharex = True, frameon=False)
    axs = axs.flatten()
    days = mdates.DayLocator(interval=1)
    dayfmt = mdates.DateFormatter('%b %d')
    for ax in axs:
        plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(AWS_var['datetime'][0],AWS_var['datetime'][-1])
        ax.xaxis.set_major_formatter(dayfmt)
        ax.tick_params(axis='both', which='both', labelsize=32,  width = 2, length = 10,  color = 'dimgrey', labelcolor='dimgrey', pad=10)
        ax.axhline(y=0, xmin=0, xmax=1, linestyle='--', linewidth=1)
        ax.set_ylabel('Energy flux \n (W m$^{-2}$)', rotation=0, fontsize=32, labelpad=70, color='dimgrey')
        [l.set_visible(False) for (w, l) in enumerate(ax.yaxis.get_ticklabels()) if w % 2 != 0]
        [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 2 != 0]
        ax.set_ylim(-200, 400)
    axs[0].plot(AWS_var['datetime'], AWS_var['SWnet_corr'], color = '#b733ff', markersize = 10, marker = 'o', lw = 4, label = 'SW$_{net}$')
    axs[0].plot(AWS_var['datetime'], AWS_var['LWnet_corr'], color = '#de2d26', marker = 'X', lw = 4, label = 'LW$_{net}$')
    axs[0].plot(AWS_var['datetime'], AWS_var['Hsen'],  color = '#ff6500', marker = '^', lw = 4, label = 'H$_{S}$')
    axs[0].plot(AWS_var['datetime'], AWS_var['Hlat'],  color = '#33aeff', marker = '*', lw = 4, label = 'H$_{L}$')
    axs[0].plot(AWS_var['datetime'], AWS_var['melt_energy'], color = '#222222', marker = 'P', lw = 4, label =  'E$_{melt}$')
    axs[1].plot(SEB_1p5['Time_srs'], SEB_1p5['SW_n'], color = '#b733ff', markersize = 10, marker = 'o', lw = 4, label = 'SW$_{net}$')
    axs[1].plot(SEB_1p5['Time_srs'], SEB_1p5['LW_n'],color = '#de2d26', marker = 'X', lw = 4, label = 'LW$_{net}$')
    axs[1].plot(SEB_1p5['Time_srs'], SEB_1p5['SH'],  color = '#ff6500', marker = '^', lw = 4, label = 'H$_{S}$')
    axs[1].plot(SEB_1p5['Time_srs'], SEB_1p5['LH'],  color = '#33aeff', marker = '*', lw = 4, label = 'H$_{L}$')
    axs[1].plot(SEB_1p5['Time_srs'], SEB_1p5['melt'],color = '#222222', marker = 'P', lw = 4, label =  'E$_{melt}$')
    axs[2].plot(SEB_1p5['Time_srs'], SEB_1p5['SW_n'], color = '#b733ff', markersize = 10, marker = 'o', lw = 4, label = 'SW$_{net}$')
    axs[2].plot(SEB_1p5['Time_srs'], SEB_1p5['LW_n'], color = '#de2d26', marker = 'X', lw = 4, label = 'LW$_{net}$')
    axs[2].plot(SEB_1p5['Time_srs'], SEB_1p5['SH'],  color = '#ff6500', marker = '^', lw = 4, label = 'H$_{S}$')
    axs[2].plot(SEB_1p5['Time_srs'], SEB_1p5['LH'],  color = '#33aeff', marker = '*', lw = 4, label = 'H$_{L}$')
    axs[2].plot(SEB_1p5['Time_srs'], SEB_1p5['melt_forced'], color = '#222222', marker = 'P', lw = 4, label =  'E$_{melt}$, forced')
    lgd = plt.legend(fontsize=24, bbox_to_anchor = (1., 1.2))
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey')
    lgd.get_frame().set_linewidth(0.0)
    axs[0].text(0.08, 0.85, zorder=100, transform=axs[0].transAxes, s='a', fontsize=32, fontweight='bold', color='dimgrey')
    axs[1].text(0.08, 0.85, zorder=100, transform=axs[1].transAxes, s='b', fontsize=32, fontweight='bold', color='dimgrey')
    axs[2].text(0.08, 0.85, zorder=100, transform=axs[2].transAxes, s='c', fontsize=32, fontweight='bold',color='dimgrey')
    plt.subplots_adjust(left=0.22, bottom=0.1, right=0.95, hspace = 0.1, wspace = 0.1)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/'+case_study+'total_SEB.eps', transparent = True)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/'+case_study+'total_SEB.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/' + case_study + 'total_SEB.pdf', transparent=True)
    plt.show()

#total_SEB()

def cmv_melt_plot():
    '''Plot cumulative meltwater production and instantaneous melt flux time series for AWS time series. Thesis Figure 4.14.'''
    fig, ax = plt.subplots(1,1, figsize = (18,7))
    ax2 = ax.twinx()
    for axs in [ax, ax2]:
        plt.setp(axs.spines.values(), linewidth=2, color='dimgrey')
        axs.tick_params(axis='both', which='both', labelsize=32,length = 8, color='dimgrey', width =2, labelcolor='dimgrey', pad=10)
        axs.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.plot(AWS_total['datetime'], AWS_total['melt_energy'], color = '#1f78b4', zorder = 2)
    ax2.plot(AWS_total['datetime'], total_melt_cmv, color='#f68080', lw = 5, zorder = 4)
    # Format x tick labels
    days = mdates.MonthLocator(interval=1)
    dayfmt = mdates.DateFormatter('%b %Y')
    ax2.set_ylabel('Cumulative \nmeltwater \nproduction \n (mm w.e.)', rotation=0, fontsize=32, labelpad=100, color='dimgrey')
    ax.set_ylabel('E$_{melt}$ \n(W m$^{-2}$)', rotation=0, fontsize=32, labelpad=100, color='dimgrey')
    ax.set_xlim(AWS_total['datetime'][0], AWS_total['datetime'][-1])
    ax.xaxis.set_major_formatter(dayfmt)
    ax.yaxis.set_label_coords(-0.23, 0.4)
    ax2.yaxis.set_label_coords(1.3, 0.65)
    [l.set_visible(False) for (w, l) in enumerate(ax.xaxis.get_ticklabels()) if w % 4 != 0]
    # Set limits
    ax2.set_ylim(0,1000)
    ax.set_ylim(0, 250)
    ax2.set_yticks([0,500,1000])
    ax.set_yticks([0,100,200])
    plt.subplots_adjust(left = 0.23, right = 0.75)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/cmv_melt.png')
    plt.savefig('/users/ellgil82/figures/Wintertime melt/cmv_melt.eps')
    plt.show()

#cmv_melt_plot()