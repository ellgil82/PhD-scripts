'''Script to plot synoptic meteorological conditions before and during the onset of foehn from ERA-5 re-analysis and MetUM model output.

Author: Ella Gilbert, 2018. Updated March 2020.

'''

import iris
import numpy as np
import matplotlib.pyplot as plt
import fnmatch
import os
import matplotlib.dates as mdates
import sys
reload(sys)
sys.getdefaultencoding()
import numpy.ma as ma
import cartopy.crs as ccrs
import matplotlib
sys.path.append('/users/ellgil82/scripts/Tools/')
from rotate_data import rotate_data
from divg_temp_colourmap import shiftedColorMap

def load_files(case, period, res): #period should be either 'pre' for pre-foehn, or 'onset' for foehn conditions
    '''Load MetUM model output data from file.

    Inputs:

        - case: string indicating which case study is loaded, either 'CS1' or 'CS2'
        - period: string indicating either pre-foehn onset or during foehn onset, either 'pre' or 'onset'
        - res: string indicating resolution of model output loaded, either 'km1p5' or 'km4p0'

    Outputs:

        - dictionary of variables for plotting.

    '''
    if case == 'CS1':
        os.chdir('/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS1/')  # path to data
        filepath = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS1/'
        if period == 'pre':
            case_date = '20160507T1200Z'
        elif period == 'onset':
            case_date = '20160510T1200Z'
        else:
            print('period must be one of the following strings: "onset",  "pre"')
    elif case == 'CS2':
        os.chdir('/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/')
        filepath = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/'
        if period == 'pre':
            case_date = '20160522T1200Z'
        elif period ==  'onset':
            case_date = '20160526T0000Z'
        else:
            print('period must be one of the following strings: "onset","pre"')
    os.chdir(filepath)
    if res == 'km1p5':
        surf = filepath + case_date + '_Peninsula_km1p5_ctrl_pa000.pp'
    elif res == 'km4p0':
        surf = filepath+case_date+'_Peninsula_km4p0_ctrl_pa000.pp'
    print ('importing cubes...')
    T_air = iris.load_cube(surf, 'air_temperature')
    T_surf = iris.load_cube(surf, 'surface_temperature')
    lsm = iris.load_cube(surf, 'land_binary_mask')
    orog = iris.load_cube(surf, 'surface_altitude')
    P = iris.load_cube(surf, 'surface_air_pressure')
    P.convert_units('hPa')
    T_air.convert_units('celsius')
    T_surf.convert_units('celsius')
    ## Iris v1.11 version
    u_wind = iris.load_cube(surf, 'x_wind')
    v_wind = iris.load_cube(surf, 'y_wind')
    v_wind = v_wind[:,1:,:]
    ## Rotate projection
    for var in [T_air, T_surf, u_wind, v_wind, P]:
        real_lon, real_lat = rotate_data(var,1,2)
    for var in [lsm, orog]:
        real_lon, real_lat = rotate_data(var, 0, 1)
     #orog.data > 0 &
    var_dict: {'u': u_wind[0,:,:], 'v': v_wind[0,:,:], 'T_air': T_air[0,:,:], 'T_surf': T_surf[0,:,:], 'P': P[0,:,:],
               'lsm': lsm, 'real_lat': real_lat, 'real_lon': real_lon, 'orog': orog}
    return var_dict

def load_ERA(case): #period should be either 'pre' for pre-foehn, or 'onset' for foehn conditions
    '''Load ERA-5 re-analysis data from file.

    Inputs:

        - case: string indicating which case study is loaded, either 'CS1' or 'CS2'

    Outputs:

        - dictionary of variables for plotting.

    '''
    if case == 'CS1':
        os.chdir('/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS1/')  # path to data
        surf = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS1/20160510T1200Z_sfc_temp_MSLP.nc'
        winds = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS1/20160510T1200Z_750_wind_geopot.nc'
    elif case == 'CS2':
        os.chdir('/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/')
        surf = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/20160526T0000Z_sfc_temp_MSLP.nc'
        winds = '/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/20160526T0000Z_750_wind_geopot.nc'
    print ('importing cubes...')
    u = iris.load_cube(winds, 'eastward_wind')
    v = iris.load_cube(winds, 'northward_wind')
    P = iris.load_cube(surf, 'Mean sea level pressure')
    P.convert_units('hPa')
    geopot = iris.load_cube(winds, 'geopotential')
    air_T = iris.load_cube(surf, '2 metre temperature')
    air_T.convert_units('celsius')
    lon = P.coord('longitude')
    lat = P.coord('latitude')
    var_dict = {'u': u[0,:,:], 'v': v[0,:,:], 'air_T': air_T[0,:,:], 'geopot': geopot[0,:,:], 'P': P[0,:,:], 'lat': lat, 'lon': lon}
    return var_dict

def plot_both(period, res):
    ''' Plot spatial maps of synoptic conditions during both cases from MetUM model output, either before or during the onset
    of foehn conditions. Thesis Figure 4.4.

    Inputs:

        - period: string indicating either pre-foehn onset or during foehn onset, either 'pre' or 'onset'
        - res: string indicating resolution of model output loaded, either 'km1p5' or 'km4p0'

    Outputs:

        - image files in .png and .eps format

    '''
    fig, axs = plt.subplots(1,2, figsize=(20, 12), frameon=False)
    lab_dict = {0: 'c', 1: 'd'}
    #ax = fig.add_axes([0.18, 0.25, 0.75, 0.63], frameon=False) # , projection=ccrs.PlateCarree())#
    case_list = ['CS1', 'CS2']
    for a in [0,1]:
        axs[a].spines['right'].set_visible(False)
        axs[a].spines['left'].set_visible(False)
        axs[a].spines['top'].set_visible(False)
        axs[a].spines['bottom'].set_visible(False)
        MetUM_dict = load_files(case = case_list[a], period = period, res = res)
        axs[a].contour(MetUM_dict['lsm'].coord('longitude').points, MetUM_dict['lsm'].coord('latitude').points, MetUM_dict['lsm'].data, colors='#535454', linewidths=2.5,
                zorder=2)  # , transform=RH.coord('grid_longitude').coord_system.as_cartopy_projection())
        axs[a].contour(MetUM_dict['orog'].coord('longitude').points, MetUM_dict['orog'].coord('latitude').points, MetUM_dict['orog'].data, colors='#535454', levels=[50],
                linewidths=2.5, zorder=3) # Plot orography at approx. height of grounding line
        P_lev = axs[a].contour(MetUM_dict['lsm'].coord('longitude').points, MetUM_dict['lsm'].coord('latitude').points, MetUM_dict['P'], colors='#222222', linewidths=3,
                        levels=range(960, 1020, 4), zorder=4) # Plot MSLP contours
        axs[a].clabel(P_lev, v=[960, 968, 976, 982, 990], inline=True, inline_spacing=3, fontsize=28, fmt='%1.0f')
        bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-30., max_val=10., name='bwr_zero', var=T_air.data, start = 0., stop = 1.)
        c = axs[a].pcolormesh(MetUM_dict['real_lon'], MetUM_dict['real_lat'], MetUM_dict['T_air'].data, cmap=bwr_zero, vmin=-30., vmax=10., zorder=1)  # Plot temperatures
        x, y = np.meshgrid(MetUM_dict['real_lon'], MetUM_dict['real_lat'])
        q = axs[a].quiver(x[::25, ::25], y[::25, ::25], MetUM_dict['u'].data[::25, ::25], MetUM_dict['v'].data[::25, ::25], color='#414345',
                      pivot='middle', scale=100, zorder=5) # Plot 10 m wind vectors
        axs[a].tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
        # Set up plot ticks and labels
        PlotLonMin = np.min(real_lon)
        PlotLonMax = np.max(real_lon)
        PlotLatMin = np.min(real_lat)
        PlotLatMax = np.max(real_lat)
        XTicks = np.linspace(PlotLonMin, PlotLonMax, 3)
        XTickLabels = [None] * len(XTicks)
        for i, XTick in enumerate(XTicks):
            if XTick < 0:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
            else:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')
        plt.sca(axs[a])
        plt.xticks(XTicks, XTickLabels)
        axs[a].set_xlim(PlotLonMin, PlotLonMax)
        axs[a].tick_params(which='both', pad=10, labelsize=34, color='dimgrey')
        YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
        YTickLabels = [None] * len(YTicks)
        for i, YTick in enumerate(YTicks):
            if YTick < 0:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
            else:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
        plt.sca(axs[a])
        plt.yticks(YTicks, YTickLabels)
        axs[a].set_ylim(PlotLatMin, PlotLatMax)
        lab = axs[a].text(-80, -61.5, zorder=100,  s=lab_dict[a], fontsize=32, fontweight='bold', color='dimgrey')
        axs[a].set_title(case_list[a], fontsize=34, color='dimgrey')
    CBarXTicks = [-30, -10, 10]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.35, 0.15, 0.3, 0.03])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks)  #
    CBar.set_label('1.5 m air temperature ($^{\circ}$C)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                        tick2On=False)
    CBar.outline.set_linewidth(2)
    plt.sca(axs[1])
    plt.tick_params(axis = 'y', which=  'both', labelleft = 'off', labelright = 'on')
    #yaxis.set_label_coords(1.27, 0.5)
    plt.quiverkey(q, 0.51, 0.9, 10, r'$10$ $m$ $s^{-1}$', labelpos='N', color='#414345', labelcolor='#414345',
                  fontproperties={'size': '32', 'weight': 'bold'},
                  coordinates='figure', )
    plt.draw()
    plt.subplots_adjust(bottom = 0.25, top = 0.85)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/synop_cond_both_cases_'+ period + res + '.png')
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/synop_cond_both_cases_'+ period + res + '.eps')
    plt.show()

plot_both(period = 'pre', res = 'km4p0')

def plot_ERA():
    'Plot ERA-5 reanalysis data at the onset of foehn conditions during both cases. Thesis Figure 4.1.'
    fig, axs = plt.subplots(1,2, figsize=(20, 12), frameon=False)
    lab_dict = {0: 'a', 1: 'b'}
    #ax = fig.add_axes([0.18, 0.25, 0.75, 0.63], frameon=False) # , projection=ccrs.PlateCarree())#
    case_list = ['CS1', 'CS2']
    for a in [0,1]:
        axs[a].spines['right'].set_visible(False)
        axs[a].spines['left'].set_visible(False)
        axs[a].spines['top'].set_visible(False)
        axs[a].spines['bottom'].set_visible(False)
        ERA_dict = load_ERA(case=case_list[a])
        MetUM_dict = load_files(case=case_list[a], period='onset',res='km4p0')
        # Regrid ERA data onto MetUM 4 km domain grid
        regridded_P = P_ERA.regrid(MetUM_dict['P'], iris.analysis.Linear())
        regridded_P = np.ma.masked_where(lsm.data == 1, regridded_P.data)
        regridded_Tair = T_ERA.regrid(MetUM_dict['T_air'], iris.analysis.Linear())
        regridded_u = u_ERA.regrid(MetUM_dict['u_wind'], iris.analysis.Linear())
        regridded_v = v_ERA.regrid(MetUM_dict['v_wind'], iris.analysis.Linear())
        axs[a].contour(MetUM_dict['lsm'].coord('longitude').points, MetUM_dict['lsm'].coord('latitude').points, MetUM_dict['lsm'].data, colors='#535454', linewidths=2.5,
                zorder=2)  # , transform=RH.coord('grid_longitude').coord_system.as_cartopy_projection())
        axs[a].contour(MetUM_dict['orog'].coord('longitude').points, MetUM_dict['orog'].coord('latitude').points, MetUM_dict['orog'].data, colors='#535454', levels=[15],
                linewidths=2.5, zorder=3)
        P_lev = axs[a].contour(MetUM_dict['lsm'].coord('longitude').points, MetUM_dict['lsm'].coord('latitude').points, regridded_P, colors='#222222', linewidths=3,
                        levels=range(960, 1040, 8), zorder=4)
        axs[a].clabel(P_lev, v=[960, 968, 976, 982, 990], inline=True, inline_spacing=3, fontsize=28, fmt='%1.0f')
        bwr_zero = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-20., max_val=10., name='bwr_zero', var=regridded_Tair.data, start = 0., stop = 1.)
        c = axs[a].pcolormesh(MetUM_dict['real_lon'], MetUM_dict['real_lat'], regridded_Tair.data, cmap=bwr_zero, vmin=-20., vmax=10., zorder=1)  #
        x, y = np.meshgrid(MetUM_dict['real_lon'], MetUM_dict['real_lat'])
        q = axs[a].quiver(x[::40, ::40], y[::40, ::40], regridded_u.data[::40, ::40], regridded_v.data[::40, ::40], color='#414345',
                      pivot='middle', scale=150, zorder=5)
        axs[a].tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
        PlotLonMin = np.min(MetUM_dict['real_lon'])
        PlotLonMax = np.max(MetUM_dict['real_lon'])
        PlotLatMin = np.min(MetUM_dict['real_lat'])
        PlotLatMax = np.max(MetUM_dict['real_lat'])
        XTicks = np.linspace(PlotLonMin, PlotLonMax, 3)
        XTickLabels = [None] * len(XTicks)
        for i, XTick in enumerate(XTicks):
            if XTick < 0:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$W')
            else:
                XTickLabels[i] = '{:.0f}{:s}'.format(np.abs(XTick), '$^{\circ}$E')
        plt.sca(axs[a])
        plt.xticks(XTicks, XTickLabels)
        axs[a].set_xlim(PlotLonMin, PlotLonMax)
        axs[a].tick_params(which='both', pad=10, labelsize=34, color='dimgrey')
        YTicks = np.linspace(PlotLatMin, PlotLatMax, 4)
        YTickLabels = [None] * len(YTicks)
        for i, YTick in enumerate(YTicks):
            if YTick < 0:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$S')
            else:
                YTickLabels[i] = '{:.0f}{:s}'.format(np.abs(YTick), '$^{\circ}$N')
        plt.sca(axs[a])
        plt.yticks(YTicks, YTickLabels)
        axs[a].set_ylim(PlotLatMin, PlotLatMax)
        lab = axs[a].text(-80, -61.5, zorder=100,  s=lab_dict[a], fontsize=32, fontweight='bold', color='dimgrey')
        axs[a].set_title(case_list[a], fontsize=34, color='dimgrey')
    CBarXTicks = [-20, -10, 0, 10]  # CLevs[np.arange(0,len(CLevs),int(np.ceil(len(CLevs)/5.)))]
    CBAxes = fig.add_axes([0.35, 0.15, 0.3, 0.03])
    CBar = plt.colorbar(c, cax=CBAxes, orientation='horizontal', ticks=CBarXTicks)  #
    CBar.set_label('2 m air temperature ($^{\circ}$C)', fontsize=34, labelpad=10, color='dimgrey')
    CBar.solids.set_edgecolor("face")
    CBar.outline.set_edgecolor('dimgrey')
    CBar.ax.tick_params(which='both', axis='both', labelsize=34, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                        tick2On=False)
    CBar.outline.set_linewidth(2)
    plt.sca(axs[1])
    plt.tick_params(axis = 'y', which=  'both', labelleft = 'off', labelright = 'on')
    #yaxis.set_label_coords(1.27, 0.5)
    plt.quiverkey(q, 0.51, 0.9, 20, r'$20$ $m$ $s^{-1}$', labelpos='N', color='#414345', labelcolor='#414345',
                  fontproperties={'size': '32', 'weight': 'bold'},
                  coordinates='figure', )
    plt.draw()
    plt.subplots_adjust(bottom = 0.25, top = 0.85)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/ERA_foehn_onset_both_cases.png', transparent=True)
    plt.savefig('/users/ellgil82/figures/Wintertime melt/Re-runs/ERA_foehn_onset_both_cases.eps', transparent=True)
    plt.show()

plot_ERA()
