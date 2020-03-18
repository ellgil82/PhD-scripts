""" Script for calculating differences between default MetUM ancillary files for the land-sea mask
and orography, and for plotting snapshot maps of near-surface temperature and wind vectors to visualise the effect of this.

N.B. Default UM orography and coastlines are based on the GLOBE dataset (NGDC, 1999) at 1 km resolution and are based on
data collected in 1993, so predates the collapse of the Larsen A and B ice shelves.

The updated land-sea mask is based on the SCAR Antarctic Digital Database coastline, version 7.0 (released January 2016
and available at https://www.add.scar.org/). The orography ancillary file is based on the Ohio State University RAMP
200 m resolution Antarctic Digital Elevation Model (DEM) (Hongxing, 1999).

References:
Hongxing, L. (1999). Development of an Antarctic Digital Elevation model. Technical report. Byrd Polar Research Center,
 	Ohio State University, page 157.
NGDC (National Geophysical Data Center) (1999). Global Land One-kilometer Base Elevation (GLOBE) v.1.

Dependencies:
- iris 1.11.0
- matplotlib 1.5.1
- numpy 1.10.4

Author: Ella Gilbert, 2018. Updated March 2020.

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
sys.path.append('/users/ellgil82/scripts/Tools/')
from rotate_data import rotate_data
from divg_temp_colourmap import shiftedColorMap

def load_vars(file):
	T_air = iris.load_cube(file, 'air_temperature')
	T_surf = iris.load_cube(file, 'surface_temperature')
	T_air.convert_units('celsius')
	T_surf.convert_units('celsius')
	u = iris.load_cube(file, 'x_wind')
	v = iris.load_cube(file, 'y_wind')
	v = v[:,:400]
	old_lsm = iris.load_cube('/data/clivarm/wip/ellgil82/May_2016/Compare/CS1/km1p5/20160525T1200Z_Peninsula_km1p5_ctrl_pa000.pp', 'land_binary_mask')
	old_orog = iris.load_cube('/data/clivarm/wip/ellgil82/May_2016/Compare/CS1/km1p5/20160525T1200Z_Peninsula_km1p5_ctrl_pa000.pp', 'surface_altitude')
	new_lsm = iris.load_cube('/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/20160522T1200Z_Peninsula_km1p5_Smith_tnuc_pa000.pp', 'land_binary_mask')
	new_orog = iris.load_cube('/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/20160522T1200Z_Peninsula_km1p5_Smith_tnuc_pa000.pp', 'surface_altitude')
	rotate_me = [T_air, T_surf, u, v]
	for i in rotate_me:
		real_lon, real_lat = rotate_data(i, 1, 2)
	me_too = [new_lsm, new_orog, old_lsm, old_orog]
	for j in me_too:
		real_lon, real_lat = rotate_data(j, 0, 1)
	return T_air[0,:,:], T_surf[0,:,:], u[0,:,:], v[0,:,:], new_orog, new_lsm, old_orog, old_lsm,  real_lon, real_lat

## Set up plotting options
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans','Verdana']

def orog_dif_plot():
	'''Plot spatial differences between MetUM model output using default and updated orography and coastline files during
	foehn and non-foehn conditions. Thesis Figure 4.8. '''
	# Load necessary files
	old_orog = iris.load_cube('/data/clivarm/wip/ellgil82/new_ancils/km1p5/orog/orog_original.nc', 'OROGRAPHY (/STRAT LOWER BC)')[0,0,:,:]
	new_orog = iris.load_cube('/data/clivarm/wip/ellgil82/new_ancils/km1p5/orog/new_orog_smoothed.nc', 'Height')[0, 0,:, :]
	old_lsm = iris.load_cube('/data/clivarm/wip/ellgil82/new_ancils/km1p5/lsm/lsm_original.nc', 'LAND MASK (No halo) (LAND=TRUE)')[0, 0, :, :]
	new_lsm = iris.load_cube('/data/clivarm/wip/ellgil82/new_ancils/km1p5/lsm/new_mask.nc', 'LAND MASK (No halo) (LAND=TRUE)')[0,0,:,:]
	# Set up figure
	fig, ax = plt.subplots(1, 1, figsize=(10,11.5))
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.tick_params(axis='both', which='both', length=0, labelbottom='off', labelleft='off')
	# Plot Cabinet Inlet AWS
	cube = iris.load_cube('/data/clivarm/wip/ellgil82/May_2016/Re-runs/CS2/20160522T1200Z_Peninsula_km1p5_ctrl_pa000.pp','surface_altitude')
	real_lon, real_lat = rotate_data(cube, 0, 1)
	ax.plot(-63.37105, -66.48272, markersize=15, marker='o', color='#f68080', zorder=10)
	# Calculate differences
	orog_dif = new_orog.data - old_orog.data
	lsm_dif = new_lsm.data - old_lsm.data
	# Mask data where no difference is seen
	orog_dif = ma.masked_where((lsm_dif == 0) & (new_lsm.data == 0), orog_dif)
	lsm_dif = ma.masked_where(lsm_dif == 0, lsm_dif)
	# Truncate colormap to minimise visual impact of one or two extreme values
	squished_bwr = shiftedColorMap(cmap=matplotlib.cm.bwr, min_val=-800, max_val=800, name='squished_bwr', var=orog_dif, start = .15, stop = .85)
	# Plot differences between old and new orography and LSM
	c = ax.pcolormesh(real_lon, real_lat, orog_dif, cmap='squished_bwr', vmin=-800, vmax=800, zorder = 1)
	lsm = ax.contourf(real_lon, real_lat, lsm_dif, cmap = 'bwr', vmax = 1, vmin = -1, zorder = 2)
	# Add new LSM and 25 m orography contour
	ax.contour(real_lon, real_lat, new_lsm.data, lw=3, colors='dimgrey', zorder = 3)
	ax.contour(real_lon, real_lat, new_orog.data, lw = 2, levels = [100], colors = 'dimgrey', zorder = 4)
	# Set up colour bar
	cbaxes = fig.add_axes([0.22, 0.12, 0.56, 0.03])
	cbticks = np.linspace(-800, 800, 4)
	cbticklabs = [-800, 0, 800]
	cb = plt.colorbar(c, cax=cbaxes, orientation='horizontal', ticks=cbticks)
	cb.set_ticks(cbticks, cbticklabs)
	cb.ax.set_xlabel('Surface elevation difference (m)', fontsize=30, labelpad=20, color='dimgrey')
	cb.ax.text(-0.3, 2.2, 'Area removed \nfrom new LSM', fontsize = 30, color = 'dimgrey')
	cb.ax.text(0.78, 2.2, 'Area added \nto new LSM', fontsize = 30, color = 'dimgrey')
	cb.outline.set_edgecolor('dimgrey')
	cb.outline.set_linewidth(2)
	cb.solids.set_edgecolor('face')
	cb.ax.tick_params(labelsize=30, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
	[l.set_visible(False) for (w, l) in enumerate(cb.ax.xaxis.get_ticklabels()) if w % 3 != 0]
	plt.subplots_adjust(bottom=0.27, left=0.11, right=0.89, top=0.95, hspace=0.05)
	plt.savefig('/users/ellgil82/figures/new_ancils/orog_difs_km1p5.png', transparent = True)
	plt.savefig('/users/ellgil82/figures/new_ancils/orog_difs_km1p5.eps', transparent = True)
	plt.savefig('/users/ellgil82/figures/new_ancils/orog_difs_km1p5.pdf', transparent=True)
	plt.show()

orog_dif_plot()