import matplotlib as mpl
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
import astropy.wcs.utils as au
from astroquery.simbad import Simbad
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from photutils import CircularAperture, aperture_photometry, utils
from scipy.optimize import curve_fit

mpl.use('TkAgg')


def onclick(event):
    onclick.counter += 1
    if onclick.counter > len(colour_list):
        print('Maximum targets reached')
    else:
        x_centroids.append(event.xdata)
        y_centroids.append(event.ydata)
        np.append(y_centroids, np.array([event.ydata]))
        ax.add_patch(plt.Circle((event.xdata, event.ydata), radius=0.5 * aperture_size, fill=False,
                                label=f'Target {onclick.counter}', color=colour_list[onclick.counter - 1]))
        plt.legend()
        plt.draw()
        plt.show()


onclick.counter = 0


def power_law(x, a, alpha):
    return a * x ** (-alpha)


file = '\\Users\\lloyd\\Desktop\\Abell data cube\\Abell_85_aFix_pol_I_15arcsec_fcube_cor.fits'
hdu = fits.open(file)
hdr = hdu[0].header
hdu.info()
data = hdu[0].data
hdu.close()
# print(repr(hdr))

w = WCS(hdr)
slices, std = [], []
x_centroids, y_centroids = [], []
aperture_size = 100
annulus_size = 130
for s in range(12):
    background = sigma_clipped_stats(data[0, s, :, :], sigma=3)[2]
    slices.append(np.nan_to_num(data[0, s, :, :], background))
    std.append(background)
    if s == 0:
        colour_list = ['blue', 'green', 'yellow', 'white', 'cyan']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=w, slices=('x', 'y', 0, 0))
        # ax.set_title(
        #     'Slice: ' + str(s + 1) + ' @ ' + str(round(hdr["FREQ" + "{:04d}".format(s + 1)] * 1e-9, 4))
        #     + ' GHz')
        ax.set_title('Chosen targets from Abell 85')
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        pos = ax.imshow(slices[s], cmap='Reds_r', norm=SymLogNorm(0.005, vmin=-.00003, vmax=.4))
        fig.colorbar(pos, ax=ax, label='flux (Jy)')
        targets = 0
        # noinspection PyTypeChecker
        coords = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()

X, Y = [], []
chi_sqr, R = [], []
for k in range(len(x_centroids)):
    flux_densities, frequency, flux_error = [], [], []
    for i in range(12):
        data = slices[i]
        frequency.append(hdr['FREQ' + str(i+1).zfill(4)])

        positions = [(x_centroids[k], y_centroids[k])]
        aperture = CircularAperture(positions, r=aperture_size)
        photo_table = aperture_photometry(data, aperture,
                                          error=utils.calc_total_error(data, np.full_like(data, std[i]), 0))
        total_flux = photo_table['aperture_sum'][0]
        flux_err = photo_table['aperture_sum_err'][0]
        flux_densities.append(total_flux)
        flux_error.append(flux_err)
    flux_densities = np.array(flux_densities)
    frequency = np.array(frequency)
    popt, pcov = curve_fit(power_law, frequency, flux_densities, sigma=flux_error, p0=[1e9, 1], maxfev=99999)
    x_model = np.linspace(0.9e9, 1.67e9, 2000)
    y_model = power_law(x_model, *popt)

    expected = np.array(power_law(frequency, *popt))
    chi = np.sum((flux_densities - expected) ** 2 / expected)
    R2 = 1 - np.sum((flux_densities - expected) ** 2) / np.sum((flux_densities - np.mean(flux_densities)) ** 2)
    chi_sqr.append(chi)
    R.append(R2)
    print(
        f'Target {k + 1}: scale = {round(popt[0], 4)},  '
        f'alpha = {round(popt[1], 4)} +/- {round(np.sqrt(np.diag(pcov)[1]), 4)}'
        f' chi = {chi} & R^2 = {R2}')
    plt.plot(frequency, flux_densities, '.r', label='flux density')
    plt.plot(x_model, y_model, '-b', label='Power law best fit')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Flux (Jy)')
    plt.title(f'Flux density vs frequency for source {k + 1}')
    plt.tight_layout()
    plt.legend()
    # plt.savefig(f'assignment 2/Flux_density fit {k + 1}.svg', format='svg', dpi=1000)
    plt.show()

simbad = Simbad()
simbad.ROW_LIMIT = 5
simbad.add_votable_fields('otype')
sky = np.array([au.pixel_to_skycoord(x_centroids[i], y_centroids[i], w) for i in range(len(x_centroids))])

result = simbad.query_region(SkyCoord(sky), radius=2 * u.arcmin)

pd_result = result.to_pandas()
pd.set_option('display.max_rows', None)
print(pd_result.to_string())
