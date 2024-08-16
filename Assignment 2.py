# Written by L.Landsberg & F.Campher
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
import matplotlib as mpl

# Set matplotlib backend for handling click events
mpl.use('TkAgg')


def onclick(event):
    """Handles click events on the plot, marking selected targets."""
    onclick.counter += 1
    if onclick.counter > len(color_list):
        print('Maximum targets reached')
    else:
        x_centroids.append(event.xdata)
        y_centroids.append(event.ydata)
        ax.add_patch(plt.Circle(
            (event.xdata, event.ydata),
            radius=0.5 * aperture_size,
            fill=False,
            label=f'Target {onclick.counter}',
            color=color_list[onclick.counter - 1]
        ))
        plt.legend()
        plt.draw()
        plt.show()


onclick.counter = 0  # Initialize counter for click events


def power_law(x, a, alpha):
    """Defines the power law function for curve fitting."""
    return a * x ** (-alpha)


# Load FITS file and extract data
file_path = '\\Users\\lloyd\\Desktop\\Abell data cube\\Abell_85_aFix_pol_I_15arcsec_fcube_cor.fits'
hdu = fits.open(file_path)
hdr = hdu[0].header
data_cube = hdu[0].data
hdu.close()

wcs = WCS(hdr)  # Create a WCS object from the header
slices, std_background = [], []
x_centroids, y_centroids = [], []
aperture_size = 100  # Aperture radius in pixels
annulus_size = 130  # Annulus size in pixels for background subtraction

# Process each slice in the data cube
for slice_idx in range(12):
    # Perform sigma-clipping to estimate background noise
    background = sigma_clipped_stats(data_cube[0, slice_idx, :, :], sigma=3)[2]
    slice_data = np.nan_to_num(data_cube[0, slice_idx, :, :], nan=background)
    slices.append(slice_data)
    std_background.append(background)

    if slice_idx == 0:
        color_list = ['blue', 'green', 'yellow', 'white', 'cyan']
        fig, ax = plt.subplots(subplot_kw={'projection': wcs, 'slices': ('x', 'y', 0, 0)})
        ax.set_title('Chosen targets from Abell 85')
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        image = ax.imshow(slice_data, cmap='Reds_r', norm=SymLogNorm(0.005, vmin=-0.00003, vmax=0.4))
        fig.colorbar(image, ax=ax, label='flux (Jy)')
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.tight_layout()
        plt.show()

# Analyze each selected target
chi_squares, r_squared_values = [], []
for target_idx in range(len(x_centroids)):
    flux_densities, frequencies, flux_errors = [], [], []

    for slice_idx in range(12):
        slice_data = slices[slice_idx]
        frequency = hdr[f'FREQ{str(slice_idx + 1).zfill(4)}']
        frequencies.append(frequency)

        # Perform aperture photometry
        position = [(x_centroids[target_idx], y_centroids[target_idx])]
        aperture = CircularAperture(position, r=aperture_size)
        photometry_results = aperture_photometry(
            slice_data,
            aperture,
            error=utils.calc_total_error(slice_data, np.full_like(slice_data, std_background[slice_idx]), 0)
        )
        flux_densities.append(photometry_results['aperture_sum'][0])
        flux_errors.append(photometry_results['aperture_sum_err'][0])

    # Fit a power law to the flux density data
    flux_densities = np.array(flux_densities)
    frequencies = np.array(frequencies)
    popt, pcov = curve_fit(
        power_law, frequencies, flux_densities,
        sigma=flux_errors, p0=[1e9, 1], maxfev=99999
    )

    # Model data using the fitted power law
    x_model = np.linspace(0.9e9, 1.67e9, 2000)
    y_model = power_law(x_model, *popt)

    # Calculate chi-squared and R-squared
    expected_flux = power_law(frequencies, *popt)
    chi_square = np.sum((flux_densities - expected_flux) ** 2 / expected_flux)
    r_squared = 1 - np.sum((flux_densities - expected_flux) ** 2) / np.sum(
        (flux_densities - np.mean(flux_densities)) ** 2)
    chi_squares.append(chi_square)
    r_squared_values.append(r_squared)

    # Print the fit parameters and plot the results
    print(
        f'Target {target_idx + 1}: scale = {round(popt[0], 4)},  '
        f'alpha = {round(popt[1], 4)} +/- {round(np.sqrt(np.diag(pcov)[1]), 4)} '
        f'chi = {chi_square} & R^2 = {r_squared}'
    )
    plt.plot(frequencies, flux_densities, '.r', label='flux density')
    plt.plot(x_model, y_model, '-b', label='Power law best fit')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Flux (Jy)')
    plt.title(f'Flux density vs frequency for source {target_idx + 1}')
    plt.tight_layout()
    plt.legend()
    plt.show()

# Convert pixel coordinates to sky coordinates
sky_coords = np.array([
    au.pixel_to_skycoord(x_centroids[i], y_centroids[i], wcs)
    for i in range(len(x_centroids))
])

# Query Simbad parameters for object types around selected targets
simbad = Simbad()
simbad.ROW_LIMIT = 5
simbad.add_votable_fields('otype')

# Query Simbad for objects within a 2 arcminute radius of the target coordinates
query_result = simbad.query_region(SkyCoord(sky_coords), radius=2 * u.arcmin)

# Convert query results to a Pandas DataFrame for easier viewing
pd_result = query_result.to_pandas()
pd.set_option('display.max_rows', None)
print(pd_result.to_string())
