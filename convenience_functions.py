# import Cython.Compiler.TypeSlots
import matplotlib
import numpy as np
from astropy.io import ascii
from astropy import constants as const

# matplotlib.use('Qt5Agg')  # Force a backend that supports specifying the location of the plot window
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from scipy import stats
import sys, argparse
from astropy.modeling import models, fitting
from scipy.spatial import ConvexHull
from matplotlib.collections import LineCollection
from pylab import MaxNLocator


def read_riess():
    """
    Read the Cepheid photometry in Table 2 of Riess+2019 and the parameters of the best-fitting PLs (Table 3).
    Sort by object name (ID).
    :return: star ID, log(period), magnitudes in the different bands (dictionary), slope, intercept and scatter of
                the best fitting PLs (all dictionaries).
    """
    # Photometry (Table 2)
    tab = ascii.read('Files/Riess_2019_ApJ_876_85_Tab2.txt')
    id_mag = np.char.replace(tab['ID'].data, 'OGL', 'OGLE')
    sort_ind = np.argsort(id_mag)
    #
    id_mag = id_mag[sort_ind]
    logP = tab['logP'].data[sort_ind]
    mag = dict()
    mag['F555W'], mag['F814W'] = tab['F555Wmag'].data[sort_ind], tab['F814Wmag'].data[sort_ind]
    mag['F160W'], mag['WH'] = tab['F160Wmag'].data[sort_ind], tab['mWH'].data[sort_ind]
    #
    emag = dict()
    emag['F555W'], emag['F814W'] = tab['e_F555Wmag'].data[sort_ind], tab['e_F814Wmag'].data[sort_ind]
    emag['F160W'], emag['WH'] = tab['e_F160Wmag'].data[sort_ind], tab['e_mWH'].data[sort_ind]
    #
    geo = tab['Geo'].data[sort_ind]

    # Best fitting-PL (Table 3)
    tab = ascii.read('Files/Riess_2019_ApJ_876_85_Tab3.txt')
    # Format as dictionaries
    pl_slope, pl_intercept, pl_scatter = dict(), dict(), dict()

    for row in tab:
        pl_slope[row['Band']] = row['Slope']
        pl_intercept[row['Band']] = row['Intercept']
        pl_scatter[row['Band']] = row['Scatter']
    del (row)

    return id_mag, logP, mag, emag, geo, pl_slope, pl_intercept, pl_scatter


def read_parameters(infile):
    """
    Read the values of metallicities and their uncertainties. Sort by object name (ID).
    :return: star ID, dictionary with the stellar parameters: [Fe/H], Teff, log(g), v_turb.
    """
#     indir = '/Users/mromanie/Desktop/PROPs/P106/SH0ES-LMC/'
    indir = 'Files/'
    tab = ascii.read(indir + infile)
    ids = tab['CEP'].data
    sort_ind = np.argsort(ids)

    ids = ids[sort_ind]
    Fe, Teff = tab['feh'].data[sort_ind], tab['teff'].data[sort_ind]
    logg, vturb = tab['logg'].data[sort_ind], tab['vtur'].data[sort_ind]
    #
    try:
        dFe = tab['dfeh'].data[sort_ind]
    except:
        dFe = 0.001 * np.ones(len(Fe))
    try:
        dTeff = tab['dteff'].data[sort_ind]
    except:
        dTeff = 1 * np.ones(len(Fe))
    try:
        dlogg = tab['dlogg'].data[sort_ind]
    except:
        dlogg = 0.001 * np.ones(len(Fe))
    try:
        dvturb = tab['dvtur'].data[sort_ind]
    except:
        dvturb = 0.001 * np.ones(len(Fe))

    stellar_parameters = dict()
    stellar_parameters['Fe'] = {'value': Fe, 'error': dFe, 'label': '[Fe/H]\ [dex]'}
    stellar_parameters['Teff'] = {'value': Teff, 'error': dTeff, 'label': 'T_{eff}\ [K]'}
    stellar_parameters['logTeff'] = {'value': np.log10(Teff), 'error': np.log10(1 + dTeff / Teff),
                                     'label': 'log(T_{eff})\ [K]'}
    stellar_parameters['logg'] = {'value': logg, 'error': dlogg, 'label': 'log(g)\ [cm/s^2]'}
    stellar_parameters['vturb'] = {'value': vturb, 'error': dvturb, 'label': 'v_{turb}\ [km/s]'}

    # Sara's tables don't have oxygen abundances ... protect against errors ..
    try:
        O = tab['oh'].data[sort_ind]
        dO = tab['doh'].data[sort_ind]
    except:
        O = np.empty(len(Fe))
        O[:] = np.nan
        #
        dO = np.empty(len(Fe))
        dO[:] = np.nan
    stellar_parameters['O'] = {'value': O, 'error': dO, 'label': '[O/H]'}

    # https://stackoverflow.com/questions/2333593/return-common-element-indices-between-two-numpy-arrays
    indices_with_Fe = np.nonzero(np.in1d(ids_mag, ids))[0]
    # https://numpy.org/doc/stable/reference/generated/numpy.setdiff1d.html
    indices_without_Fe = np.setdiff1d(np.arange(len(ids_mag)), indices_with_Fe)
    # print('Stars with no spectroscopy: %s\n' % ', '.join(map(str, ids_mag[indices_without_Fe])))

    stellar_parameters['ID'] = {'value': ids}
    stellar_parameters['logP'] = {'value': logPs[indices_with_Fe], 'label': 'log(P) [days]'}

    tab = ascii.read('Files/mag_hst_filters.dat')
    ids_ph, phases = tab['id'].data, tab['phase'].data
    phases = phases[np.argsort(ids_ph)]  # Make sure that the order is alphabetical by star ID
    stellar_parameters['phase'] = {'value': phases, 'label': 'Phase'}

    return stellar_parameters, indices_with_Fe, indices_without_Fe


def read_r08(mc):
    """
    Read the stellar and intrinsic parameters for Magellanic Cepheids from Table 9 and 3 of Romaniello+2008.
    :return: stellar and intrinsic parameters as dictionaries
    """
    if mc == 'LMC':
        data_start, data_end = 13, 35
    elif mc == 'SMC':
        data_start, data_end = 36, 50
    #
    tab = ascii.read('Files/Romaniello_2008_AA_488_731_Tab9.txt', data_start=data_start, data_end=data_end,
                     delimiter='&', format='no_header', fast_reader=False)
    id = np.char.replace(tab['col1'].data, '~', '')
    id = np.char.replace(id, 'HV', mc)
    sort_ind = np.argsort(id)
    #
    Fe = np.char.replace(tab['col5'].data, '--', '-').astype(np.float)
    dFe = 0.1 * np.ones(len(Fe))
    #
    Teff = tab['col2'].data
    dTeff = 100 * np.ones(len(Teff))
    #
    logg = tab['col4'].data
    dlogg = 0.1 * np.ones(len(logg))
    #
    vturb = tab['col3'].data
    dvturb = 0.1 * np.ones(len(vturb))
    #
    stellar_parameters = {'id': id[sort_ind]}
    stellar_parameters['Fe'] = {'value': Fe[sort_ind], 'error': dFe[sort_ind], 'label': '[Fe/H]\ [dex]'}
    stellar_parameters['Teff'] = {'value': Teff[sort_ind], 'error': dTeff[sort_ind], 'label': 'T_{eff}\ [K]'}
    stellar_parameters['logTeff'] = {'value': np.log10(Teff[sort_ind]),
                                     'error': np.log10(1 + dTeff[sort_ind] / Teff[sort_ind]),
                                     'label': 'log(T_{eff})'}
    stellar_parameters['logg'] = {'value': logg[sort_ind], 'error': dlogg[sort_ind], 'label': 'log(g)\ [cm/s^2]'}
    stellar_parameters['vturb'] = {'value': vturb[sort_ind], 'error': dvturb[sort_ind], 'label': 'v_{turb}\ [km/s]'}

    if mc == 'LMC':
        data_start, data_end = 14, 36
    elif mc == 'SMC':
        data_start, data_end = 38, 52
    #
    tab = ascii.read('Files/Romaniello_2008_AA_488_731_Tab3.txt', data_start=data_start, data_end=data_end,
                     delimiter='&', format='no_header', fast_reader=False)
    id = np.char.replace(tab['col1'].data, '~', '')
    id = np.char.replace(id, 'HV', mc)
    sort_ind = np.argsort(id)
    #
    logP = tab['col2'].data
    phase = tab['col3'].data
    b = tab['col4'].data
    v = tab['col5'].data
    try:
        b = np.char.replace(b, '\\ldots', 'nan').astype(np.float)
        v = np.char.replace(v, '\\ldots', 'nan').astype(np.float)
    except:
        pass
    k = tab['col6'].data
    ebv = np.char.replace(tab['col7'].data, '\\\\', '')
    ebv = np.char.replace(ebv, '\\hline', '')
    ebv = np.char.replace(ebv, '--', '-').astype(np.float)
    #
    intrinsic_parameters = {'id': id[sort_ind]}
    intrinsic_parameters['logP'] = logP[sort_ind]
    intrinsic_parameters['phase'] = phase[sort_ind]
    intrinsic_parameters['B0'] = b[sort_ind]
    intrinsic_parameters['V0'] = v[sort_ind]
    intrinsic_parameters['K0'] = k[sort_ind]
    intrinsic_parameters['EBV'] = ebv[sort_ind]

    '''
    ff, aa = plt.subplots()
    aa.plot(logP, ebv, 'o')
    popt, pcov = curve_fit(func_line, logP, ebv, (0, np.mean(ebv)))
    perr = np.sqrt(np.diag(pcov))
    aa.axline((np.mean(logP), np.mean(ebv)), slope=popt[0])
    print('qaz', popt[0], perr[0])
    plt.show()
    '''

    return stellar_parameters, intrinsic_parameters


# ______________________________________________________________________________________________________________________
# _______________________________________________ Convenience functions ________________________________________________
def is_ipython():
    try:
        get_ipython
        return True
    except:
        return False


def tab2arr(tab, col):
    if np.ma.is_masked(tab[col]):
        # The column read from the file is Masked ... replace the missing values with NaN
        return np.array(tab[col].filled(np.nan).data)
    else:
        return np.array(tab[col].data)

def log2lin(xx):
    return 10 ** xx


def lin2log(xx):
    return np.log10(xx)


def make_format(current, other):
    """
    https://stackoverflow.com/questions/21583965/matplotlib-cursor-value-with-two-axes
    """

    # current and other are axes
    def format_coord(x, y):
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x, y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        ax_coord = inv.transform(display_coord)

        coords_str = 'x=' + "{:.2f}".format(x) + ' y_left=' + "{:.2f}".format(y) + ' y_right=' + \
                     "{:.1f}".format(ax_coord[1])  # MRO
        return (coords_str)
        # coords = [ax_coord, (x, y)]
        # return ('Left: {:<40}    Right: {:<}'
        #         .format(*['({:.3f}, {:.3f})'.format(x, y) for x, y in coords]))

    return format_coord


def set_window_position(fig, x, y):
    """
    Set the absolute on-screen position of the window depending on the backend in use
    """
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'Qt5Agg':
        # https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
        __, __, wdx, wdy = plt.get_current_fig_manager().window.geometry().getRect()
        plt.get_current_fig_manager().window.setGeometry(x, y, wdx, wdy)


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def func_line(x, slope, intercept):
    """
    Service function to return a straight line to fit.
    :return: straight line.
    """
    return x * slope + intercept


def func_gauss(x, a, x0, sigma):
    """
    Service function to define a Gaussian to fit.
    :return: Gaussian
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy

    One-line alternative from the same stackoverflow page:
    np.sqrt(np.cov(values, aweights=weights))
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return average, np.sqrt(variance)


def peak_to_sigma(x):
    """
    Transformation from peak value to sigma for a Gaussian normalised to have an integral of 1.
    It is used to compute the secondary y axis.
    :return:
    """
    # Treating x==0 manually, see:
    # https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/secondary_axis.html#sphx-glr-gallery-subplots-axes-and-figures-secondary-axis-py
    x = np.array(np.sqrt(2 * np.pi) * x).astype(float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1 / x[~near_zero]

    return x


def deLatex(instring):
    """
    Remove the LaTex markups from the input string
    :param instring: input string to be de-LaTeX-ed.
    :return: de-LaTex-ed string.
    """
    return instring.replace('_', '').replace('{', '').replace('}', '').replace('\ ', '_')


def latex(instring, markups):
    """
    Wrap the instring with LaTex markup.
    :param instring: input string to be LaTex-ed
    :param markups: LaTex markups to be inserted
    :return: LaTex-ed string
    """
    out = '$'
    for n, i in enumerate(markups.split('\\')[1:]):
        out += '\\' + i + '{'
    out += instring + '}' * (n + 1) + '$'
    return out


def select(sample, what):
    """
    :param sample:
    :return:
    """
    if sample == 'S20':
        yys, dyys = stellar_parameters[what]['value'], stellar_parameters[what]['error']
        label = stellar_parameters[what]['label']
        lgP = stellar_parameters['logP']['value']
        phases = stellar_parameters['phase']['value']
        ids = stellar_parameters['ID']['value']
        window_title = 'SH0ES2020 sample analysed by Sara with 2020 LDR Teff'
    elif sample == 'S20all':
        yys, dyys = stellar_parameters_alllines[what]['value'], stellar_parameters_alllines[what]['error']
        label = stellar_parameters_alllines[what]['label']
        lgP = stellar_parameters_alllines['logP']['value']
        phases = stellar_parameters_alllines['phase']['value']
        ids = stellar_parameters_alllines['ID']['value']
        window_title = 'SH0ES2020 sample analysed by Sara with 2020 LDR Teff, full Genovali linelist'
    elif sample == 'R20':
        yys, dyys = stellar_parameters08[what]['value'], stellar_parameters08[what]['error']
        label = stellar_parameters08[what]['label']
        lgP = intrinsic_parameters_r08['logP']
        phases = intrinsic_parameters_r08['phase']
        ids = intrinsic_parameters_r08['id']
        window_title = 'R08 sample analysed by Sara with 2020 LDR Teff'
    elif sample == 'R20M':
        yys, dyys = stellar_parameters08M[what]['value'], stellar_parameters08M[what]['error']
        label = stellar_parameters08M[what]['label']
        lgP = intrinsic_parameters_r08['logP']
        phases = intrinsic_parameters_r08['phase']
        ids = intrinsic_parameters_r08['id']
        window_title = 'R08 sample analysed by Martino with 2020 LDR Teff'
    elif sample == 'R20M08':
        yys, dyys = stellar_parameters0808M[what]['value'], stellar_parameters0808M[what]['error']
        label = stellar_parameters0808M[what]['label']
        lgP = intrinsic_parameters_r08['logP']
        phases = intrinsic_parameters_r08['phase']
        ids = intrinsic_parameters_r08['id']
        window_title = 'R08 sample analysed by Martino with 2008 LDR Teff'
    elif sample == 'RT20':
        yys, dyys = stellar_parametersT08[what]['value'], stellar_parametersT08[what]['error']
        label = stellar_parametersT08[what]['label']
        lgP = intrinsic_parameters_r08['logP']
        phases = intrinsic_parameters_r08['phase']
        ids = intrinsic_parameters_r08['id']
        window_title = 'R08 sample analysed by Sara with 2020 excitation equilibrium Teff'
    elif sample == 'RT20M':
        yys, dyys = stellar_parametersT08M[what]['value'], stellar_parametersT08M[what]['error']
        label = stellar_parametersT08M[what]['label']
        lgP = intrinsic_parameters_r08['logP']
        phases = intrinsic_parameters_r08['phase']
        ids = intrinsic_parameters_r08['id']
        window_title = 'R08 sample analysed by Martino with 2020 excitation equilibrium Teff from 5500K, full ' \
                       'Genovali linelist'
    elif sample == 'RT20MSMC':
        yys, dyys = stellar_parametersT08MSMC[what]['value'], stellar_parametersT08MSMC[what]['error']
        label = stellar_parametersT08MSMC[what]['label']
        lgP = intrinsic_parameters_r08smc['logP']
        phases = intrinsic_parameters_r08smc['phase']
        ids = intrinsic_parameters_r08smc['id']
        window_title = 'R08 SMC sample analysed by Martino with 2020 excitation equilibrium Teff from 5500K, full ' \
                       'Genovali linelist'
    elif sample == 'R08':
        yys, dyys = stellar_parameters_r08[what]['value'], stellar_parameters_r08[what]['error']
        label = stellar_parameters_r08[what]['label']
        lgP = intrinsic_parameters_r08['logP']
        phases = intrinsic_parameters_r08['phase']
        ids = intrinsic_parameters_r08['id']
        window_title = 'R08 sample straight from the 2008 paper'
    elif sample == 'A20':
        yys, dyys = stellar_parametersA[what]['value'], stellar_parametersA[what]['error']
        label = stellar_parametersA[what]['label']
        lgP = stellar_parametersA['logP']['value']
        phases = stellar_parametersA['phase']['value']
        ids = stellar_parametersA['ID']['value']
        window_title = 'SH0ES2020 sample analysed by Alessio with excitation equilibrium Teff'
    elif sample == 'T20':
        yys, dyys = stellar_parametersT[what]['value'], stellar_parametersT[what]['error']
        label = stellar_parametersT[what]['label']
        lgP = stellar_parametersT['logP']['value']
        phases = stellar_parametersT['phase']['value']
        ids = stellar_parametersT['ID']['value']
        window_title = 'SH0ES2020 sample analysed by Sara with excitation equilibrium Teff'
    elif sample == 'T20all':
        yys, dyys = stellar_parametersT_alllines[what]['value'], stellar_parametersT_alllines[what]['error']
        label = stellar_parametersT_alllines[what]['label']
        lgP = stellar_parametersT_alllines['logP']['value']
        phases = stellar_parametersT_alllines['phase']['value']
        ids = stellar_parametersT_alllines['ID']['value']
        window_title = 'SH0ES2020 sample analysed by Sara/Mqrtino with excitation equilibrium Teff from LDR, full ' \
                       'Genovali linelist'
    elif sample == 'Tot20all':
        yys, dyys = stellar_parametersTotal_alllines[what]['value'], stellar_parametersTotal_alllines[what]['error']
        label = stellar_parametersTotal_alllines[what]['label']
        lgP = stellar_parametersTotal_alllines['logP']['value']
        phases = stellar_parametersTotal_alllines['phase']['value']
        ids = stellar_parametersTotal_alllines['ID']['value']
        window_title = 'SH0ES2020 sample analysed by Martino with excitation equilibrium Teff from 5500K, full ' \
                       'Genovali linelist'
    elif sample == 'Tot20allRT20M':
        yys1, dyys1 = stellar_parametersTotal_alllines[what]['value'], stellar_parametersTotal_alllines[what]['error']
        yys2, dyys2 = stellar_parametersT08M[what]['value'], stellar_parametersT08M[what]['error']
        yys, dyys = np.concatenate((yys1, yys2)), np.concatenate((dyys1, dyys2))
        label = stellar_parametersTotal_alllines[what]['label']
        lgP1 = stellar_parametersTotal_alllines['logP']['value']
        ids1 = stellar_parametersTotal_alllines['ID']['value']
        phases1 = stellar_parametersTotal_alllines['phase']['value']
        lgP2 = intrinsic_parameters_r08['logP']
        ids2 = intrinsic_parameters_r08['id']
        phases2 = intrinsic_parameters_r08['phase']
        lgP, ids = np.concatenate((lgP1, lgP2)), np.concatenate((ids1, ids2))
        phases = np.concatenate((phases1, phases2))
        window_title = 'SH0ES2020 + R08 samples analysed by Martino with excitation equilibrium Teff from 5500K, full ' \
                       'Genovali linelist'
        # sys.exit()
    else:
        print('Unknown sample name %s, exiting...' % sample)
        sys.exit()

    return ids, yys, dyys, lgP, phases, label, window_title

# ______________________________________________________________________________________________________________________
