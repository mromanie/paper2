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
def tab2arr(tab, col):
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


def ages():
    """
    Compute ages for the individual stars based on the prescriptions of De Somma et al 2021, submitted to MNRAS.
    The corresponding histogram is plotted.
    :return:
    """
    ids, fes, dfes, lgPs, phases, fe_label, window_title = select(what, 'Fe')

    # log t = a + b log P + c[Fe / H]
    # Canonical Mass-Light
    canonical_coeff = {'a': 8.419, 'b': -0.775, 'c': -0.015, 'sig_a': 0.006, 'sig_b': 0.006, 'sig_c': 0.007,
                       'rms': 0.083}
    # Noncanonical ML
    ncanonical_coeff = {'a': 8.423, 'b': -0.642, 'c': -0.067, 'sig_a': 0.006, 'sig_c': 0.004, 'sig_c': 0.006,
                        'rms': 0.081}

    lgts_canonical = canonical_coeff['a'] + canonical_coeff['b'] * lgPs + canonical_coeff['c'] * fes
    lgts_ncanonical = ncanonical_coeff['a'] + ncanonical_coeff['b'] * lgPs + ncanonical_coeff['c'] * fes

    fig1, ax11 = plt.subplots(figsize=(8, 8))
    fig1.canvas.set_window_title(window_title)
    # fig1.subplots_adjust(top=0.95, bottom=0.2)
    ax11.set_xlabel('log(age) [Myrs]')
    ax11.set_ylabel('Number')
    ax11.set_xlim(6.81, 8.1)

    bins = 10
    ax11.hist(lgts_canonical, bins=bins, zorder=1, color='royalblue', histtype=u'step', linewidth=3)
    ax11.hist(lgts_ncanonical, bins=bins, zorder=5, color='orangered', histtype=u'step', linewidth=3, linestyle='--')


def fig3():
    """
    Reproduce Figure 3 of Riess et al 2019: plot of the PLs in the different bands.
    :return: None
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    set_window_position(fig, 1050, 20)
    fig.canvas.set_window_title('Figure 1: Riess+2019 Figure 3')
    ax.set_ylim(18, 9.5)
    ax.set_xlim(0.75, 1.75)
    ax.set_xlabel('log(P) [days]', size='large', fontsize=20)
    ax.set_ylabel('magnitude', size='large', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)

    labels = ['F555W+1', 'F814W+1', 'F160W+1', '$\mathrm{\mathbf{m}}_\mathrm{\mathbf{H}}^\mathrm{\mathbf{W}}$']
    ylabels = [14.48, 13.35, 12.3, 10.9]

    pl_x = np.array(plt.xlim())
    for band, label, ylabel in zip(bands, labels, ylabels):
        ax.plot(logPs, mags[band] + plt_offsets[band], linestyle='', marker='o', color=plt_colors[band], markersize=12)
        ax.plot(pl_x, pl_slope[band] * pl_x + pl_intercept[band] + plt_offsets[band], color=plt_colors[band],
                linestyle='--')
        ax.annotate(label, (1.6, ylabel), weight='bold', size='large', color=plt_colors[band], fontsize=16)
        res_all = mags[band] - (pl_slope[band] * logPs + pl_intercept[band])
        sigma_all = np.std(res_all)
        res_clip = res_all[
            (ids_mag != 'OGLE0992') & (ids_mag != 'OGLE0847')]  # 2.7sigma outliers identified in the paper
        sigma_clip = np.std(res_clip)
        print(f'PL scatter in band {band}: all {sigma_all:.3f}, clipped {sigma_clip:.3f} [mag] ' \
              f'(paper: {pl_scatter[band]:.3f}).')
    del band


    print()
    # Reconstruct the W(H, RI) PL from the magnitudes in the individual bands read from the paper.
    # Correction for CRNL and for geometric LMC inclination:  mags['WH'] are corrected for LMC line of nodes and CRNL,
    #     the individual magnitudes are not.
    LR_HVIs = ['R19', 'Fitz(Rv=3.3, EBV=1)', 'Fitz(RV=3.3, EBV=0.3)', 'LMC Average(RV=3.41, EBV=0.3)', \
               'LMC Average(RV=3.41, EBV=1)', 'Maiz(RV=3.3, EBV=0.3)']
    R_HVIs = [0.389, 0.415, 0.405, 0.545, 0.452, 0.471]
    iii = 0;
    R_HVI, LR_HVI = R_HVIs[iii], LR_HVIs[iii]
    mags['myWH'] = mags['F160W'] - R_HVI * (mags['F555W'] - mags['F814W']) - geo + 0.0300
    ax.plot(logPs, mags['myWH'] + plt_offsets['WH'], linestyle='', marker='o', color='cyan', markersize=12)
    dWH = mags['myWH'] - mags['WH']
    print('myWH coeff: %s, %.3f' % (LR_HVI, R_HVI))
    print('myWH - WH: mean=%.4f, std=%.4f' % (np.mean(dWH), np.std(dWH)))
    #
    wmags = np.ones(len(emags['WH']))  # / emags['WH']
    fitter = fitting.LinearLSQFitter()
    model = models.Linear1D(slope=pl_slope['WH'])
    model.slope.fixed = True
    # cond = (ids_mag != 'LMC0992') & (ids_mag != 'LMC0847')
    cond = [True] * len(ids_mag)
    xfit, yfit, eyfits, idfit = logPs[(cond)], mags['myWH'][(cond)], emags['WH'][(cond)], ids_mag[(cond)]
    wyfits = wmags[(cond)]
    fitted_line = fitter(model, xfit, yfit, weights=wyfits)
    res_all = yfit - fitted_line(xfit)
    sigma_all = np.std(res_all)
    #
    cond = np.abs(res_all) < 2.7 * sigma_all
    print('Outliers: ', idfit[(~cond)], np.abs(res_all[(~cond)]) / sigma_all)
    fitter = fitting.LinearLSQFitter()
    model = models.Linear1D(slope=pl_slope['WH'])
    model.slope.fixed = True
    xfit, yfit, eyfits, idfit = logPs[(cond)], mags['myWH'][(cond)], emags['WH'][(cond)], ids_mag[(cond)]
    wyfits = wmags[(cond)]
    fitted_line = fitter(model, xfit, yfit, weights=wyfits)
    res_clip = yfit - fitted_line(xfit)
    sigma_clip = np.std(res_clip)
    ax.plot(pl_x, fitted_line(pl_x) + plt_offsets['WH'], linestyle='--', color='cyan')

    print(f'PL scatter in band myWH: all {sigma_all:.3f}, clipped 2.7 sigma {sigma_clip:.3f}')
    intercept = fitter.fit_info['params'][0]
    offset = intercept - pl_intercept['WH']
    print(f'Intercept (here): {intercept:.3f}')
    print(f'Intercept offset (here - paper): {offset:.3f}')

    residuals = res_clip
    ids, fes, dfes, lgP, phases, fe_label, window_title = select(what, 'Fe')
    fig1, ax1 = plt.subplots(figsize=(12, 12))
    set_window_position(fig1, 2, 20)
    fig1.canvas.set_window_title('Figure 2: Magnitude residuals vs Fe')
    # ax.set_ylim(18, 9.5)
    # ax.set_xlim(0.75, 1.75)
    ax1.set_xlabel('[Fe/H]', size='large', fontsize=20)
    ax1.set_ylabel('$\mathrm{m}_\mathrm{H}^\mathrm{W}$ PL residuals [mag]q', size='large', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.errorbar(fes, residuals, xerr=dfes, yerr=eyfits, fmt='o', capsize=5,
                 color='C0', markersize=12)
    # ax1.errorbar([-0.75], [-0.15], xerr=[np.mean(dfes)], yerr=[np.mean(eyfits)], fmt='', capsize=5, mfc='none',
    #              color='C0', markersize=8)
    ax1.axhline(y=0, linewidth=0.5)

    res_gt0 = residuals[(residuals > 0)]
    res_le0 = residuals[(residuals <= 0)]
    print('Number of stars with residuals>0: %i; number with reisduals<=0: %i' % (len(res_gt0), len(res_le0)))

    plt.show()


def PL_met():
    """
    Plot the PL of the stars we have metallicity for.
    :return:
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    set_window_position(fig, 1050, 20)
    fig.canvas.set_window_title('Riess+2019 Figure 3')
    ax.set_ylim(18, 9.5)
    ax.set_xlim(0.75, 1.75)
    ax.set_xlabel('log(P) [days]', size='large', fontsize=20)
    ax.set_ylabel('mag', size='large', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)

    labels = ['F555W+1', 'F814W+1', 'F160W+1', '$\mathrm{\mathbf{m}}_\mathrm{\mathbf{H}}^\mathrm{\mathbf{W}}$']
    yys = [14.48, 13.25, 12.25, 10.82]

    pl_x = np.array(plt.xlim())
    for band, label, yy in zip(bands, labels, yys):
        ax.scatter(logPs[indices_with_Fe], mags[band][indices_with_Fe] + plt_offsets[band], marker='o', s=100,
                   color=plt_colors[band])
        ax.scatter(logPs[indices_without_Fe], mags[band][indices_without_Fe] + plt_offsets[band], marker='o',
                   facecolors='none', edgecolors=plt_colors[band], s=100)
        ax.plot(pl_x, pl_slope[band] * pl_x + pl_intercept[band] + plt_offsets[band], color=plt_colors[band],
                linestyle='--')
        ax.annotate(label, (1.6, yy), weight='bold', size='large', color=plt_colors[band])
        res = mags[band] - (pl_slope[band] * logPs + pl_intercept[band])
        print(f'PL scatter in band {band}: {np.std(res):.2f} [mag] (paper {pl_scatter[band]:.2f}).')
    del band

    plt.show()


def periods():
    '''
    Plot the histogram of pulsational periods
    :param what: what dataset to plot
    :return:
    '''

    # Transform functions for secondary x axis
    def log2lin(x):
        return 10**x
    def lin2log(x):
        return np.log10(x)

    ___, ___, ___, lgP_all, ___, ___, window_title = select('Tot20allRT20M', 'Teff')
    ___, ___, ___, lgP_riess19, ___, ___, window_title = select('Tot20all', 'Teff')
    #
    tab = ascii.read('Files/Ripepi_2021_MNRAS_Tab4.txt', fast_reader=False)
    stars_mode_rip = np.char.replace(tab2arr(tab, 'Star'), '_', '')
    lgP_rip = np.log10(tab2arr(tab, 'P'))
    mode_rip = tab2arr(tab, 'Mode')

    fig1, ax1 = plt.subplots(figsize=(12, 12))
    ax2 = ax1.secondary_xaxis('top', functions=(log2lin, lin2log))
    set_window_position(fig1, 0, 20)
    fig1.canvas.set_window_title(window_title)
    fig1.subplots_adjust(top=0.9, bottom=0.1)
    ax1.set_xlabel('log(P) [days]')
    ax2.set_xlabel('Period [days]')
    ax1.set_ylabel('Number')
    ax1.get_yaxis().set_major_locator(MaxNLocator(integer=True))
    ax2.set_ticks([10, 20, 40, 60, 80, 100])

    ___, bins, ___ = ax1.hist(lgP_all, bins=5, zorder=1, color='royalblue', histtype=u'step', linewidth=3,
                              label='Romaniello+21 (90)')
    ax1.hist(lgP_riess19, bins=bins, zorder=1, color='royalblue', histtype=u'step', linestyle='--', linewidth=3,
             label='Riess+19 (68)')
    ax1.hist(lgP_rip[(mode_rip == 'DCEP_F')], bins=bins, zorder=1, color='orangered', histtype=u'step',
             linestyle='-', linewidth=3, label='Ripepi+21 FU')

    ax1.legend()

    # print(10**np.min(lgP_riess19), 10**(np.min(lgP_all)), len(lgP_rip[(mode_rip == 'DCEP_F')]))

def phase_teff(what):
    """
    Plot the measure Teff vs pulsational phase
    :param what:
    :return:
    """
    '''
    tab = ascii.read('Files//Users/mromanie/LACES/data/Kurucz/Comparisons/SaraDesktop/synthmags/mag_hst_filters.dat')
    ids_ph, phases = tab['id'].data, tab['phase'].data
    phases = phases[np.argsort(ids_ph)]  # Make sure that the order is alphabetical by star ID
    '''

    ids, tes, dtes, lgP, phases, te_label, window_title = select(what, 'Teff')
    ___, loggs, dloggs, ___, ___, logg_label, ___ = select(what, 'logg')
    ___, vturbs, dvturbs, ___, ___, vturb_label, ___ = select(what, 'vturb')

    # fig1 = plt.figure(figsize=(25, 8))
    fig1 = plt.figure(figsize=(10, 10))
    set_window_position(fig1, 0, 20)
    fig1.canvas.set_window_title('Figure 1: ' + window_title)
    # fig1.subplots_adjust(left=0.05, right=0.95, top=0.975, bottom=0.1)
    # gs1 = fig1.add_gridspec(nrows=1, ncols=3)
    fig1.canvas.set_window_title('Figure 1: ' + window_title)
    # ax11 = fig1.add_subplot(gs1[0, 0])
    ax11 = fig1.add_subplot()

    ax11.errorbar(phases, tes, yerr=dtes, fmt='o', capsize=5, markersize=12)
    # ax11.errorbar([0.], [5000], yerr=[np.mean(dtes)], fmt='o', capsize=5, mfc='none', markersize=12)
    ax11.set_xlabel('Phase', fontsize=20)
    ax11.set_ylabel(latex(te_label, '\mathrm'), fontsize=20)
    ax11.tick_params(axis='both', which='major', labelsize=16)

    # ax12 = fig1.add_subplot(gs1[0, 1])
    # ax12.scatter(phases, loggs)
    # ax12.set_xlabel('Phase', fontsize=20)
    # ax12.set_ylabel(latex(logg_label, '\mathrm'), fontsize=20)
    # ax12.tick_params(axis='both', which='major', labelsize=16)
    #
    # ax13 = fig1.add_subplot(gs1[0, 2])
    # ax13.scatter(phases, vturbs)
    # ax13.set_xlabel('Phase', fontsize=20)
    # ax13.set_ylabel(latex(vturb_label, '\mathrm'), fontsize=20)
    # ax13.tick_params(axis='both', which='major', labelsize=16)


def R08_Tot20all():
    """
    Plot data from Romaniello+2008 vs the 2020 data
    :return:
    """
    ids_r08, fes_r08, dfes_r08, lgP_r08, phases_r08, fe_label_r08, ___ = select('R08', 'Fe')
    ___, fes_Tot20all, dfes_Tot20all, ___, ___, ___, ___ = select('Tot20all', 'Fe')

    fig1 = plt.figure(figsize=(10, 10))
    set_window_position(fig1, 550, 20)
    # fig1.subplots_adjust(left=0.15, right=0.95, top=0.975, bottom=0.05)
    # gs1 = fig1.add_gridspec(nrows=3, ncols=1)
    fig1.canvas.set_window_title('Figure 1: R08 vs current analysis')
    ax11 = fig1.add_subplot()
    ax11.hist(fes_Tot20all, bins=20, histtype=u'step', zorder=5)
    ax11.hist(fes_r08, bins=10, zorder=3, linestyle='--', histtype=u'step')
    ax11.set_xlabel(latex(fe_label_r08, '\mathrm'), fontsize=20)
    ax11.tick_params(axis='both', which='major', labelsize=16)

    print('rms Tot20all', np.std(fes_Tot20all))
    print('rms R08', np.std(fes_r08))


def compare_R08():
    """
    Compare the original R08 parameters with those from the 2020 reanalysis.
    :return:
    """
    what = 'R08'  # R08 sample straight from the paper
    ids_r08, fes_r08, dfes_r08, lgP_r08, phases_r08, fe_label, window_title_r08 = select(what, 'Fe')
    ___, tes_r08, dtes_r08, ___, ___, te_label, ___ = select(what, 'Teff')
    #
    what = 'R20M'  # R08 sample analysed by Martino with 2020 LDR Teff
    ids_r20m, fes_r20m, dfes_r20m, lgP_r20m, phases_r20m, ___, window_title_r20m = select(what, 'Fe')
    ___, tes_r20m, dtes_r20m, ___, ___, ___, ___ = select(what, 'Teff')
    #
    what = 'RT20M'  # R08 analysed by Martino w/ excitation balance Teff
    ids_tr20m, fes_tr20m, dfes_tr20m, lgP_tr20m, phases_tr20m, ___, window_title_tr20m = select(what, 'Fe')
    ___, tes_tr20m, dtes_tr20m, ___, ___, ___, ___ = select(what, 'Teff')
    #
    what = 'R20M08'  # R08 sample analysed by Martino with 2008 LDR Teff
    ids_r20m08, fes_r20m08, dfes_r20m08, lgP_r20m08, phases_r20m08, ___, window_title_r20m08 = select(what, 'Fe')
    ___, tes_r20m08, dtes_r20m08, ___, ___, ___, ___ = select(what, 'Teff')

    fig1 = plt.figure(figsize=(16, 7))
    set_window_position(fig1, 50, 20)
    fig1.subplots_adjust(left=0.15, right=0.95, top=0.975, bottom=0.1)
    gs1 = fig1.add_gridspec(nrows=1, ncols=2)
    fig1.canvas.set_window_title('Figure 1 ... R08: excitation balance 2020 (Martino) vs paper')
    ax11 = fig1.add_subplot(gs1[0, 0])
    ax12 = fig1.add_subplot(gs1[0, 1])
    #
    ax11.scatter(fes_tr20m, fes_tr20m - fes_r08)
    ax11.set_xlabel(fe_label + '   ' + window_title_tr20m, fontsize=20)
    ax11.set_ylabel(window_title_tr20m + '\n' + window_title_r08, fontsize=20)
    ax11.tick_params(axis='both', which='major', labelsize=16)
    #
    ax12.scatter(tes_tr20m, tes_tr20m - tes_r08)
    ax12.set_xlabel(latex(te_label, '\mathrm') + '   ' + window_title_tr20m, fontsize=20)
    ax12.set_ylabel(window_title_tr20m + '\n' + window_title_r08, fontsize=20)
    ax12.tick_params(axis='both', which='major', labelsize=16)

    fig2 = plt.figure(figsize=(16, 7))
    set_window_position(fig2, 50, 2000)
    fig2.subplots_adjust(left=0.15, right=0.95, top=0.975, bottom=0.1)
    gs2 = fig2.add_gridspec(nrows=1, ncols=2)
    fig2.canvas.set_window_title('Figure 2 ... R08: excitation balance 2020 vs LDR 2020 (Martino)')
    ax21 = fig2.add_subplot(gs2[0, 0])
    ax22 = fig2.add_subplot(gs2[0, 1])
    #
    ax21.scatter(fes_tr20m, fes_tr20m - fes_r20m)
    ax21.set_xlabel(fe_label + '   ' + window_title_tr20m, fontsize=20)
    ax21.set_ylabel(window_title_tr20m + '\n' + window_title_r20m, fontsize=20)
    ax21.tick_params(axis='both', which='major', labelsize=16)
    #
    ax22.scatter(tes_tr20m, tes_tr20m - tes_r20m)
    ax22.set_xlabel(latex(te_label, '\mathrm') + '   ' + window_title_tr20m, fontsize=20)
    ax22.set_ylabel(window_title_tr20m + '\n' + window_title_r20m, fontsize=20)
    ax22.tick_params(axis='both', which='major', labelsize=16)

    fig3 = plt.figure(figsize=(16, 7))
    set_window_position(fig3, 1650, 20)
    fig3.subplots_adjust(left=0.15, right=0.95, top=0.975, bottom=0.1)
    gs3 = fig3.add_gridspec(nrows=1, ncols=2)
    fig3.canvas.set_window_title('Figure 3 ... R08: LDR 2008 vs LDR 2020 (Martino)')
    ax31 = fig3.add_subplot(gs3[0, 0])
    ax32 = fig3.add_subplot(gs3[0, 1])
    #
    ax31.scatter(fes_r08, fes_r08 - fes_r20m08)
    ax31.set_xlabel(fe_label + '   ' + window_title_r08, fontsize=20)
    ax31.set_ylabel(window_title_r08 + '\n' + window_title_r20m08, fontsize=20)
    ax31.tick_params(axis='both', which='major', labelsize=16)
    #
    ax32.scatter(tes_r08, tes_r08 - tes_r20m08)
    ax32.set_xlabel(latex(te_label, '\mathrm') + '   ' + window_title_r08, fontsize=20)
    ax32.set_ylabel(window_title_r08 + '\n' + window_title_r20m08, fontsize=20)
    ax32.tick_params(axis='both', which='major', labelsize=16)


def vsP(what, orientation='vertical'):
    """
    Plot the measured metallicity/temperature vs the pulsational period.
    :return:
    """
    ids, fes, dfes, lgP, phases, fe_label, window_title = select(what, 'Fe')
    ___, tes, dtes, ___, ___, te_label, ___ = select(what, 'Teff')
    ___, logtes, dlogtes, ___, ___, logte_label, ___ = select(what, 'logTeff')
    ___, loggs, dloggs, ___, ___, logg_label, ___ = select(what, 'logg')
    ___, vturbs, dvturbs, ___, ___, vturb_label, ___ = select(what, 'vturb')
    lims = [(4610, 6760), (-0.86, 0.23)]

    print('\n' + color.BOLD + color.CYAN + window_title + color.END)

    # Fit a plane
    XX = np.array([[t, g, v] for t, g, v in zip(tes, loggs, vturbs)])
    YY = fes
    plane = LinearRegression().fit(XX, YY, sample_weight=dfes)
    plane_residuals = YY - plane.predict(XX)
    std = np.std(YY)
    std_detrended = np.std(plane_residuals)
    std_sqrt_diff = np.sqrt(std ** 2 - std_detrended ** 2)
    print('\n[Fe/H] dispersion: %.3f ; 4D de-trended: %.3f (sqrt-diff: %.3f, mean of residuals %.2e)' %
          (std, std_detrended, std_sqrt_diff, np.mean(plane_residuals)))
    print('       error: mean %.3f, median %.3f' % (np.mean(dfes), np.median(dfes)))

    if orientation == 'vertical':
        fig1 = plt.figure(figsize=(8, 13))
        set_window_position(fig1, 1550, 20)
        fig1.subplots_adjust(left=0.125, right=0.95, top=0.975, bottom=0.05)
        gs1 = fig1.add_gridspec(nrows=3, ncols=1)
        yticks = True
        plot_regression = True
    elif orientation == 'horizontal':
        fig1 = plt.figure(figsize=(16, 7))
        set_window_position(fig1, 1550, 20)
        fig1.subplots_adjust(left=0.065, right=0.975, top=0.975, bottom=0.125)
        gs1 = fig1.add_gridspec(nrows=1, ncols=3, wspace=0.075)
        yticks = False
        plot_regression = False
    #
    fig1.canvas.set_window_title('Figure 1: ' + window_title)
    ax11 = fig1.add_subplot(gs1[0])
    ax12 = fig1.add_subplot(gs1[1])
    ax13 = fig1.add_subplot(gs1[2])
    #
    plot_regression = False
    plot_fit(ax11, tes, fes, dfes, te_label, fe_label, dxxs=dtes, second_slope=plane.coef_[0], phases=phases,
             plot_regression=plot_regression)
    plot_fit(ax12, loggs, fes, dfes, logg_label, fe_label, dxxs=dloggs, second_slope=plane.coef_[1], phases=phases,
             yticklabels=yticks, plot_regression=plot_regression)
    plot_fit(ax13, vturbs, fes, dfes, vturb_label, fe_label, dxxs=dvturbs, second_slope=plane.coef_[2], phases=phases,
             yticklabels=yticks, colorbar=True, orientation=orientation, plot_regression=plot_regression)

    fig4 = plt.figure(figsize=(8, 13))
    set_window_position(fig4, 2350, 20)
    fig4.subplots_adjust(left=0.15, right=0.95, top=0.975, bottom=0.05)
    gs4 = fig4.add_gridspec(nrows=3, ncols=1)
    fig4.canvas.set_window_title('Figure 4: ' + window_title)
    ax41 = fig4.add_subplot(gs4[0, 0])
    ax42 = fig4.add_subplot(gs4[1, 0])
    ax43 = fig4.add_subplot(gs4[2, 0])
    #
    fes_detrended = np.mean(fes) + plane_residuals
    plot_fit(ax41, tes, fes_detrended, dfes, te_label, fe_label + '\ detrended', phases=phases)
    plot_fit(ax42, loggs, fes_detrended, dfes, logg_label, fe_label + '\  detrended', phases=phases)
    plot_fit(ax43, vturbs, fes_detrended, dfes, vturb_label, fe_label + '\  detrended', phases=phases)

    fig2 = plt.figure(figsize=(8.5, 13))
    fig2.subplots_adjust(left=0.14, right=0.935, top=0.975, bottom=0.075)
    gs2 = fig2.add_gridspec(nrows=4, ncols=1, hspace=0., wspace=0.)
    set_window_position(fig2, 750, 20)
    fig2.canvas.set_window_title('Figure 2: ' + window_title)
    ax21 = fig2.add_subplot(gs2[0, 0])
    ax22 = fig2.add_subplot(gs2[1, 0], sharex=ax21)
    ax23 = fig2.add_subplot(gs2[2, 0], sharex=ax21)
    ax24 = fig2.add_subplot(gs2[3, 0], sharex=ax21)
    #
    ax21.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax22.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax23.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    # Secondary y axis with linear Teff scale
    ax22_secax = ax22.secondary_yaxis('right', functions=(log2lin, lin2log))
    for tick in ax22_secax.get_yticklabels():
        tick.set_rotation(90)
    #   print cursor position for both y axis
    ax22.format_chrdoord = make_format(ax22, ax22_secax)
    #
    plot_fit(ax21, lgP, fes, dfes, 'Log(P) [days]', fe_label, phases=phases)
    plot_fit(ax22, lgP, logtes, dlogtes, 'Log(P) [days]', logte_label, phases=phases)
    plot_fit(ax23, lgP, loggs, dloggs, 'Log(P) [days]', logg_label, phases=phases)
    plot_fit(ax24, lgP, vturbs, dvturbs, 'Log(P) [days]', vturb_label, phases=phases, colorbar=True)

    fig3 = plt.figure(figsize=(8, 13))
    fig3.subplots_adjust(top=0.975, bottom=0.05)
    gs3 = fig3.add_gridspec(nrows=3, ncols=1)
    set_window_position(fig3, 0, 20)
    fig3.canvas.set_window_title('Figure 3: ' + window_title)
    ax31 = fig3.add_subplot(gs3[0, 0])
    ax32 = fig3.add_subplot(gs3[1, 0])
    ax33 = fig3.add_subplot(gs3[2, 0])
    #
    plot_fit(ax31, tes, loggs, dloggs, te_label, logg_label, phases=phases)
    plot_fit(ax32, tes, vturbs, dvturbs, te_label, vturb_label, phases=phases)
    plot_fit(ax33, loggs, vturbs, dvturbs, logg_label, vturb_label, phases=phases)


def plot_fit(ax, xxs, yys, dyys, xx_label, yy_label, dxxs=None, lims=None, second_slope=None, yticklabels=True,
             phases=None, colorbar=False, plot_regression=True, orientation='vertical'):
    ax.set_xlabel(latex(xx_label, '\mathrm'), fontsize=20)
    # ax.set_ylabel(latex(yy_label, '\mathrm'), fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=16)
    if not yticklabels:
        ax.set_yticklabels([])
        pass
    else:
        ax.set_ylabel(latex(yy_label, '\mathrm'))  # , fontsize=26, fontweight='bold')

    if lims is not None:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])

    # colormap = cm.copper
    # colormap = cm.Purples
    colormap = cm.viridis
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    if dxxs is None:
        dxxs = np.zeros(len((xxs)))
    if phases is None:
        color = 'C0'
    else:
        color = phases
    ax.errorbar(xxs, yys, yerr=dyys, xerr=dxxs, fmt='.', markersize=1, capsize=3, color='b', zorder=3)
    ax.scatter(xxs, yys, c=color, cmap=colormap, norm=norm, s=12**2, marker='o', zorder=5)
    ax.scatter(xxs, yys, facecolor='none', edgecolor='b', s=14 ** 2, marker='o', zorder=5)

    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        # sm.set_array([])
        if orientation == 'vertical':
            ybar = 1.95
        elif orientation == 'horizontal':
            ybar = 1
        cax = ax.inset_axes([1.005, 0., 0.025, ybar], transform=ax.transAxes)
        cb = plt.gcf().colorbar(sm, ticks=None, ax=ax, cax=cax)
        cb.ax.set_yticklabels(['{:.1f}'.format(x) for x in cb.ax.get_yticks()], fontsize=8, weight='bold', rotation=90)

    # Linear regression, weighted and not
    popt_sigma, pcov_sigma = curve_fit(func_line, xxs, yys, (0, np.mean(yys)), sigma=dyys, absolute_sigma=True)
    popt_nosigma, pcov_nosigma = curve_fit(func_line, xxs, yys, (0, np.mean(yys)))
    if plot_regression:
        ax.axline((np.mean(xxs), np.mean(yys)), slope=popt_sigma[0], color='C0')
        # ax.axline((np.mean(xxs), np.mean(yys)), slope=popt_nosigma[0], color='C0', linestyle='--')
    if second_slope is not None:
        ax.axline((np.mean(xxs), np.mean(yys)), slope=second_slope, color='red')

    print('Weighed fit: %s = %.2e(±%.2e) * %s %+.2e(±%.2e)' % (deLatex(yy_label),
                                                               popt_sigma[0], np.sqrt(np.diag(pcov_sigma)[0]),
                                                               deLatex(xx_label),
                                                               popt_sigma[1], np.sqrt(np.diag(pcov_sigma)[1])))
    print('Non-weighed fit: %s = %.2e(±%.2e) * %s %+.2e(±%.2e)' % (deLatex(yy_label),
                                                                   popt_nosigma[0], np.sqrt(np.diag(pcov_nosigma)[0]),
                                                                   deLatex(xx_label),
                                                                   popt_nosigma[1], np.sqrt(np.diag(pcov_nosigma)[1])))

    std = np.std(yys)
    std_detrended = np.std(yys - func_line(xxs, *popt_nosigma))
    std_sqrt_diff = np.sqrt(std ** 2 - std_detrended ** 2)
    print('%s dispersion: %.3f ; de-trended: %.3f; sqrt-diff: %.3f' %
          (deLatex(yy_label), std, std_detrended, std_sqrt_diff))

    # Spearmann's correlation coefficients
    spear_rho, spear_p = stats.spearmanr(xxs, yys)
    print('Spearman: rho=%.3f, p=%.1f%%' % (spear_rho, 100 * spear_p))
    print()

    # plt.show()


def histograms(what, element):
    """
    Plot a histogram of the measured Fe abundances, together with a Gaussian fit.
    :return:
    """

    ids, yys, dyys, lgP, phases, label, window_title = select(what, element)
    ids = ids[(dyys > 0)]
    yys = yys[(dyys > 0)]
    lgP = lgP[(dyys > 0)]
    dyys = dyys[(dyys > 0)]

    print('\n' + color.BOLD + color.YELLOW + window_title + color.END)

    argmin_xx, argmax_xx = np.argmin(yys), np.argmax(yys)
    cumul_xx = np.arange(yys[argmin_xx] - 3 * dyys[argmin_xx], yys[argmax_xx] + 4 * dyys[argmax_xx],
                         ((3 * dyys[argmax_xx] + 4 * dyys[argmin_xx]) / 100))
    cumul_yy = np.zeros(len(cumul_xx))

    # __________________________________________________________________________________________________________________
    #
    fig1, ax11 = plt.subplots(figsize=(12, 12))
    for yy, dyy in zip(yys, dyys):
        ampl = 1 / (dyy * np.sqrt(2 * np.pi))
        yyg = func_gauss(cumul_xx, ampl, yy, dyy)
        cumul_yy += yyg
        ax11.plot(cumul_xx, yyg, linewidth=1, color='grey', zorder=1)
        ax11.scatter([yy], [ampl], marker='o', linewidth=2, facecolors='none', edgecolors='black', s=20, zorder=1)
    del (yy, dyy, yyg)

    # Plot the cumulative distribution, also normalised to an integral of 1.
    cumul_yy = cumul_yy / np.trapz(cumul_yy, x=cumul_xx)
    ax11.plot(cumul_xx, cumul_yy, linewidth=3, color='r', zorder=5)
    cumul_mean, cumul_std = weighted_avg_and_std(cumul_xx, cumul_yy)
    cumul_percentile = np.percentile(cumul_yy, [33, 66])
    print('Figure 1 ... %s cumulative distribution: mean=%.3f, std=%.3f, 66-th percentile=%.2f '
          % (deLatex(label), cumul_mean, cumul_std, (cumul_percentile[1] - cumul_percentile[0]) / 2))

    # Secondary y axis on the right with the values of the errors on Fe (dFe) ...
    sigma_to_peak = peak_to_sigma  # 1 / x is its own inverse
    secay = ax11.secondary_yaxis('right', functions=(peak_to_sigma, sigma_to_peak))
    # ... meaningful labels
    # https://stackoverflow.com/questions/5426908/find-unique-elements-of-floating-point-array-in-numpy-with-comparison-using-a-d
    dyys_unique = np.unique(dyys.round(decimals=2))
    secay.set_yticks([round(x, 2) for x in dyys_unique])

    yy_cumul_gauss = func_gauss(cumul_xx, 1 / (cumul_std * np.sqrt(2 * np.pi)), cumul_mean, cumul_std)
    ax11.plot(cumul_xx, yy_cumul_gauss, color='r', linewidth=3, linestyle='--')
    ax11.axhline(y=np.max(yy_cumul_gauss), color='r', linewidth=0.5)

    fig1.canvas.set_window_title('Figure 1: ' + window_title)
    ax11.set_xlabel(latex(label, '\mathrm'), fontsize=20)
    ax11.set_ylabel('Distribution Function', fontsize=20)
    secay.set_ylabel('Gaussian sigma', fontsize=20)
    ax11.tick_params(axis='both', which='major', labelsize=16)
    secay.yaxis.set_tick_params(labelsize=15)
    set_window_position(fig1, 0, 20)

    # __________________________________________________________________________________________________________________
    fig2, ax21 = plt.subplots(figsize=(8, 4))
    plt.subplots_adjust(top=0.95, bottom=0.2)
    # Histogram of the [yy/H] distribution ...
    if element == 'Fe':
        if 'SMC' in what:
            ax21.set_xlim([-1.2, -0.6])
        else:
            ax21.set_xlim([-0.85, -0.05])
    elif element == 'O':
        ax21.set_xlim([-0.85, 0.1])
    #
    if 'SMC' in what:
        bin1, bin2 = -1.5, 0.05
    else:
        bin1, bin2 = -0.9, 0.05

    bins = np.arange(bin1, bin2, (bin2 - bin1) / 13)
    # bins = 12
    nums_yy, edges_yy, ___ = ax21.hist(yys, bins=bins, zorder=1, color='royalblue')
    centers_yy = (edges_yy[:-1] + edges_yy[1:]) / 2
    yy_plain_mean, yy_plain_std = np.mean(yys), np.std(yys)
    # ... and Gaussian fit to it
    try:
        ax21.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))  # Force integer ticks
        popt, pcov = curve_fit(func_gauss, centers_yy, nums_yy, (np.max(nums_yy), yy_plain_mean, yy_plain_std))
        pstd = np.sqrt(np.diag(pcov))
        yy_gauss_mean, yy_gauss_sigma = popt[1], popt[2]
        dyy_gauss_mean, dyy_gauss_sigma = pstd[1], pstd[2]
        xx_gauss = np.arange(ax21.get_xlim()[0], ax21.get_xlim()[1], (ax21.get_xlim()[1] - ax21.get_xlim()[0]) / 100)
        yy_gauss = func_gauss(xx_gauss, *popt)
        ax21.plot(xx_gauss, yy_gauss, zorder=5, color='r')
        ax21.axvline(x=yy_gauss_mean, color='r')
        ax21.axvline(x=yy_gauss_mean + yy_gauss_sigma, color='r', linestyle='--')
        ax21.axvline(x=yy_gauss_mean - yy_gauss_sigma, color='r', linestyle='--')
        #
        ax11.plot(xx_gauss, yy_gauss / np.trapz(yy_gauss, x=xx_gauss), color='b', linewidth=3,
                  linestyle='-.', zorder=5)
        gauss_success = True
    except Exception as e:
        print('Figure 2 ... Exception: ', e)
        gauss_success = False
    # ... R08 Fe distribution in the LMC
    # nums_Fe_r08, edges_Fe_r08, ___ = ax21.hist(np.repeat(Fes_r08, np.int(len(Fes)/len(Fes_r08))), 10,
    #                                                      zorder=1, color='g', facecolor='None', edgecolor='black')

    y_annotate = ax11.get_ylim()[1]
    for edge_yy, center_yy, num_yy in zip(edges_yy, centers_yy, nums_yy):
        ax11.axvline(x=edge_yy, color='r', linewidth=1)
        ax11.annotate(np.int(num_yy), (center_yy, 0.975 * y_annotate), ha='center')
    ax11.axvline(x=edges_yy[-1], color='r', linewidth=1)
    del (edge_yy, center_yy)

    if gauss_success:
        print('Figure 2 ... %s stats: Gaussian (direct) mean=%.3f±%.3f (%.3f), sigma=%.3f±%.3f (%.3f)' %
              (deLatex(label), yy_gauss_mean, dyy_gauss_mean, yy_plain_mean, yy_gauss_sigma, dyy_gauss_sigma,
               yy_plain_std))
    fig2.canvas.set_window_title('Figure 2: ' + window_title)
    ax21.set_xlabel(latex(label, '\mathrm'), fontsize=20)
    ax21.set_ylabel('Number', fontsize=20)
    ax21.tick_params(axis='both', which='major', labelsize=16)
    set_window_position(fig2, 1200, 20)

    # __________________________________________________________________________________________________________________
    fig3, ax31 = plt.subplots(figsize=(8, 4))
    hist_dyy, edges_dyy, ___ = ax31.hist(dyys, 15, zorder=1, color='royalblue')
    centers_dyy = (edges_dyy[:-1] + edges_dyy[1:]) / 2
    dyy_mean, dyy_median, dyy_mode = np.mean(dyys), np.median(dyys), centers_dyy[np.argmax(hist_dyy)]
    print('Figure 3 ... d%s stats: mean=%.3f, median=%.3f, mode=%.3f' % (deLatex(label), dyy_mean, dyy_median,
                                                                         dyy_mode))
    ax31.axvline(x=dyy_mean, color='limegreen')
    ax31.axvline(x=dyy_median, color='limegreen')
    ax31.axvline(x=dyy_mode, color='limegreen')
    fig3.canvas.set_window_title('Figure 3: ' + window_title)
    ax31.set_xlabel('d' + latex(label, '\mathrm'), fontsize=20)
    ax31.set_ylabel('Number', fontsize=20)
    ax31.tick_params(axis='both', which='major', labelsize=16)
    set_window_position(fig3, 1200, 480)

    # Mark ± the typical uncertainty in the histogram in Figure 2
    if gauss_success:
        if what[:1] != 'R':  # For the analyses of the R08 sample, do not mark the errors, which aren't meaningful
            ax21.axvline(x=yy_gauss_mean - dyy_mean, color='limegreen', linestyle='--')
            ax21.axvline(x=yy_gauss_mean + dyy_mean, color='limegreen', linestyle='--')
            # ax21.axvline(x=yy_gauss_mean - dyy_median, color='limegreen', linestyle='--')
            # ax21.axvline(x=yy_gauss_mean + dyy_median, color='limegreen', linestyle='--')
            # ax21.axvline(x=yy_gauss_mean - dyy_mode, color='limegreen', linestyle='--')
            # ax21.axvline(x=yy_gauss_mean + dyy_mode, color='limegreen', linestyle='--')
        pass
    # __________________________________________________________________________________________________________________
    if gauss_success:
        fig4, ax41 = plt.subplots(figsize=(8, 4.5))
        fig4.canvas.set_window_title('Figure 4: ' + window_title)
        # ax41.set_xlabel(latex(label) + '- $\overline{\mathrm{' + label + '}}$', fontsize=20)
        ax41.set_xlabel(latex(label, '\mathrm') + ' - ' + latex(label, '\overline\mathrm'), fontsize=20)
        ax41.set_ylabel('d' + r'$\mathrm{' + label + '}$', fontsize=20)
        ax41.tick_params(axis='both', which='major', labelsize=16)
        plt.scatter(yys - yy_gauss_mean, dyys, marker='o', color='royalblue')
        ax41.axhline(y=dyy_mean, color='limegreen');
        ax41.axhline(y=dyy_median, color='limegreen')
        ax41.axhline(y=dyy_mode, color='limegreen')
        set_window_position(fig4, 1200, 940)

    # plt.show()


def h_residuals():
    # Input files provided by Adam
    tab = ascii.read('Files/H_residualsR08.txt')
    old_fes, dold_fes = np.array(tab['Old_Fe/H']), np.array(tab['errO'])
    new_fes, dnew_fes = np.array(tab['New_Fe/H']), np.array(tab['errN'])
    h_ress, dh_ress = np.array(tab['H-band_dereddened_PL_residual']), np.array(tab['err'])
    #
    tab = ascii.read('Files/H_residualsR21.txt')
    new_fes68, dnew_fes68 = np.array(tab['Fe/H']), np.array(tab['err'])
    h_ress68, dh_ress68 = np.array(tab['H-band_dereddened_PL_residual']), np.array(tab['error'])

    fig1 = plt.figure(figsize=(16, 7))
    set_window_position(fig1, 1550, 20)
    fig1.subplots_adjust(left=0.065, right=0.975, top=0.975, bottom=0.125)
    gs1 = fig1.add_gridspec(nrows=1, ncols=3, wspace=0.075)
    yticks = False
    plot_regression = False
    #
    fig1.canvas.set_window_title('Figure 1: H-band residuals ve iron')
    ax11 = fig1.add_subplot(gs1[0])
    ax12 = fig1.add_subplot(gs1[1])
    ax13 = fig1.add_subplot(gs1[2])
    #
    ax12.set_yticklabels([])
    ax13.set_yticklabels([])
    #
    xlabel = '[Fe/H] [dex]'
    ax11.set_xlabel(xlabel)
    ax12.set_xlabel(xlabel)
    ax13.set_xlabel(xlabel)
    ax11.set_ylabel('H-band, dereddened PL residuals')
    #
    ax11.annotate('(a)', (0.85, 0.9), xycoords='axes fraction')
    ax12.annotate('(b)', (0.85, 0.9), xycoords='axes fraction')
    ax13.annotate('(c)', (0.85, 0.9), xycoords='axes fraction')

    def plot(ax, x, y, dx, dy):
        ax.errorbar(x, y, xerr=dx, yerr=dy, marker='o', markersize=12, linestyle='', capsize=3, zorder=5)
        ax.axhline(y=0, linewidth=1)
        ax.axline((np.mean(x), np.mean(y)), slope=-0.2, color='red', linewidth=3, zorder=3)

    plot(ax11, old_fes, h_ress, dold_fes, dh_ress)
    plot(ax12, new_fes, h_ress, dnew_fes, dh_ress)
    plot(ax13, new_fes68, h_ress68, dnew_fes68, dh_ress68)

# ______________________________________________________________________________________________________________________
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('what_plot', help='What plot to plot', type=str)
    args = parser.parse_args()

    bands = ['F555W', 'F814W', 'F160W', 'WH']
    plt_offsets = {'F555W': 1, 'F814W': 1, 'F160W': 1, 'WH': 0}
    plt_colors = {'F555W': 'b', 'F814W': 'g', 'F160W': 'purple', 'WH': 'r'}

    # Read the inputs from file
    ids_mag, logPs, mags, emags, geo, pl_slope, pl_intercept, pl_scatter = read_riess()

    stellar_parameters, indices_with_Fe, indices_without_Fe = read_parameters('SH0ES_atmparam_all.dat')  # Sara LDR
    stellar_parameters_alllines, __, __ = read_parameters('SH0ES_atmparam_alllines_all.dat')  # Sara LDR, all Genovali lines

    stellar_parameters_r08, intrinsic_parameters_r08 = read_r08('LMC')  # R08 from the paper
    stellar_parameters_r08smc, intrinsic_parameters_r08smc = read_r08('SMC')  # R08 from the paper

    stellar_parameters08, __, __ = read_parameters('LMC_R08_atmparam.dat')  # R08 Sara LDR
    stellar_parametersT08, __, __ = read_parameters('LMC_R08_atmparam_FREET.dat')  # R08 Sara excitation balance
    stellar_parametersT08M, __, __ = read_parameters(
        'LMC_R08_atmparam_FREETotal_alllines_all.dat')  # R08 Martino exc balance ... preferred
    stellar_parameters08M, __, __ = read_parameters('LMC_R08_atmparam_alllines_all.dat')  # R08 Martino LDR
    stellar_parameters0808M, __, __ = read_parameters(
        'LMC_R08R08_atmparam_alllines_all.dat')  # R08 Martino LDR as in 2008 paper

    stellar_parametersT08MSMC, __, __ = read_parameters(
        'SMC_R08_atmparam_FREETotal_alllines_all.dat')  # R08 Martino exc balance

    stellar_parametersA, __, __ = read_parameters('lmc_uves_results_v1.dat')  # Alessio, exc balance
    stellar_parametersT, __, __ = read_parameters('SH0ES_atmparam_FREET_all.dat')  # Sara, exc balance from LDR

    stellar_parametersT_alllines, __, __ = read_parameters(
        'SH0ES_atmparam_FREET_alllines_all.dat')  # Martino exc balance from LDR
    ### stellar_parametersT_alllines, __, __ = read_parameters('SH0ES_atmparam_FREET_alllines_all_sara.dat')  # Sara

    stellar_parametersTotal_alllines, __, __ = \
        read_parameters('SH0ES_atmparam_FREETotal_alllines_all.dat')  # Martino exc balance from 5500 K ... preferred
    ### stellar_parametersTotal_alllines, __, __ = read_parameters('SH0ES_atmparam_FREETotal_alllines_all_sara.dat')  # Sara

    # what = 'S20'  # SH0ES2020 sample analysed by Sara with 2020 LDR Teff
    # what = 'T20'  # SH0ES2020 sample analysed by Sara with exc balance
    # what = 'S20all'  # SH0ES2020 sample analysed by Sara with 2020 LDR Teff, full Genovali linelist
    # what = 'T20all'  # SH0ES2020 sample analysed by Martino  w/ exc balance Teff from LDR, full Genovali linelist
    # what = 'Tot20all'  # SH0ES sample analysed by Martino w/ exc balance Teff, full Genovali linelist ### Preferred
    #
    # what = 'R08'  # R08 sample straight from the paper
    # what = 'R20'  # R08 sample analysed by Sara with 2020 LDR Teff
    # what = 'RT20'  # R08 analysed by Sara w/ exc balance Teff
    # what = 'RT20M'  # R08 analysed by Martino w/ exc balance Teff, all Genovali lines  ... Preferred
    # what = 'R20M'  # R08 analysed by Martino w/ 2020 LDR Teff, all Genovali lines
    # what = 'R20M08'  # R08 analysed by Martino w/ 2020 LDR Teff from the original 2008 paper, all Genovali lines
    #
    # what = 'RT20MSMC'  # SMC R08 analysed by Martino w/ exc balance Teff, all Genovali lines
    #
    what = 'Tot20allRT20M'  # SH0ES2020 + R08 samples analysed by Martino w/ exc balance Teff, full Genovali linelist
    #
    # what = 'A20'  # SHOES sample analysed by Alessio

    plt.style.use('paper')

    if args.what_plot == 'fig3':
        fig3()
        plt.show()
    elif args.what_plot == 'PL_met':
        PL_met()
    elif args.what_plot == 'vsP':
        vsP(what, orientation='horizontal')
        # vsP(what)
        plt.show()
    elif args.what_plot == 'histograms':
        histograms(what, 'Fe')
        plt.show()
    elif args.what_plot == 'both':
        vsP(what)
        histograms(what, 'Fe')
        plt.show()
    elif args.what_plot == 'phase_teff':
        phase_teff(what)
        plt.show()
    elif args.what_plot == 'R08_Tot20all':
        R08_Tot20all()
        plt.show()
    elif args.what_plot == 'compare_R08':
        compare_R08()
        plt.show()
    elif args.what_plot == 'ages':
        ages()
        plt.show()
    elif args.what_plot == 'hrd':
        hrd()
        plt.show()
    elif args.what_plot == 'h_residuals':
        h_residuals()
        plt.show()
    elif args.what_plot == 'periods':
        periods()
        plt.show()
