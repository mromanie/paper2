import matplotlib

# matplotlib.use('Qt5Agg')  # Force a backend that supports specifying the location of the plot window

from astropy.io import ascii
import numpy as np
from matplotlib import pyplot as plt
import sys, importlib, argparse


def deLatex(instring):
    """
    Remove the LaTex markups from the input string
    :param instring: input string to be de-LaTeX-ed.
    :return: de-LaTex-ed string.
    """
    return instring.replace('_', '').replace('{', '').replace('}', '').replace('$', '').replace('\\', '')


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


# ______________________________________________________________________________________________________________________

def read_parameters(infile):
    """
    Read the values of metallicities and their uncertainties. Sort by object name (ID).
    :return: star ID, dictionary with the stellar parameters: [Fe/H], Teff, log(g), v_turb.
    """
    indir = 'Files/'
    tab = ascii.read(indir + infile)
    ids = tab['CEP'].data
    sort_ind = np.argsort(ids)

    Fe, Teff = tab['feh'].data[sort_ind], tab['teff'].data[sort_ind]
    logg, vturb = tab['logg'].data[sort_ind], tab['vtur'].data[sort_ind]

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
    stellar_parameters['Fe'] = {'value': Fe, 'error': dFe, 'label': '[Fe/H]'}
    stellar_parameters['Teff'] = {'value': Teff, 'error': dTeff, 'label': 'T_{eff}'}
    stellar_parameters['logg'] = {'value': logg, 'error': dlogg, 'label': 'log(g)'}
    stellar_parameters['vturb'] = {'value': vturb, 'error': dvturb, 'label': 'v_{turb}'}

    return ids, stellar_parameters


def select_in_common(ids, stellar_parameters, indices_in_common):
    """
    Select the stars in common
    :return:
    """
    ids = ids[indices_in_common]

    for key in stellar_parameters.keys():
        stellar_parameters[key]['value'] = stellar_parameters[key]['value'][indices_in_common]
        stellar_parameters[key]['error'] = stellar_parameters[key]['error'][indices_in_common]

    return ids, stellar_parameters


def plot_and_pick(ids, stellar_parameters_1, stellar_parameters_2, ylab, xlab, figsize):
    fig, ((ax00, ax01), (ax10, ax11), (ax20, ax21)) = plt.subplots(nrows=3, ncols=2, figsize=figsize)
    fig.subplots_adjust(top=0.95, bottom=0.1, left=0.075, right=0.975)
    title = latex(ylab, '\mathrm')
    fig.canvas.set_window_title('')
    fig.suptitle(title)
    # https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
#     __, __, wdx, wdy = plt.get_current_fig_manager().window.geometry().getRect()
#     plt.get_current_fig_manager().window.setGeometry(1100, 20, wdx, wdy)

    line00, = ax00.plot(stellar_parameters_1['Teff']['value'],
                      stellar_parameters_1['Teff']['value'] - stellar_parameters_2['Teff']['value'],
                      marker='o', linestyle='', markersize=10, pickradius=5, picker=True)
    ax00.set_xlabel(latex(stellar_parameters_1['Teff']['label'], '\mathrm') + ' ' + latex(xlab, '\mathrm'))
    ax00.set_ylabel(r'$\Delta$' + latex(stellar_parameters_1['Teff']['label'], '\mathrm'))
    color, marker, = line00.get_color(), line00.get_marker()
    linestyle, markersize = line00.get_linestyle(), line00.get_markersize()
    #
    line01, = ax01.plot(stellar_parameters_1['Fe']['value'],
                      stellar_parameters_1['Fe']['value'] - stellar_parameters_2['Fe']['value'],
                      marker=marker, linestyle=linestyle, color=color, markersize=markersize, pickradius=5, picker=True)
    ax01.set_xlabel(latex(stellar_parameters_1['Fe']['label'], '\mathrm') + ' ' + latex(xlab, '\mathrm'))
    ax01.set_ylabel(r'$\Delta$' + latex(stellar_parameters_1['Fe']['label'], '\mathrm'))
    #
    line10, = ax10.plot(stellar_parameters_1['logg']['value'],
                      stellar_parameters_1['logg']['value'] - stellar_parameters_2['logg']['value'],
                      marker=marker, linestyle=linestyle, color=color, markersize=markersize, pickradius=5, picker=True)
    ax10.set_xlabel(latex(stellar_parameters_1['logg']['label'], '\mathrm') + ' ' + latex(xlab, '\mathrm'))
    ax10.set_ylabel(r'$\Delta$' + latex(stellar_parameters_1['logg']['label'], '\mathrm'))
    #
    line11, = ax11.plot(stellar_parameters_1['vturb']['value'],
                      stellar_parameters_1['vturb']['value'] - stellar_parameters_2['vturb']['value'],
                      marker=marker, linestyle=linestyle, color=color, markersize=markersize, pickradius=5, picker=True)
    ax11.set_xlabel(latex(stellar_parameters_1['vturb']['label'], '\mathrm') + ' ' + latex(xlab, '\mathrm'))
    ax11.set_ylabel(r'$\Delta$' + latex(stellar_parameters_1['vturb']['label'], '\mathrm'))
    #
    dTeff = stellar_parameters_1['Teff']['value'] - stellar_parameters_2['Teff']['value']
    dFe = stellar_parameters_1['Fe']['value'] - stellar_parameters_2['Fe']['value']
    line20, = ax20.plot(dTeff, dFe,
                        marker=marker, linestyle=linestyle, color=color, markersize=markersize, pickradius=5, picker=True)
    ax20.axline((np.mean(dTeff), np.mean(dFe)), slope=0.07 / 100, label='0.07 dex / 100 K (Rom+2008)', color='C3', zorder=5)
    ax20.set_xlabel(r'$\Delta$' + latex(stellar_parameters_1['Teff']['label'], '\mathrm'))
    ax20.set_ylabel(r'$\Delta$' + latex(stellar_parameters_1['Fe']['label'], '\mathrm'))
    ax20.legend()
    #
    bins = np.arange(-0.9, 0.2, 0.1)
    ax21.hist(stellar_parameters_1['Fe']['value'], histtype=u'step', linewidth=3,
                               label=latex(ylab.split('-')[0], '\mathrm'))
    ax21.hist(stellar_parameters_2['Fe']['value'], bins=bins, histtype=u'step', linewidth=3,
             label=latex(ylab.split('-')[-1], '\mathrm'))
    ax21.set_xlabel(latex(stellar_parameters_1['Fe']['label'], '\mathrm'))
    ax21.set_ylabel('Number')
    ax21.set_xlim(left=np.min(bins), right=np.max(bins))
    ax21.legend(loc='upper left')


    index_picked = []
    def on_pick(event, figg):
        ind = event.ind.data[0]
        for axx in figg.get_axes():
            for child in axx.get_children():
                if isinstance(child, matplotlib.text.Annotation):
                    child.remove()
                if isinstance(child, matplotlib.lines.Line2D) and child.get_picker() == True:
                    xx, yy = child.get_xdata(), child.get_ydata()
            axx.annotate(ids[ind], (xx[ind], yy[ind]), color='red', size=12)
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    fig.canvas.mpl_connect('pick_event', lambda x: on_pick(x, fig))

    plt.show()


def main(what_plot, figsize):
    """
    Read the input files with the stellar parameters, match them, make the plots for the user to
    select one star from ... return the id of the selection
    """
    
    # Martino Teff fixed from our LDR
    ids_ldr, stellar_parameters_ldr = read_parameters('SH0ES_atmparam_LDR_alllines_all.dat')
    #
    # Martino Teff fixed from Proxauf's LDR
    ids_ldrProx, stellar_parameters_ldrProx = read_parameters('SH0ES_atmparam_LDRProxauf_alllines_all.dat')
    #
    # Martino exc balance from 5500 K ... preferred
    ids_ew, stellar_parameters_ew = read_parameters('SH0ES_atmparam_FREETotal_alllines_all.dat')
    
    xlab0 = ', T_{eff}\ from\ '
    if what_plot == 'ldr_ldrProx':
        ids = ids_ldr
        stellar_parameters_1 = stellar_parameters_ldr
        stellar_parameters_2 = stellar_parameters_ldrProx
        xlab = xlab0 + 'LDR_{us}'
        ylab = 'LDR_{us} - LDR_{Proxauf}'
    elif what_plot == 'ew_ldr':
        ids = ids_ew
        stellar_parameters_1 = stellar_parameters_ew
        stellar_parameters_2 = stellar_parameters_ldr
        xlab = xlab0 + 'EW'
        ylab = 'EW - LDR_{us}'
    elif what_plot == 'ew_ldrProx':
        ids = ids_ew
        stellar_parameters_1 = stellar_parameters_ew
        stellar_parameters_2 = stellar_parameters_ldrProx
        xlab = xlab0 + 'EW'
        ylab = 'EW - LDR_{Proxauf}'
        
    else:
        print('Argument %s unknown, exiting ...' % (what_plot))
        sys.exit()
    

    plot_and_pick(ids, stellar_parameters_1, stellar_parameters_2, ylab, xlab, figsize)

# ______________________________________________________________________________________________________________________
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('what_plot', help='What plot to plot', type=str)
    parser.add_argument('--figsize', default=[12, 12], help='Size of the figure', nargs='+', type=int)
    args = parser.parse_args()

    main(args.what_plot, tuple(args.figsize))
