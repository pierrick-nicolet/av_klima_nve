
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import DateFormatter
import datetime
from pyextremes import EVA
import numpy as np
import windrose
from klimadata import *

def plot_normaler(klima, ax1=None):
    statistikk_per_maaned = pd.DataFrame({
        'rr':klima['rr'].loc['1991':'2020'].groupby(pd.Grouper(freq='M')).sum(),
        'tm':klima['tm'].loc['1991':'2020'].groupby(pd.Grouper(freq='M')).mean()})

    maanedlig_gjennomsnitt = pd.DataFrame({
        'rr':statistikk_per_maaned['rr'].groupby(statistikk_per_maaned.index.month).mean(),
        'tm':statistikk_per_maaned['tm'].groupby(statistikk_per_maaned.index.month).mean()})


    if ax1 is None:
        ax1 = plt.gca()
    
    ax1.set_title('Gjennomsnittlig månedsnedbør og temperatur (1991 - 2020)')
    ax1.bar(maanedlig_gjennomsnitt.index, maanedlig_gjennomsnitt['rr'], width=0.5, snap=False)
    ax1.set_xlabel('Måned')
    ax1.set_ylabel('Nedbør (mm)')
    ax1.set_ylim(0, maanedlig_gjennomsnitt['rr'].max()+20)
    #ax1.text('1960', aar_df['rr'].max()+20, "Gjennomsnittlig månedsnedbør:  " + str(int(snitt)) + ' mm')

    ax2 = ax1.twinx()#Setter ny akse på høgre side 
    ax2.plot(maanedlig_gjennomsnitt.index, maanedlig_gjennomsnitt['tm'], 'r', label='Gjennomsnittstemperatur', linewidth=3.5)
    ax2.set_ylim(maanedlig_gjennomsnitt['tm'].min()-2,maanedlig_gjennomsnitt['tm'].max()+5)
    ax2.set_ylabel(u'Temperatur (\u00B0C)')
    ax2.yaxis.set_tick_params(length=0)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.axhline(0, linestyle='--', color='grey', linewidth=0.5)
    ax2.get_yaxis().set_visible(True)
    ax2.legend()

    return ax1, ax2


def plot_aarsnedbor(df, ax1=None):
    aarsnedbor= df['rr'].groupby(pd.Grouper(freq='Y')).sum()
    aarsgjennomsnitt_1961_1990 = int(aarsnedbor.loc['1961':'1990'].mean())
    aarsgjennomsnitt_1990_2020 = int(aarsnedbor.loc['1991':'2020'].mean())

    slope, y0, r, p, stderr = stats.linregress(
        aarsnedbor.index.map(datetime.date.toordinal),
        aarsnedbor)

    x_endpoints = pd.DataFrame([
        aarsnedbor.index.map(datetime.date.toordinal)[0],
        aarsnedbor.index.map(datetime.date.toordinal)[-1]])
    y_endpoints = y0 + slope * x_endpoints

    if ax1 is None:
        ax1 = plt.gca()

    ax1.set_title('Årsnedbør')
    ax1.bar(aarsnedbor.index,  aarsnedbor, width=320, snap=False)
    ax1.set_xlabel('Årstall')
    ax1.set_ylabel('Nedbør (mm)')
    ax1.set_ylim(aarsnedbor.min()*0.6, aarsnedbor.max()*1.1)
    ax1.text(aarsnedbor.index[0],
            aarsnedbor.max(), 
            "Gjennomsnittlig aarsnedbor(1991-2020):  " + str(int(aarsgjennomsnitt_1990_2020)) + ' mm')

    ax2 = ax1.twinx()
    ax2.plot([aarsnedbor.index[0], 
            aarsnedbor.index[-1]], 
            [y_endpoints[0][0], y_endpoints[0][1]], 
            linestyle='dashed', 
            linewidth=1, 
            color='r', 
            label='Trend')
    ax2.hlines(
        y=aarsgjennomsnitt_1961_1990, 
        xmin=datetime.datetime.strptime('1961-01-01', '%Y-%m-%d'), 
        xmax=datetime.datetime.strptime('1990-12-31', '%Y-%m-%d'), 
        linestyle='dashed', 
        linewidth=2, 
        color='g', 
        label='Snitt 1961-1990')
    ax2.hlines(
        y=aarsgjennomsnitt_1990_2020, 
        xmin=datetime.datetime.strptime('1990-01-01', '%Y-%m-%d'), 
        xmax=datetime.datetime.strptime('2020-12-31', '%Y-%m-%d'), 
        linestyle='dashed', 
        linewidth=2, 
        color='r', 
        label='Snitt 1991-2020')
    ax2.set_ylim(aarsnedbor.min()*0.6, aarsnedbor.max()*1.1)
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc='lower right')

    return ax1, ax2

def snodjupne(df, ax1=None):
    sno = df['sd'].groupby(pd.Grouper(freq='Y')).max() #Finner maksimal snødjupne per år
    maxaar = sno.idxmax() 
    snomax = sno.max() 
    snosnitt_6090 = sno.loc['1961':'1990'].mean() 
    snosnitt_9020 = sno.loc['1991':'2020'].mean() 

    slope, y0, r, p, stderr = stats.linregress(
        sno.index.map(datetime.date.toordinal),
        sno)

    x_endpoints = pd.DataFrame([
        sno.index.map(datetime.date.toordinal)[0],
        sno.index.map(datetime.date.toordinal)[-1]])
    y_endpoints = y0 + slope * x_endpoints

    # slope, y0, r, p, stderr = stats.linregress(sno.index.map(datetime.date.toordinal), sno)
    # x_endpoints = pd.DataFrame([sno.index[0], sno.index[-1]])
    # y_endpoints = y0 + slope * x_endpoints
    if ax1 is None:
        ax1 = plt.gca()

    #fig, ax1 = plt.subplots()

    ax1.set_title('Maksimal snødjupe')
    ax1.bar(sno.index, sno, width=320, snap=False, color='powderblue') 
    ax1.set_xlabel('Årstall')
    ax1.set_ylabel('Snødjupne (cm)')
    ax1.set_ylim(sno.min()*0.6, sno.max()*1.1)
    ax1.annotate('Maks snøhøgde: ' +str(df['sd'].idxmax().date()) + ' | ' + str(df['sd'].max()) + 'cm', 
                xy=(df['sd'].idxmax().date(), df['sd'].max()),  
                xycoords='data',
                xytext=(0.1, 0.9), 
                textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->"))

    ax2 = ax1.twinx()
    ax2.plot([sno.index[0], sno.index[-1]], 
            [y_endpoints[0][0], y_endpoints[0][1]], 
            linestyle='dashed', 
            linewidth=1, 
            color='b', 
            label='Trend')
    ax2.hlines(y=sno.mean(), 
            xmin=datetime.datetime.strptime('1958-01-01', '%Y-%m-%d'), 
            xmax=datetime.datetime.strptime('2020-12-31', '%Y-%m-%d'), 
            linestyle='dashed', 
            linewidth=1, color='y', 
            label='Snitt 1961-1990')
    ax2.hlines(y=snosnitt_6090, 
            xmin=datetime.datetime.strptime('1961-01-01', '%Y-%m-%d'), 
            xmax=datetime.datetime.strptime('1990-12-31', '%Y-%m-%d'), 
            linestyle='dashed', 
            linewidth=2, color='g', 
            label='Snitt 1961-1990')
    ax2.hlines(y=snosnitt_9020, 
            xmin=datetime.datetime.strptime('1991-01-01', '%Y-%m-%d'), 
            xmax=datetime.datetime.strptime('2020-12-31', '%Y-%m-%d'), 
            linestyle='dashed', 
            linewidth=2, 
            color='r', 
            label='Snitt 1991-2020')
    ax2.set_ylim(sno.min()*0.6, sno.max()*1.1)
    ax2.get_yaxis().set_visible(False)
    ax1.text(sno.index[0], 
            sno.min()*0.7, 
            "Gjennomsnittlig maksimal snødjupne (1991-2020):  " + str(int(snosnitt_9020)) + ' cm')
    ax1.text(sno.index[0], 
            sno.min()*0.7, 
            "Gjennomsnittlig maksimal snødjupne (1991-2020):  " + str(int(snosnitt_9020)) + ' cm')
    ax2.legend(loc='best')

    return ax1, ax2

def nysnodjupne_3d(df, ax1=None):
    max_df = maxdf(df)
    maksimal_sdfsw3ddato = df['sdfsw3d'].idxmax().date()
    maksimal_sdfsw3d = df['sdfsw3d'].max()

    slope, y0, r, p, stderr = stats.linregress(
        max_df['sdfsw3d'].index.map(datetime.date.toordinal),
        max_df['sdfsw3d'])

    x_endpoints = pd.DataFrame([
        max_df['sdfsw3d'].index.map(datetime.date.toordinal)[0],
        max_df['sdfsw3d'].index.map(datetime.date.toordinal)[-1]])
    y_endpoints = y0 + slope * x_endpoints

    if ax1 is None:
        ax1 = plt.gca()

    ax1.set_title('Maksimal nysnødybde 3 døgn')
    ax1.bar(max_df.index, max_df['sdfsw3d'], width=320, snap=False, color='skyblue')
    ax1.set_xlabel('Årstall')
    ax1.set_ylabel('Nysnødybde 3 døgn (cm)')
    ax1.set_ylim(max_df['sdfsw3d'].min()*0.8, max_df['sdfsw3d'].max()*1.1)

    ax1.text(max_df.index[0], df['sdfsw3d'].max(), 'Maksimalverdi: ' + str(maksimal_sdfsw3ddato) + ' | ' + str(maksimal_sdfsw3d) + ' cm' )
    ax1.text(max_df['sdfsw3d'].index[0], 
            max_df['sdfsw3d'].min()*0.9, 
            "Snitt nysnødybde 3 døgn (1991-2020):  " + str(int(max_df['sdfsw3d'].mean())) + ' cm')

    ax2 = ax1.twinx()
    ax2.plot([max_df['sdfsw3d'].index[0], max_df['sdfsw3d'].index[-1]], 
            [y_endpoints[0][0], y_endpoints[0][1]], 
            linestyle='dashed', 
            linewidth=1, 
            color='b', 
            label='Trend')
    ax2.hlines(y=max_df['sdfsw3d'].mean(), 
            xmin=datetime.datetime.strptime('1958-01-01', '%Y-%m-%d'), 
            xmax=datetime.datetime.strptime('2020-12-31', '%Y-%m-%d'), 
            linestyle='dashed', 
            linewidth=1, color='y', 
            label='Snitt 1961-1990')
    ax2.hlines(y=max_df['sdfsw3d'].loc['1961':'1990'].mean(), 
            xmin=datetime.datetime.strptime('1961-01-01', '%Y-%m-%d'), 
            xmax=datetime.datetime.strptime('1990-12-31', '%Y-%m-%d'), 
            linestyle='dashed', 
            linewidth=2, color='g', 
            label='Snitt 1961-1990')
    ax2.hlines(y=max_df['sdfsw3d'].loc['1991':'2020'].mean(), 
            xmin=datetime.datetime.strptime('1991-01-01', '%Y-%m-%d'), 
            xmax=datetime.datetime.strptime('2020-12-31', '%Y-%m-%d'), 
            linestyle='dashed', 
            linewidth=2, 
            color='r', 
            label='Snitt 1991-2020')
    ax2.set_ylim(max_df['sdfsw3d'].min()*0.8, max_df['sdfsw3d'].max()*1.1)
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc='best')

    return ax1, ax2

def snomengde(df, ax1=None):
    snodager = (df['sd']
            .groupby(df.index.strftime('%m-%d'))
            .mean()
            .to_frame()
            .rename(columns={"sd":"sd_snitt"})
            .assign(tm=df['tm'].groupby(df.index.strftime('%m-%d')).mean().rolling(7).mean(),
                    sd_max=df['sd'].groupby(df.index.strftime('%m-%d')).max().rolling(7).mean(),
                    sd_min=df['sd'].groupby(df.index.strftime('%m-%d')).min().rolling(7).mean(),
                   )
           )

    if ax1 is None:
        ax1 = plt.gca()

    ax1.plot(snodager.index, snodager['sd_snitt'], label='Snitt snømengde')
    ax1.plot(snodager.index, snodager['sd_max'], label='Max snømengde')
    ax1.plot(snodager.index, snodager['sd_min'], label='Min snømengde')
    ax1.xaxis.set_major_locator(MultipleLocator(32))
    #ax1.xaxis.set_major_formatter(FormatStrFormatter('%m'))
    ax1.set_title('Periode med snø - døgntemperatur (1958-2022)')
    ax1.set_xlabel('Måned')
    ax1.set_ylabel('Snøhøgde (cm)')
    ax1.xaxis.set_major_formatter(DateFormatter("%m"))
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(snodager.index, snodager['tm'], 'r--', label='Gjennomsnittstemperatur')
    ax2.xaxis.set_major_locator(MultipleLocator(32))
    ax2.legend(loc='lower left')
    ax2.set_ylim(snodager['tm'].min()-5, snodager['tm'].max()+5)
    ax2.axhline(0, linestyle='--', color='grey', linewidth=0.5)
    ax2.set_ylabel(u'Temperatur (\u00B0C)')

    return ax1, ax2


def ekstremverdi_3d_sd(df, ax1=None):
    data = df['sdfsw3d']
    model = EVA(data=data)
    model.get_extremes(
    method="BM",
    extremes_type="high",
    block_size="365.2425D",
    errors="raise",
    )
    model.fit_model()

    if ax1 is None:
        ax1 = plt.gca()

    fig, ax1 = model.plot_return_values(
    return_period=np.logspace(0.01, 3.75, 5000),
    alpha=0.95,
    )

    summary = model.get_summary(
    return_period=[100, 1000, 5000],
    alpha=0.95,
    n_samples=1000,
    )

    ax1.set_xlabel('Returperiode (År)')
    ax1.set_ylabel('Maksimal årlig 3 døgns nysnøhøgde (cm)')
    ax1.text(100, 
            summary['return value'].min()*0.3, 
            '100 år returverdi: ' + str(round(summary['return value'].loc[100.0])) + ' cm \n'
            '1000 år returverdi: ' + str(round(summary['return value'].loc[1000.0])) + ' cm \n'
            '5000 år returverdi: ' + str(round(summary['return value'].loc[5000.0])) + ' cm \n' )

    return fig

def vind(vind_df):
    vind_df['retning'] = vind_df['windDirection10m24h06']*45
    vind_regn_df = vind_regn(vind_df)
    vind_sno_df= vind_sno_fsw(vind_df)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, subplot_kw=dict(projection='windrose'), figsize=(20,20))

    ax1.bar(vind_df['retning'], vind_df['windSpeed10m24h06'], normed=True, opening=1.8)
    ax1.set_title(f"%-vis med dager vindretning ({len(vind_df['retning'])} dager)")
    #ax1.legend(title='Vindstyrke (m/s')
    ax1.set_legend(title='Vindstyrke (m/s)')

    ax2.bar(vind_regn_df['retning'], vind_regn_df['rr'], normed=True, opening=1.8)
    ax2.set_title(f"%-vis med dager vindretning og regn ({len(vind_regn_df['retning'])} dager)")
    ax2.set_legend(title='Regn (rrl) (mm)')

    ax3.bar(vind_sno_df['retning'], vind_sno_df['rr'], normed=True,opening=1.8)
    ax3.set_title(f"%-vis med dager vindretning og snø ({len(vind_sno_df['retning'])} dager)")
    ax3.set_legend(title='Snø (fsw) (mm)')

    return fig

def klimaoversikt(df, lokalitet):
    fig = plt.figure(figsize=(20, 12))

    ax1 = fig.add_subplot(221)

    ax1, ax2 = plot_normaler(df)
    ax3 = fig.add_subplot(222)

    ax3, ax4 = snomengde(df)
    ax5 = fig.add_subplot(223)

    ax5 = plot_aarsnedbor(df)
    ax6 = fig.add_subplot(224)

    ax6, ax7 = snodjupne(df)

    fig.suptitle(f'Klimaoversikt for {lokalitet}', fontsize=30, y=0.9, va='bottom')

    return fig

def klima_sno_oversikt(df, lokalitet):
    fig = plt.figure(figsize=(20, 18))

    ax1 = fig.add_subplot(321)
    ax1, ax2 = plot_normaler(df)
    
    ax3 = fig.add_subplot(322)
    ax3, ax4 = snomengde(df)
    
    ax5 = fig.add_subplot(323)
    ax5 = plot_aarsnedbor(df)
    
    ax6 = fig.add_subplot(324)
    ax6, ax7 = snodjupne(df)

    ax8 = fig.add_subplot(325)
    ax8, ax9 = nysnodjupne_3d(df)

    ax10 = fig.add_subplot(326)
    ax10 = gammel_plot_ekstremverdier_3dsno(df)


    fig.suptitle(f'Klimaoversikt for {lokalitet}', fontsize=30, y=0.9, va='bottom')

    return fig

def gammel_plot_ekstremverdier_3dsno(df, ax1=None):
    maximal = maxdf(df)
    liste =  maximal['sdfsw3d'].tolist()
    array = np.array(liste)
    model = e.Gumbel(array, fit_method = 'mle', ci = 0.05, ci_method = 'delta')
    
    if ax1 is None:
        ax1 = plt.gca()

    return model.plot_return_values('3ds')

def ekstrem_3d_sno_oversikt(df):
    fig = plt.figure(figsize=(20, 8))

    ax1 = fig.add_subplot(121)
    ax1, ax2 = nysnodjupne_3d(df)

    ax3 = fig.add_subplot(122)
    ax3 = gammel_plot_ekstremverdier_3dsno(df)

    return fig

    