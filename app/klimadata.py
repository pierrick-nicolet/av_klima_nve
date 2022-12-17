import pandas as pd
import datetime
import requests
import extreme as e
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def nve_api(lat, lon, startdato, sluttdato, para):
    """Henter data frå NVE api GridTimeSeries
    Parameters:
        lat (str): øst-vest koordinat (i UTM33)
        output er verdien i ei liste, men verdi per dag, typ ne
        lon (str): nord-sør koordinat (i UTM33)
        startdato (str): startdato for dataserien som hentes ned
        sluttdato (str): sluttdato for dataserien som hentes ned
        para (str): kva parameter som skal hentes ned f.eks rr for nedbør
        
    Returns:
        verdier (list) : returnerer i liste med klimaverdier
        
    """
    api = 'http://h-web02.nve.no:8080/api/'
    url = api + '/GridTimeSeries/' + str(lat) + '/' + str(lon) + '/' + str(startdato) + '/' + str(sluttdato) + '/' + para + '.json'
    r = requests.get(url)

    verdier = r.json()
    return verdier

def stedsnavn(utm_nord, utm_ost):
    '''Henter stedsnavn fra geonorge
    Parameters:
        utm_nord (str): nord koordinat i UTM33
        utm_ost (str): øst koordinat i UTM33

    Returns:
        verier (list): Liste med stedsnavn innanfor radius på 500m
    
    '''
    url = f'https://ws.geonorge.no/stedsnavn/v1/punkt?nord={utm_nord}&ost={utm_ost}&koordsys=5973&radius=500&utkoordsys=4258&treffPerSide=1&side=1'
    r = requests.get(url)
    verdier = r.json()
    #for verdi in
    return verdier

def hent_data_klima_dogn(lat, lon, startdato, sluttdato, parametere):
    '''Henter ned klimadata basert på liste av parametere
        Parameters:
            lat (str): øst-vest koordinat (i UTM33)
            output er verdien i ei liste, men verdi per dag, typ ne
            lon (str): nord-sør koordinat (i UTM33)
            startdato (str): startdato for dataserien som hentes ned
            sluttdato (str): sluttdato for dataserien som hentes ned
            parametere (str): liste med parametere som skal hentes ned f.eks rr for nedbør
        
        Returns:
            parameterdict (dict): dict med parameternavn til key, og liste med verdier som value

    '''
    parameterdict = {}
    for parameter in parametere:
        
        parameterdict[parameter] = nve_api(lat, lon, startdato, sluttdato, parameter)['Data']
    return parameterdict    


def klima_dataframe(lat, lon, startdato, sluttdato, parametere):
    '''Lager dataframe basert på klimadata fra NVE api. Bruker underfunksjoner. Bruker start og sluttdato
    for å generere index i pandas dataframe.

        Parameters:
            lat (str): øst-vest koordinat (i UTM33)
            output er verdien i ei liste, men verdi per dag, typ ne
            lon (str): nord-sør koordinat (i UTM33)
            startdato (str): startdato for dataserien som hentes ned
            sluttdato (str): sluttdato for dataserien som hentes ned
            parametere (str): liste med parametere som skal hentes ned f.eks rr for nedbør
        
        Return:
            df (pandas dataframe): dataframe med klimadata.
    
    '''
    parameterdict = {}
    for parameter in parametere:
        
        parameterdict[parameter] = nve_api(lat, lon, startdato, sluttdato, parameter)['Data']
     
    df = pd.DataFrame(parameterdict)
    df = df.set_index(pd.date_range(
        datetime.datetime(int(startdato[0:4]), int(startdato[5:7]), int(startdato[8:10])),
        datetime.datetime(int(sluttdato[0:4]), int(sluttdato[5:7]), int(sluttdato[8:10])))
    )
    df[df > 1000] = 0
    df = rullande_3dogn_nedbor(df)
    print(df.head())
    return df

def maxdf(df):
    '''Tar in klimadataframe, og returnerer ny dataframe med årlige maksimalverdier'''
    maxdf = (pd.DataFrame(df['sdfsw3d'].groupby(pd.Grouper(freq='Y')).max())
             .assign(rr = df['rr'].groupby(pd.Grouper(freq='Y')).max(),
                    rr3 = df['rr3'].groupby(pd.Grouper(freq='Y')).max(),
                    sd = df['sd'].groupby(pd.Grouper(freq='Y')).max())
        )
    return maxdf

def vind_nedbor(df):
    '''Tar in vinddataframe, og returnerer ny dataframe med der nedbør under 0.2mm blir fjerna'''
    return df.where(df.rr > 0.2).dropna()

def vind_regn(df):
    '''Tar in vinddataframe, og returnerer ny dataframe med der regn under 0.2mm blir fjerna'''
    return df.where(df.rrl > 0.2).dropna()

def vind_sno_fsw(df):
    '''Tar in vinddataframe, og returnerer ny dataframe med der nysnø under 0.2mm blir fjerna'''
    return df.where(df.fsw > 0.2).dropna()

def vind_sno_rr_tm(df):
    '''Tar in vinddataframe, og returnerer ny dataframe med der nedbør under 0.2mm og temperatur blir fjerna'''
    return (df.where(df.rr > 0.2).dropna()
            .where(df.tm < 1).dropna())

def rullande_3dogn_nedbor(dataframe):
    '''Tar in klimadataframe og returnerer med ny kollonne med utrekna 3 døgs nedbør basert på døgnnedbør'''
    df = (dataframe.assign(rr3 = dataframe['rr'].rolling(3).sum().round(2))
     .fillna(0)
    )
    return df

def gammel_plot_ekstremverdier_3dsno(df, ax1=None):
    maximal = maxdf(df)
    liste =  maximal['sdfsw3d'].tolist()
    array = np.array(liste)
    model = e.Gumbel(array, fit_method = 'mle', ci = 0.05, ci_method = 'delta')
    
    if ax1 is None:
        ax1 = plt.gca()

    return model.plot_return_values('3ds')