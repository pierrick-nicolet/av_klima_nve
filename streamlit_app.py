from matplotlib import pyplot as plt
import streamlit as st
from pyproj import CRS
from pyproj import Transformer
import klimadata
import plot
import folium
from streamlit_folium import st_folium
import pandas as pd

parameterliste = ['rr', 'tm', 'sd', 'fsw', 'sdfsw', 'sdfsw3d']
transformer = Transformer.from_crs(4326, 5973)
m = folium.Map(location=[62.14497, 9.404296], zoom_start=5)
folium.raster_layers.WmsTileLayer(
    url='https://opencache.statkart.no/gatekeeper/gk/gk.open_gmaps?layers=topo4&zoom={z}&x={x}&y={y}',
    name='Norgeskart',
    fmt='image/png',
    layers='topo4',
    attr=u'<a href="http://www.kartverket.no/">Kartverket</a>',
    transparent=True,
    overlay=True,
    control=True,
    
).add_to(m)
m.add_child(folium.ClickForMarker(popup="Waypoint"))
#from folium.plugins import Draw
#Draw().add_to(m)
output = st_folium(m, width = 700, height=500)
utm_lat = 0
utm_lon = 0
st.write('Trykk i kartet, eller skriv inn koordinater for å velge klimapunkt.')
st.write('Finne automatisk nærmaste stadsnavn dersom det finnes eit registert navn innafor 500m radius.')


try:
    kart_kord_lat = output['last_clicked']['lat']
    kart_kord_lng = output['last_clicked']['lng']
    utm_øst, utm_nord = transformer.transform(kart_kord_lat, kart_kord_lng)
    utm_nord = round(utm_nord,2)
    utm_øst = round(utm_øst,2)

except TypeError:
    utm_nord  = 'Trykk i kart, eller skriv inn koordinat'
    utm_øst = 'Trykk i kart, eller skriv inn koordinat'


lat = st.text_input("NORD(UTM 33)", utm_nord)
lon = st.text_input("ØST  (UTM 33)", utm_øst)


try:
    navn = klimadata.stedsnavn(utm_nord, utm_øst)['navn'][0]['stedsnavn'][0]['skrivemåte']
except (IndexError, KeyError):
    navn = 'Skriv inn navn'
#st.write(navn)
#st.write(navn['navn'][0]['stedsnavn'][0]['skrivemåte'])

#st.write(klimadata.stedsnavn(lng, lat))
lokalitet = st.text_input("Gi navn til lokalitet", navn)

startdato = '1958-01-01'
#sluttdato = st.text_input('Gi sluttdato', '2019-12-31')
sluttdato = '2021-12-31'

plottype = st.radio('Velg plottype', ('Klimaoversikt', 'Klimaoversikt med 3 døgn snø og returverdi'))
vind = st.checkbox('Vindanalyse')

knapp = st.button('Vis plott')

if knapp:
    lon = int(float(lon.strip()))
    lat = int(float(lat.strip()))
    df = klimadata.klima_dataframe(lon, lat, startdato, sluttdato, parameterliste)

    if plottype == 'Klimaoversikt':
        st.pyplot(plot.klimaoversikt(df, lokalitet))
        #klimaoversikt(df)
        st.download_button(
            "Last ned klimadata",
            df.to_csv().encode('utf-8'),
            "klimadata.csv",
            "text/csv",
            key='download-csv'
            )
    if plottype == 'Klimaoversikt med 3 døgn snø og returverdi':
        st.pyplot(plot.klima_snø_oversikt(df, lokalitet))
        st.download_button(
            "Last ned klimadata",
            df.to_csv().encode('utf-8'),
            "klimadata.csv",
            "text/csv",
            key='download-csv'
            )
    if vind:
        vind_para = ['windDirection10m24h06', 'windSpeed10m24h06', 'rr', 'tm', 'fsw', 'rrl']
        vindslutt = '2022-03-01'
        vindstart = '2018-03-01'
        vind_df = klimadata.klima_dataframe(lon, lat, vindstart, vindslutt, vind_para)
        st.pyplot(plot.vind(vind_df))
        st.download_button(
            "Last ned vinddata",
            vind_df.to_csv().encode('utf-8'),
            "vinddata.csv",
            "text/csv",
            key='download-csv'
            )

st.write('Scriptet henter ned data frå NVE sitt Grid Time Series API, som er visualisert på xgeo.no')
st.write('Parametere som er brukt er: ')

parametere = {
    'rr':'Døgnnedbør v2.0 - mm',
    'tm':'Døgntemperatur v2.0	 - Celcius',
    'sd':'Snødybde v2.0.1 - cm',
    'fsw':'Nysnø siste døgn	 - mm',
    'sdfsw3d':'Nysnødybde 3 døgn - cm',
    'rrl':'Regn - mm',
    'windDirection10m24h06':'Vindretning 10m døgn',
    'windSpeed10m24h06':'Vindhastighet 10m døgn -  m/s'
}
st.json(parametere)