#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

#for the data preprocessing i have removed all the whitespaces,
#the punctuation and put a keyboard script between the hours and days

files = ['caudal_aforo/Arroyo.csv','caudal_aforo/Calatayud.csv','caudal_aforo/Logroño.csv',
            'caudal_aforo/Mendavia.csv','caudal_aforo/Miranda.csv', 'caudal_aforo/Tudela.csv']

df = pd.read_csv(files[0], sep=';', decimal=',',dtype={'fecha':str,'Media':float})
df = df.drop(columns=['Mximo','Mnimo','Fechamximo','Fechamnimo'])
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d-%H:%M:%S')

for f in files[1:]:
    df_aux = pd.read_csv(f, sep=';', decimal=',',dtype={'fecha':str,'Media':float})
    df_aux = df_aux.drop(columns=['Mximo','Mnimo','Fechamximo','Fechamnimo'])
    df_aux['fecha'] = pd.to_datetime(df_aux['fecha'])
    df = pd.merge_asof(df, df_aux, on='fecha')

#process the volume of the embalse data
files = ['volumen_embalse/ebro.csv','volumen_embalse/mansilla.csv','volumen_embalse/tranquera.csv','volumen_embalse/yesa.csv']

for f in files:
    df_aux = pd.read_csv(f, sep=';', decimal=',',dtype={'fecha':str,'Acumulado':float})
    df_aux = df_aux.drop(columns=['Fechaacumulado'])
    df_aux['fecha'] = pd.to_datetime(df_aux['fecha'])
    df = pd.merge_asof(df, df_aux, on='fecha')

print(df[0:50])
