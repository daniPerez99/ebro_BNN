#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import config as cfg

def process_rios():
    
    files = cfg.RIVER_FILES

    df = pd.read_csv(files[0], sep=';', decimal=',')
    df.drop(columns=['Mximo','Mnimo','Fechamximo','Fechamnimo'],inplace=True)
    df['fecha'] = pd.to_datetime(df['fecha'])
    #get the name from the string
    files[0] = files[0].split('/')[2].split('.')[0]
    df.rename(columns={'Media':files[0]}, inplace=True)

    for f in files[1:]:
        df_aux = pd.read_csv(f, sep=';', decimal=',')
        df_aux.drop(columns=['Mximo','Mnimo','Fechamximo','Fechamnimo'],inplace=True)
        df_aux['fecha'] = pd.to_datetime(df_aux['fecha'])
        #get the name from the string
        f = f.split('/')[2].split('.')[0]
        df_aux.rename(columns={'Media':f}, inplace=True)
        df = pd.merge_asof(df, df_aux, on='fecha')
    
    return df

def process_embalse(df):
        #process the volume of the embalse data
    files = cfg.EMBALSE_FILES

    for f in files:
        df_aux = pd.read_csv(f, sep=';', decimal=',')
        df_aux = df_aux.drop(columns=['Fechaacumulado'])
        df_aux['fecha'] = pd.to_datetime(df_aux['fecha'])
        #get the name from the string
        f = f.split('/')[2].split('.')[0]
        df_aux.rename(columns={'Acumulado':f}, inplace=True)
        df = pd.merge_asof(df, df_aux, on='fecha')
    return df

def process_precipitaciones(df):
    #process the precipitation data
    files = cfg.PRECIPITATION_FILES

    for f in files:
        df_aux = pd.read_csv(f, sep=';', decimal=',')
        df_aux = df_aux.drop(columns=['Fechaacumulado','Fechamximo','Mximo'])
        df_aux['fecha'] = pd.to_datetime(df_aux['fecha'])
        #get the name from the string
        f = f.split('/')[2].split('.')[0]
        df_aux.rename(columns={'Acumulado':f}, inplace=True)
        df = pd.merge_asof(df, df_aux, on='fecha')
    return df

def process_date(df):

    #use sin and cos to represent the date
    df.insert(loc = 0, column = 'day_sin', value = 0)
    df.insert(loc = 0, column = 'day_cos', value = 0)
    df['day_sin'] = np.sin((df.fecha.dt.day - 1) / (366 - 1) * 2 * np.pi)
    df['day_cos'] = np.cos((df.fecha.dt.day - 1) / (366 - 1) * 2 * np.pi)
    df.drop('fecha',axis=1,inplace=True)
    return df

def process_result(df):
    df_aux = pd.read_csv('DATA/prediccion/pred.csv', sep=';', decimal=',')
    df_aux.drop(columns=['Mximo','Mnimo','Fechamximo','Fechamnimo'],inplace=True)
    df_aux['fecha'] = pd.to_datetime(df_aux['fecha'])
    df_aux.rename(columns={'Media':'pred'}, inplace=True)
    #setting an offset to fecha to align with the prediction
    df['fecha'] = df['fecha'].add(pd.Timedelta('1 days'))
    df = pd.merge_asof(df,df_aux,on='fecha')
    return df


#for the data preprocessing i have removed all the whitespaces,
#the punctuation and put a keyboard script between the hours and days
def prepare_data():

    df = process_rios()

    df = process_embalse(df)

    df = process_precipitaciones(df)
    
    df = process_result(df)

    df = process_date(df)

    #drop all the inputs that contain nan
    df.dropna(inplace=True)

    return df