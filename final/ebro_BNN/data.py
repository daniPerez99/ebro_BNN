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
    f = files[0].split('/')[2].split('.')[0]
    df.rename(columns={'Media':'aforo_'+f}, inplace=True)

    for f in files[1:]:
        df_aux = pd.read_csv(f, sep=';', decimal=',')
        df_aux.drop(columns=['Mximo','Mnimo','Fechamximo','Fechamnimo'],inplace=True)
        df_aux['fecha'] = pd.to_datetime(df_aux['fecha'])
        #get the name from the string
        f = f.split('/')[2].split('.')[0]
        df_aux.rename(columns={'Media':'aforo_'+f}, inplace=True)
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
        df_aux.rename(columns={'Acumulado':'embalse_'+f}, inplace=True)
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
        df_aux.rename(columns={'Acumulado':'precipitacion_'+f}, inplace=True)
        df = pd.merge_asof(df, df_aux, on='fecha')
    return df

def process_date(df):
    #use sin and cos to represent the date
    df.insert(loc = 0, column = 'day_sin', value = 0)
    df.insert(loc = 0, column = 'day_cos', value = 0)
    df['day_sin'] = np.sin((df.fecha.dt.dayofyear - 1) / (366 - 1) * 2 * np.pi)
    df['day_cos'] = np.cos((df.fecha.dt.dayofyear - 1) / (366 - 1) * 2 * np.pi)
    df.drop('fecha',axis=1,inplace=True)
    return df

def process_result(df,pred='24H'):
    df_aux = pd.read_csv('DATA/prediccion/pred.csv', sep=';', decimal=',')
    df_aux.drop(columns=['Mximo','Mnimo','Fechamximo','Fechamnimo'],inplace=True)
    df_aux['fecha'] = pd.to_datetime(df_aux['fecha'])
    df_aux.rename(columns={'Media':'pred'}, inplace=True)
    #setting an offset to fecha to align with the prediction
    if pred == '24H':
        df['fecha'] = df['fecha'].add(pd.Timedelta('1 days'))
    elif pred == '48H':
        df['fecha'] = df['fecha'].add(pd.Timedelta('2 days'))
    elif pred == '72H':
        df['fecha'] = df['fecha'].add(pd.Timedelta('3 days'))
    df = pd.merge_asof(df,df_aux,on='fecha')
    return df

def process_caudal(df):
    #process the flow
    files = cfg.FLOW_FILES
    for f in files:
        df_aux = pd.read_csv(f, sep=';', decimal=',')
        df_aux.drop(columns=['Mximo','Mnimo','Fechamximo','Fechamnimo'],inplace=True)
        df_aux['fecha'] = pd.to_datetime(df_aux['fecha'])
        #get the name from the string
        f = f.split('/')[2].split('.')[0]
        df_aux.rename(columns={'Media':f}, inplace=True)
        df = pd.merge_asof(df, df_aux, on='fecha')
    
    return df

#for the data preprocessing i have removed all the whitespaces,
#the punctuation and put a keyboard script between the hours and days
def prepare_data(pred='24H'):

    df = process_rios()

    df = process_embalse(df)

    #df = process_precipitaciones(df)

    df = process_caudal(df)

    df = process_result(df,pred)

    df = process_date(df)
    #drop all the inputs that contain nan
    df.dropna(inplace=True)
    print('number of rows in '+pred+': ',len(df))

    return df


def normalize_data_mean(df):
    #normalize the data
    df_norm = (df - df.mean()) / df.std()
    #restore the original values
    df_norm['pred'] = df['pred']
    df_norm['day_cos'] = df['day_cos']
    df_norm['day_sin'] = df['day_sin']
    return df_norm

def normalize_data_minmax(df):
    #normalize the data
    df_norm = (df - df.min()) / (df.max() - df.min())
    #restore the original values
    df_norm['pred'] = df['pred']
    df_norm['day_cos'] = df['day_cos']
    df_norm['day_sin'] = df['day_sin']
    return df_norm

def split_flood(df):
    #split the data in flood and non flood
    df_flood = df[df['pred'] > 3]
    df_no_flood = df[df['pred'] <= 3]
    print(f'number of flood rows: {len(df_flood)}')
    print(f'number of no flood rows: {len(df_no_flood)}')

    return df_flood, df_no_flood

ITER = 'mejora2'
pred_date = '24H'
df = prepare_data(pred_date)
#the shuffle is done here to have the same distribution of data in all cases.
#df = shuffle(df, random_state=seed)

df.to_csv('DATA/datos_procesados/'+ITER+'/datos_'+pred_date+'_NoRand.csv',index=False)

df_aux = normalize_data_mean(df)
df_aux.to_csv('DATA/datos_procesados/'+ITER+'/datos_mean_'+pred_date+'_NoRand.csv',index=False)

df_aux = normalize_data_minmax(df)
df_aux.to_csv('DATA/datos_procesados/'+ITER+'/datos_minmax_'+pred_date+'_NoRand.csv',index=False)