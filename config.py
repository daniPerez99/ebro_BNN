#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#variables globales de configuracion

DATA_PATH = "/DATA"
LOG_DIR = "/LOGS"

RIVER_FILES = ['DATA/caudal_aforo/Arroyo.csv',
                'DATA/caudal_aforo/Calatayud.csv',
                'DATA/caudal_aforo/Logroño.csv',
                'DATA/caudal_aforo/Mendavia.csv',
                'DATA/caudal_aforo/Miranda.csv', 
                'DATA/caudal_aforo/Tudela.csv']

EMBALSE_FILES = ['DATA/volumen_embalse/ebro.csv',
                'DATA/volumen_embalse/mansilla.csv',
                'DATA/volumen_embalse/tranquera.csv',
                'DATA/volumen_embalse/yesa.csv']

PRECIPITATION_FILES = ['DATA/precipitaciones/arce.csv',
                        'DATA/precipitaciones/calatayud.csv',
                        'DATA/precipitaciones/ebro.csv',
                        'DATA/precipitaciones/laloteta.csv',
                        'DATA/precipitaciones/mansilla.csv',
                        'DATA/precipitaciones/romeral.csv',
                        'DATA/precipitaciones/tauste.csv',
                        'DATA/precipitaciones/tranquera.csv',
                        'DATA/precipitaciones/yesa.csv']


# Data parameters
NUM_FEATURES = 21

# Model parameters
LAYER1_NEURONS = 7
LAYER2_NEURONS = 3

# Training parameters
P_TRAIN = 0.8
NUM_EPOCHS = 1000
LEARNING_RATE = 0.01

# Callback parameters
PRINT_EPOCH = 100
LOSSES_AVG_NO = 10