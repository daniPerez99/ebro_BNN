#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#variables globales de configuracion

DATA_PATH = "/DATA"
LOG_DIR = "/LOGS"

#comentados en la iteracion 1
RIVER_FILES = ['DATA/nivel_aforo/Arroyo.csv',
                'DATA/nivel_aforo/Calatayud.csv',
                #'DATA/nivel_aforo/Logroño.csv',
                'DATA/nivel_aforo/Mendavia.csv',
                'DATA/nivel_aforo/Miranda.csv', 
                #'DATA/nivel_aforo/Tudela.csv']
                'DATA/nivel_aforo/castejon.csv',
                'DATA/nivel_aforo/palazuelos.csv']

FLOW_FILES = ['DATA/caudal/caudal_arroyo.csv',
                'DATA/caudal/caudal_calatayud.csv',
                'DATA/caudal/caudal_mendavia.csv',
                'DATA/caudal/caudal_miranda.csv', 
                'DATA/caudal/caudal_castejon.csv',
                'DATA/caudal/caudal_palazuelos.csv']

EMBALSE_FILES = ['DATA/volumen_embalse/ebro.csv',
                'DATA/volumen_embalse/mansilla.csv',
                'DATA/volumen_embalse/tranquera.csv',
                'DATA/volumen_embalse/yesa.csv']

"""PRECIPITATION_FILES = ['DATA/precipitaciones/arce.csv',
                        'DATA/precipitaciones/calatayud.csv',
                        'DATA/precipitaciones/ebro.csv',
                        'DATA/precipitaciones/laloteta.csv',
                        'DATA/precipitaciones/mansilla.csv',
                        'DATA/precipitaciones/romeral.csv',
                        'DATA/precipitaciones/tauste.csv',
                        'DATA/precipitaciones/tranquera.csv',
                        'DATA/precipitaciones/yesa.csv']"""

PRECIPITATION_FILES = ['DATA/precipitaciones/arce.csv',
                        'DATA/precipitaciones/calatayud.csv']

# Data parameters
NUM_FEATURES_1 = 12

NUM_FEATURES_2 = 21

NUM_FEATURES_3 = 20

# Model parameters
LAYER1_NEURONS_1 = 3
LAYER2_NEURONS_1 = 2

LAYER1_NEURONS_2 = 7
LAYER2_NEURONS_2 = 3

LAYER1_NEURONS_3 = 5
LAYER2_NEURONS_3 = 2

# Training parameters
P_TRAIN = 0.8
NUM_EPOCHS = 1000
LEARNING_RATE = 0.01

# Callback parameters
PRINT_EPOCH = 100
LOSSES_AVG_NO = 10