# -*- coding: utf-8 -*-
#variables globales de configuracion

DATA_PATH = "/DATA"
LOG_DIR = "/LOGS"

#comentados en la iteracion 1
RIVER_FILES = ['DATA/nivel_aforo/Arroyo.csv',
                'DATA/nivel_aforo/Calatayud.csv',
                'DATA/nivel_aforo/Aragon.csv',
                'DATA/nivel_aforo/Mendavia.csv',
                'DATA/nivel_aforo/Miranda.csv', 
                'DATA/nivel_aforo/castejon.csv',
                'DATA/nivel_aforo/palazuelos.csv']

FLOW_FILES = ['DATA/caudal/caudal_arroyo.csv',
                'DATA/caudal/caudal_aragon.csv',
                'DATA/caudal/caudal_calatayud.csv',
                'DATA/caudal/caudal_mendavia.csv',
                'DATA/caudal/caudal_miranda.csv', 
                'DATA/caudal/caudal_castejon.csv',
                'DATA/caudal/caudal_palazuelos.csv']

EMBALSE_FILES = ['DATA/volumen_embalse/ebro.csv',
                'DATA/volumen_embalse/mansilla.csv',
                'DATA/volumen_embalse/tranquera.csv',
                'DATA/volumen_embalse/yesa.csv']

PRECIPITATION_FILES = ['DATA/precipitaciones/arce.csv',
                        'DATA/precipitaciones/calatayud.csv',
                        'DATA/precipitaciones/anguiamo.csv',
                        'DATA/precipitaciones/caparroso.csv',
                        'DATA/precipitaciones/leza.csv',
                        'DATA/precipitaciones/onsella.csv',
                        'DATA/precipitaciones/sobron.csv',
                        'DATA/precipitaciones/soto.csv']

# Data parameters
NUM_FEATURES = 18 # base-> 18 mejora1-28 prebase-> 14

# Model parameters
LAYER1_NEURONS = 13 #base-> 7 mejora1-> 9 prebase-> 5
LAYER2_NEURONS = 11 #base-> 6 mejora1-> 3 prebase-> 4

# Training parameters
P_TRAIN = 0.8
NUM_EPOCHS = 20000 #base-> 2875 mejora1-> 20000
LEARNING_RATE = 0.001

# Callback parameters
PRINT_EPOCH = 10
LOSSES_AVG_NO = 10

ITER = "base" # this value can be: base or mejora1