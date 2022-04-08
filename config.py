#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#variables globales de configuracion

DATA_PATH = "/DATA"
LOG_DIR = "/LOGS"

# Data parameters
NUM_FEATURES = 12

# Model parameters
LAYER1_NEURONS = 5
LAYER2_NEURONS = 2

# Training parameters
P_TRAIN = 0.8
NUM_EPOCHS = 1000
LEARNING_RATE = 0.01

# Callback parameters
PRINT_EPOCH = 100
LOSSES_AVG_NO = 10