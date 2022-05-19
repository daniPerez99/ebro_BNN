#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys 
import math

# UNCERTAINTY FUNCTIONS
# =============================================================================

# H --> PREDICTIVE (ALEATORIC + EPISTEMIC)
def _predictive_entropy(prediction):
    
    _, num_pixels, num_classes = prediction.shape
    
    entropy = np.zeros(num_pixels)
    for p in range(num_pixels):
        for c in range(num_classes):
            avg = np.mean(prediction[..., p, c])
            if avg == 0.0:
                avg = sys.float_info.min
            entropy[p] += avg * math.log(avg)
    
    return -1 * entropy

# Ep --> ALEATORIC
def _expected_entropy(prediction):
    
    num_tests, num_pixels, num_classes = prediction.shape
    
    entropy = np.zeros(num_pixels)
    for p in range(num_pixels):
        for t in range(num_tests):
            class_sum = 0
            for c in range(num_classes):
                val = prediction[t][p][c]
                if val == 0.0:
                    val = sys.float_info.min
                class_sum += val * math.log(val)
            entropy[p] -= class_sum
    
    return entropy / num_tests