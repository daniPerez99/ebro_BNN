#!/bin/bash
#python ajustarModelo.py SGD 2>/dev/null 1>LOGS/LearningRate0_001/SGD.txt
python ajustarModelo.py Adam 2>/dev/null 1>LOGS/LearningRate0_001/Adam.txt
python ajustarModelo.py RMSprop 2>/dev/null 1>LOGS/LearningRate0_001/RMSprop.txt