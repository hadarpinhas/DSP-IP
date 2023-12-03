import pandas as pd
import os
import csv
from matplotlib import pyplot as plt

basePath = r"/home/yossi/Documents/database/hadar"

relPath = r"drones/f4c2b4n755-1/DroneRF/AR_drone/RF_Data_10100_H/10100H_0.csv"

dataCsvPath = os.path.join(basePath , relPath) 

rfData = None
with open(dataCsvPath,"r") as f:
    rfData = f.read().split(',')

rfData[0:1000]

print(f"{dataCsvPath=}") 

data_pd = pd.read_csv(dataCsvPath)

print(f"{type(data_pd)}=")

print(data_pd.head())