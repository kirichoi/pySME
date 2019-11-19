# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 Kiri Choi

pySME is a Python script to run R SME package 
(https://cran.r-project.org/web/packages/sme/index.html). SME package generates
smoothing-splines mixed-effects models from metabolomics data. This script 
follows methodology given by Berk et al. (2011) and utilizes bootstrapping to 
approximate p-values. Running this script requires R with SME package installed.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

path = './output_leukemia'

# Parameters
N = 12 # Number of subjects
t_n = 5 # Number of time points
iplN = 121 # Number of interpolated time points

showFig = False # Flag to plot figures
figSize = (20,16) # Size of figures
plotLegend = False # Flag to plot legend
colorMap = 'viridis' # kwarg for colormap
plotSMEMeanOnly = False # Only plot SME mean trace
mergePlot = True # Merge multiple plots
plotHeatmap = False # Plot heatmap comparing two data groups

t = np.array([1,2,4,6,7])
iplT = np.linspace(1, 7, iplN)

fulldata = pd.read_csv(os.path.join(path, 'fulldata.csv')).iloc[:, 1:]
data_fid = np.array(fulldata.columns[3:])

grp0_f = fulldata[(fulldata.grp == 0)]['ind']
grp1_f = fulldata[(fulldata.grp == 1)]['ind']
grp0 = np.unique(fulldata[(fulldata.grp == 0)]['ind'])
grp1 = np.unique(fulldata[(fulldata.grp == 1)]['ind'])

ys0mu = np.array(pd.read_csv(os.path.join(path, 'ys0mu.csv')).iloc[:, 1:])
ys1mu = np.array(pd.read_csv(os.path.join(path, 'ys1mu.csv')).iloc[:, 1:])
ys0vHat = np.empty((len(data_fid), len(grp0), iplN))
ys0vHatDir = os.path.join(path, 'ys0vHat')
ys0vHatF = [f for f in os.listdir(ys0vHatDir) if os.path.isfile(os.path.join(ys0vHatDir, f))]
ys0vHatF.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
ys1vHat = np.empty((len(data_fid), len(grp1), iplN))
ys1vHatDir = os.path.join(path, 'ys1vHat')
ys1vHatF = [f for f in os.listdir(ys1vHatDir) if os.path.isfile(os.path.join(ys1vHatDir, f))]
ys1vHatF.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

for i in range(len(ys0vHatF)):
    ys0vHat[i] = pd.read_csv(os.path.join(ys0vHatDir, ys0vHatF[i])).iloc[:, 1:]
    
for i in range(len(ys1vHatF)):
    ys1vHat[i] = pd.read_csv(os.path.join(ys1vHatDir, ys1vHatF[i])).iloc[:, 1:]

cmap1 = cm.get_cmap(colorMap, 2)
cmap2 = cm.get_cmap(colorMap, N)
cmap3 = cm.get_cmap(colorMap, len(data_fid))
cmap_grp0 = cm.get_cmap('viridis', len(grp0))
cmap_grp1 = cm.get_cmap('viridis', len(grp1))


def plotC(idx):
    """
    Plots data points, individual, and mean curves of the control group
    
    :param idx: index of the selection list
    """
    
    fdgrp0tme_arr = np.array(fulldata[fulldata.grp == 0]["tme"])
    fdgrp0sel_arr = np.array(fulldata[fulldata.grp == 0][data_fid])
    
    plt.figure(figsize=figSize)
    
    if not plotSMEMeanOnly:
        for g0 in range(len(grp0)):
            tmeIdx = np.where(grp0_f == grp0[g0])
            plt.plot(fdgrp0tme_arr[tmeIdx], fdgrp0sel_arr[:,idx][tmeIdx], color=cmap_grp0(g0), marker='o', linestyle='')
            plt.plot(iplT, ys0vHat[idx][g0], color=cmap_grp0(g0), linestyle='dashed')
    
    plt.plot(iplT, ys0mu[idx], lw=3, color=cmap1(0))
    plt.show()


def plotT(idx):
    """
    Plots data points, individual, and mean curves of the treatment group
    
    :param idx: index of the selection list
    """
    
    fdgrp1tme_arr = np.array(fulldata[fulldata.grp == 1]["tme"])
    fdgrp1sel_arr = np.array(fulldata[fulldata.grp == 1][data_fid])
    
    plt.figure(figsize=figSize)
    
    if not plotSMEMeanOnly:
        for g1 in range(len(grp1)):
            tmeIdx = np.where(grp1_f == grp1[g1])
            plt.plot(fdgrp1tme_arr[tmeIdx], fdgrp1sel_arr[:,idx][tmeIdx], color=cmap_grp1(g1), marker='o', linestyle='')
            plt.plot(iplT, ys1vHat[idx][g1], color=cmap_grp1(g1), linestyle='dashed')
    
    plt.plot(iplT, ys1mu[idx], lw=3, color=cmap1(1))
    plt.show()


def plotCT(idx):
    """
    Plots data points, individual, and mean curves of both the control and the treatment group
    
    :param idx: index of the selection list
    """
        
    fdgrp0tme_arr = np.array(fulldata[fulldata.grp == 0]["tme"])
    fdgrp0sel_arr = np.array(fulldata[fulldata.grp == 0][data_fid])
    
    fdgrp1tme_arr = np.array(fulldata[fulldata.grp == 1]["tme"])
    fdgrp1sel_arr = np.array(fulldata[fulldata.grp == 1][data_fid])
    
    plt.figure(figsize=figSize)
    
    if not plotSMEMeanOnly:
        for g0 in range(len(grp0)):
            tmeIdx = np.where(grp0_f == grp0[g0])
            plt.plot(fdgrp0tme_arr[tmeIdx], fdgrp0sel_arr[:,idx][tmeIdx], color=cmap1(0), marker='o', linestyle='')
            plt.plot(iplT, ys0vHat[idx][g0], color=cmap1(0), linestyle='dashed')
        for g1 in range(len(grp1)):
            tmeIdx = np.where(grp1_f == grp1[g1])
            plt.plot(fdgrp1tme_arr[tmeIdx], fdgrp1sel_arr[:,idx][tmeIdx], color=cmap1(1), marker='o', linestyle='')
            plt.plot(iplT, ys1vHat[idx][g1], color=cmap1(len(data_fid)), linestyle='dashed')
    
    plt.plot(iplT, ys0mu[idx], lw=3, color=cmap1(0))
    plt.plot(iplT, ys1mu[idx], lw=3, color=cmap1(1))
    plt.show()

    