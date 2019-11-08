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
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import statsmodels.stats.multitest as smm
import time
import copy
import smeutils

smePack = importr('sme', lib_loc="C:/Users/user/Documents/R/win-library/3.6")
statsPack = importr('stats')

# Settings ====================================================================

# Input files
info = pd.read_csv('./sme_info.csv')
data = pd.read_csv('./sme_data.csv')

info_arr = np.array(info)
data_fid = np.array(data.columns)
data_arr = np.array(data)
selIdx = np.arange(len(data_fid))

# Parameters
RUN = True
N = 12                                  # Number of subjects
t_n = 4                                 # Number of time points
iplN = 100                              # Number of interpolated time points
n_bootstrap = 500                       # Number of bootstrap sampling
selIdx = selIdx[:]                      # List of metabolites to analyze
relative = False                        # Scale data to initial values
correctOutlier = False
SAVE = False
USEMEAN = True

# SME Parameters
ctra = "AICc"                           # Criteria
init_l_mc = 1e-8                        # Initial lambda_mu
init_l_vc = 1e-8                        # Initial lambda_v
init_l_mt = 5e-8                        # Initial lambda_mu
init_l_vt = 5e-8                        # Initial lambda_v
maxIter = 100000                        # Maximum iteration
deltaEM = 1e-3                          # Threshold for expetation maximization
deltaNM = 1e-3                          # Threshold for nelder mead
normalizeTime = True

seed = 1234                             # RNG seed

showFig = False                         # Flag to plot figures
figSize = (20,16)                       # Size of figures
plotLegend = False                      # Flag to plot legend
colorMap = 'viridis'                    # kwarg for colormap
plotSMEMeanOnly = False                 # Only plot SME mean trace
mergePlot = True                        # Merge multiple plots
plotHeatmap = False                     # Plot heatmap comparing two data groups

t = np.array([1,3,5,7])
iplT = np.linspace(1, 7, iplN)
iplTIdx = np.empty(t_n)

for i in range(t_n):
    iplTIdx[i] = np.where(iplT == t[i])[0]
iplTIdx = iplTIdx.astype(int)

sel = np.array([data_fid[selIdx]]).flatten() 

#==============================================================================

np.random.seed(seed) # Set seed

#==============================================================================

if relative:
    data = smeutils.normalizeData(data, N, t_n, data_fid)

#==============================================================================

t0 = time.time()

fulldataRaw = pd.concat([info,data], axis=1)
fulldataRaw = fulldataRaw.astype('float64')

fulldata = copy.deepcopy(fulldataRaw)
fulldata = fulldata.drop(fulldata.index[16]) # ind 5 has an outlier

if correctOutlier:
    fulldata = smeutils.correctOutlier(fulldata, sel, t, t_n)

# Initialize ==================================================================

grp0_f = fulldata[(fulldata.grp == 0)]['ind']
grp1_f = fulldata[(fulldata.grp == 1)]['ind']
grp0 = np.unique(fulldata[(fulldata.grp == 0)]['ind'])
grp1 = np.unique(fulldata[(fulldata.grp == 1)]['ind'])

pandas2ri.activate()
fd_ri = pandas2ri.py2ri(fulldata)

fd_rigrp0 = fd_ri.rx(fd_ri.rx2("grp").ro == 0, True)
fd_rigrp1 = fd_ri.rx(fd_ri.rx2("grp").ro == 1, True)

fd_rigrp0tme = fd_rigrp0.rx2("tme")
fd_rigrp0ind = fd_rigrp0.rx2("ind")
fd_rigrp1tme = fd_rigrp1.rx2("tme")
fd_rigrp1ind = fd_rigrp1.rx2("ind")

ys0mu = np.empty((len(sel), iplN))
ys1mu = np.empty((len(sel), iplN))
ys0vHat = np.empty((len(sel), len(grp0), iplN))
ys1vHat = np.empty((len(sel), len(grp1), iplN))

l2 = np.empty(len(sel))
se = np.empty(len(sel))
se0 = np.empty((len(sel), len(grp0)))
se1 = np.empty((len(sel), len(grp1)))
sem = np.empty(len(sel))
tval = np.empty(len(sel))

ys0v = np.empty((len(sel), len(grp0), t_n))
ys1v = np.empty((len(sel), len(grp1), t_n))
ys0eta = np.empty((len(sel), len(grp0), t_n))
ys1eta = np.empty((len(sel), len(grp1), t_n))

ys0mubs = np.empty((n_bootstrap, len(sel), iplN))
ys1mubs = np.empty((n_bootstrap, len(sel), iplN))
ys0vHatbs = np.empty((n_bootstrap, len(sel), len(grp0), iplN))
ys1vHatbs = np.empty((n_bootstrap, len(sel), len(grp1), iplN))

l2bs = np.empty((n_bootstrap, len(sel)))
sebs = np.empty((n_bootstrap, len(sel)))
se0bs = np.empty((n_bootstrap, len(sel), len(grp0)))
se1bs = np.empty((n_bootstrap, len(sel), len(grp1)))
sembs = np.empty((n_bootstrap, len(sel)))
tvalbs = np.empty((n_bootstrap, len(sel)))
pval = np.empty(len(sel))

t1 = time.time()

print(t1 - t0)


# SME =========================================================================

if RUN:
    for m_i in range(len(sel)):
        fd_rigrp0obj = fd_rigrp0.rx2(sel[m_i])
        fd_rigrp1obj = fd_rigrp1.rx2(sel[m_i])
        
        fit0 = smePack.sme(fd_rigrp0obj, 
                           fd_rigrp0tme, 
                           fd_rigrp0ind, 
                           criteria=ctra, 
                           maxIter=maxIter, 
                           deltaEM=deltaEM, 
                           deltaNM=deltaNM, 
                           initial_lambda_mu=init_l_mc, 
                           initial_lambda_v=init_l_mc,
                           normalizeTime=normalizeTime)
        fit1 = smePack.sme(fd_rigrp1obj, 
                           fd_rigrp1tme, 
                           fd_rigrp1ind, 
                           criteria=ctra, 
                           maxIter=maxIter, 
                           deltaEM=deltaEM, 
                           deltaNM=deltaNM, 
                           initial_lambda_mu=init_l_mt, 
                           initial_lambda_v=init_l_vt,
                           normalizeTime=normalizeTime)
        
        fit0coef = np.array(fit0.rx2('coefficients'))
        fit1coef = np.array(fit1.rx2('coefficients'))
        
        spl0mu = interpolate.CubicSpline(t, fit0coef[0], bc_type='natural')
        ys0mu[m_i] = spl0mu(iplT)
        spl1mu = interpolate.CubicSpline(t, fit1coef[0], bc_type='natural')
        ys1mu[m_i] = spl1mu(iplT)
        
        l2[m_i] = np.sqrt(np.trapz(np.square(ys0mu[m_i] - ys1mu[m_i]), x=iplT))
        
        for g0 in range(len(grp0)):
            spl0 = interpolate.CubicSpline(t, fit0coef[g0 + 1] + fit0coef[0], bc_type='natural')
            ys0vHat[m_i][g0] = spl0(iplT)
            ys0v[m_i][g0] = ys0mu[m_i][iplTIdx] - ys0vHat[m_i][g0][iplTIdx]
            ys0eta[m_i][g0] = fulldataRaw.loc[fulldataRaw.ind == grp0[g0], sel[m_i]] - ys0vHat[m_i][g0][iplTIdx]
            se0[m_i][g0] = np.trapz(np.square(ys0mu[m_i] - ys0vHat[m_i][g0]), x=iplT)
            
        for g1 in range(len(grp1)):
            spl1 = interpolate.CubicSpline(t, fit1coef[g1 + 1] + fit1coef[0], bc_type='natural')
            ys1vHat[m_i][g1] = spl1(iplT)
            ys1v[m_i][g1] = ys1mu[m_i][iplTIdx] - ys1vHat[m_i][g1][iplTIdx]
            ys1eta[m_i][g1] = fulldataRaw.loc[fulldataRaw.ind == grp1[g1], sel[m_i]] - ys1vHat[m_i][g1][iplTIdx]
            se1[m_i][g1] = np.trapz(np.square(ys1mu[m_i] - ys1vHat[m_i][g1]), x=iplT)
        
        se[m_i] = np.sqrt(np.mean(se0[m_i])/len(grp0) + np.mean(se1[m_i])/len(grp1))
    
    sem = 0.
    
    tval = np.divide(l2, se + sem)
    
    ys0vFlat = ys0v.reshape((ys0v.shape[0], -1))
    ys0etaFlat = ys0eta.reshape((ys0eta.shape[0], -1))
    ys0etaFlat = np.delete(ys0etaFlat, 13, 1) # ind 5 has an outlier
    ys1vFlat = ys1v.reshape((ys1v.shape[0], -1))
    ys1etaFlat = ys1eta.reshape((ys1eta.shape[0], -1))
    
    t2 = time.time()
    print(t2 - t1)
    
    
# Bootstrapping ===============================================================
    
    fulldataS = []
    
    for bcount in range(n_bootstrap):
        
        print("Bootstrap run: " + str(bcount))
        
        fulldataC = copy.deepcopy(fulldataRaw)
        
        for m_i in range(len(sel)):
            if USEMEAN:
                for Di in range(N):
                    ysmuMean = (ys0mu[m_i][iplTIdx] + ys1mu[m_i][iplTIdx])/2
                    if Di in grp0:
                        fulldataC[sel[m_i]][np.arange(0,t_n*N,N)+Di] = (ysmuMean 
                                 + np.random.choice(ys0vFlat[m_i], size=t_n) 
                                 + np.random.choice(ys0etaFlat[m_i], size=t_n))
                    else:
                        fulldataC[sel[m_i]][np.arange(0,t_n*N,N)+Di] = (ysmuMean 
                                 + np.random.choice(ys1vFlat[m_i], size=t_n) 
                                 + np.random.choice(ys1etaFlat[m_i], size=t_n))
            else:
                ct_rand = np.random.rand()
                for Di in range(N):
                    if ct_rand < 0.5:
                        if Di in grp0:
                            fulldataC[sel[m_i]][np.arange(0,t_n*N,N)+Di] = (ys0mu[m_i][iplTIdx] 
                            + np.random.choice(ys0vFlat[m_i], size=t_n) 
                            + np.random.choice(ys0etaFlat[m_i], size=t_n))
                        else:
                            fulldataC[sel[m_i]][np.arange(0,t_n*N,N)+Di] = (ys0mu[m_i][iplTIdx] 
                            + np.random.choice(ys1vFlat[m_i], size=t_n) 
                            + np.random.choice(ys1etaFlat[m_i], size=t_n))
                    else:
                        if Di in grp0:
                            fulldataC[sel[m_i]][np.arange(0,t_n*N,N)+Di] = (ys1mu[m_i][iplTIdx] 
                            + np.random.choice(ys0vFlat[m_i], size=t_n) 
                            + np.random.choice(ys0etaFlat[m_i], size=t_n))
                        else:
                            fulldataC[sel[m_i]][np.arange(0,t_n*N,N)+Di] = (ys1mu[m_i][iplTIdx]
                            + np.random.choice(ys1vFlat[m_i], size=t_n)
                            + np.random.choice(ys1etaFlat[m_i], size=t_n))
    
        fulldataC = fulldataC.drop(fulldataC.index[16]) # ind 5 has an outlier
        fulldataS.append(fulldataC)
        
        fd_ri = pandas2ri.py2ri(fulldataC)
        
        fd_rigrp0 = fd_ri.rx(fd_ri.rx2("grp").ro == 0, True)
        fd_rigrp1 = fd_ri.rx(fd_ri.rx2("grp").ro == 1, True)
        
        for m_i in range(len(sel)):
            fd_rigrp0objbs = fd_rigrp0.rx2(sel[m_i])
            fd_rigrp1objbs = fd_rigrp1.rx2(sel[m_i])
            
            fit0 = smePack.sme(fd_rigrp0objbs, 
                               fd_rigrp0tme, 
                               fd_rigrp0ind, 
                               criteria=ctra, 
                               maxIter=maxIter, 
                               deltaEM=deltaEM, 
                               deltaNM=deltaNM, 
                               initial_lambda_mu=init_l_mc, 
                               initial_lambda_v=init_l_vc,
                               normalizeTime=normalizeTime)
            fit1 = smePack.sme(fd_rigrp1objbs, 
                               fd_rigrp1tme, 
                               fd_rigrp1ind, 
                               criteria=ctra, 
                               maxIter=maxIter, 
                               deltaEM=deltaEM, 
                               deltaNM=deltaNM, 
                               initial_lambda_mu=init_l_mt, 
                               initial_lambda_v=init_l_vt,
                               normalizeTime=normalizeTime)
            
            fit0coefbs = np.array(fit0.rx2('coefficients'))
            fit1coefbs = np.array(fit1.rx2('coefficients'))
            
            spl0mubs = interpolate.CubicSpline(t, fit0coefbs[0], bc_type='natural')
            ys0mubs[bcount][m_i] = spl0mubs(iplT)
            spl1mubs = interpolate.CubicSpline(t, fit1coefbs[0], bc_type='natural')
            ys1mubs[bcount][m_i] = spl1mubs(iplT)
            
            l2bs[bcount][m_i] = np.sqrt(np.trapz(np.square(ys0mubs[bcount][m_i] - ys1mubs[bcount][m_i]), x=iplT))
            
            for g0 in range(len(grp0)):
                spl0bs = interpolate.CubicSpline(t, fit0coefbs[g0 + 1] + fit0coefbs[0], bc_type='natural')
                ys0vHatbs[bcount][m_i][g0] = spl0bs(iplT)
                se0bs[bcount][m_i][g0] = np.trapz(np.square(ys0mubs[bcount][m_i] - ys0vHatbs[bcount][m_i][g0]), x=iplT)
                
            for g1 in range(len(grp1)):
                spl1bs = interpolate.CubicSpline(t, fit1coefbs[g1 + 1] + fit1coefbs[0], bc_type='natural')
                ys1vHatbs[bcount][m_i][g1] = spl1bs(iplT)
                se1bs[bcount][m_i][g1] = np.trapz(np.square(ys1mubs[bcount][m_i] - ys1vHatbs[bcount][m_i][g1]), x=iplT)
            
            sebs[bcount][m_i] = np.sqrt(np.mean(se0bs[bcount][m_i])/len(grp0) + np.mean(se1bs[bcount][m_i])/len(grp1))
    
        sembs = 0.
        
        tvalbs[bcount] = np.divide(l2bs[bcount], sebs[bcount] + sembs)
    

    t3 = time.time()
    print(t3 - t2)    
    
    for m_i in range(len(sel)):
        pval[m_i] = (tvalbs[:,m_i] >= tval[m_i]).sum()/n_bootstrap
    
    pvalCorr = smm.multipletests(pval, alpha=0.05, method='fdr_bh')[1]
    
    print('p-value: ' + str(len(np.where(pval <= 0.05)[0])))
    print(np.where(pval <= 0.05)[0])


# Plotting ====================================================================

cmap1 = cm.get_cmap(colorMap, 2)
cmap2 = cm.get_cmap(colorMap, N)
cmap3 = cm.get_cmap(colorMap, len(sel))
cmap_grp0 = cm.get_cmap('viridis', len(grp0))
cmap_grp1 = cm.get_cmap('viridis', len(grp1))


def plotC(idx):
    """
    Plots data points, individual, and mean curve of control group
    
    :param idx: index of the selection
    """
    
    fdgrp0tme_arr = np.array(fulldata[fulldata.grp == 0]["tme"])
    fdgrp0sel_arr = np.array(fulldata[fulldata.grp == 0][sel])
    
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
    Plots data points, individual, and mean curve of treatment group
    
    :param idx: index of the selection
    """
    
    fdgrp1tme_arr = np.array(fulldata[fulldata.grp == 1]["tme"])
    fdgrp1sel_arr = np.array(fulldata[fulldata.grp == 1][sel])
    
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
    Plots data points, individual, and mean curve of both control and treatment group
    
    :param idx: index of the selection
    """
        
    fdgrp0tme_arr = np.array(fulldata[fulldata.grp == 0]["tme"])
    fdgrp0sel_arr = np.array(fulldata[fulldata.grp == 0][sel])
    
    fdgrp1tme_arr = np.array(fulldata[fulldata.grp == 1]["tme"])
    fdgrp1sel_arr = np.array(fulldata[fulldata.grp == 1][sel])
    
    plt.figure(figsize=figSize)
    
    if not plotSMEMeanOnly:
        for g0 in range(len(grp0)):
            tmeIdx = np.where(grp0_f == grp0[g0])
            plt.plot(fdgrp0tme_arr[tmeIdx], fdgrp0sel_arr[:,idx][tmeIdx], color=cmap1(0), marker='o', linestyle='')
            plt.plot(iplT, ys0vHat[idx][g0], color=cmap1(0), linestyle='dashed')
        for g1 in range(len(grp1)):
            tmeIdx = np.where(grp1_f == grp1[g1])
            plt.plot(fdgrp1tme_arr[tmeIdx], fdgrp1sel_arr[:,idx][tmeIdx], color=cmap1(1), marker='o', linestyle='')
            plt.plot(iplT, ys1vHat[idx][g1], color=cmap1(len(sel)), linestyle='dashed')
    
    plt.plot(iplT, ys0mu[idx], lw=3, color=cmap1(0))
    plt.plot(iplT, ys1mu[idx], lw=3, color=cmap1(1))
    plt.show()


def plotCTbs(bcount, idx):
    """
    Plots data points, individual, and mean curve of both control and treatment group for a bootstrapping sample
    
    :param bcount: index of bootstrapping sample
    :param idx: index of the selection
    """    
    
    fdgrp0tme_arr = np.array(fulldataS[bcount][fulldataS[bcount].grp == 0]["tme"])
    fdgrp0sel_arr = np.array(fulldataS[bcount][fulldataS[bcount].grp == 0][sel])
    
    fdgrp1tme_arr = np.array(fulldataS[bcount][fulldataS[bcount].grp == 1]["tme"])
    fdgrp1sel_arr = np.array(fulldataS[bcount][fulldataS[bcount].grp == 1][sel])
    
    plt.figure(figsize=figSize)
    
    if not plotSMEMeanOnly:
        for g0 in range(len(grp0)):
            tmeIdx = np.where(grp0_f == grp0[g0])
            plt.plot(fdgrp0tme_arr[tmeIdx], fdgrp0sel_arr[:,idx][tmeIdx], color=cmap1(0), marker='o', linestyle='')
            plt.plot(iplT, ys0vHatbs[bcount][idx][g0], color=cmap1(0), linestyle='dashed')
        for g1 in range(len(grp1)):
            tmeIdx = np.where(grp1_f == grp1[g1])
            plt.plot(fdgrp1tme_arr[tmeIdx], fdgrp1sel_arr[:,idx][tmeIdx], color=cmap1(1), marker='o', linestyle='')
            plt.plot(iplT, ys1vHatbs[bcount][idx][g1], color=cmap1(len(sel)), linestyle='dashed')
    
    plt.plot(iplT, ys0mubs[bcount][idx], lw=3, color=cmap1(0))
    plt.plot(iplT, ys1mubs[bcount][idx], lw=3, color=cmap1(1))
    plt.show()

def exportOutput(path=None):
        """
        Export an output to specified path
        
        """
        
        if path:
            outputdir = path
        else:
            outputdir = os.path.join(os.getcwd(), 'output')
            
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        
        fulldataRaw.to_csv(os.path.join(outputdir, 'fulldataRaw.csv'))
        fulldata.to_csv(os.path.join(outputdir, 'fulldata.csv'))
        
        df = pd.DataFrame(ys0mu)
        df.to_csv(os.path.join(outputdir, 'ys0mu.csv'))
        df = pd.DataFrame(ys1mu)
        df.to_csv(os.path.join(outputdir, 'ys1mu.csv'))
        
        if not os.path.exists(os.path.join(outputdir, 'ys0vHat')):
            os.mkdir(os.path.join(outputdir, 'ys0vHat'))
        if not os.path.exists(os.path.join(outputdir, 'ys1vHat')):
            os.mkdir(os.path.join(outputdir, 'ys1vHat'))
        
        for i in range(len(ys0vHat)):
            df1 = pd.DataFrame(ys0vHat[i])
            df1.to_csv(os.path.join(os.path.join(outputdir, 'ys0vHat'), 'ys0vHat_' + str(i) + '.csv'))
            df2 = pd.DataFrame(ys1vHat[i])
            df2.to_csv(os.path.join(os.path.join(outputdir, 'ys1vHat'), 'ys1vHat_' + str(i) + '.csv'))

        df = pd.DataFrame(pval)
        df.to_csv(os.path.join(outputdir, 'pval.csv'))
        
if RUN and SAVE:
    exportOutput()


