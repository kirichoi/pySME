# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 Kiri Choi

pySME is a Python script to run R SME package 
(https://cran.r-project.org/web/packages/sme/index.html). SME package generates
smoothing-splines mixed-effects models from metabolomics data. This script 
follows methodology given by Berk et al. (2011) and utilizes bootstrapping to 
approximate p-values. Running this script requires R with SME package installed.

"""

import numpy as np
import pandas as pd

def correctOutlier(fulldata, sel, t, t_n):
    """
    Attempt to correct outliers using least squares polynomial fit.
    Outliers are detected using inter-quartile range (IQR).
    
    """
    cCount = 0
    for m_i in range(len(sel)):
        for g in range(2):
            for tidx in range(t_n):
                ridx = np.where((fulldata["tme"] == t[tidx]) & (fulldata["grp"] == g))[0]
                tgtarr = fulldata.iloc[ridx, m_i+3]
                q1 = tgtarr.quantile(0.25)
                q3 = tgtarr.quantile(0.75)
                iqr = q3 - q1
                o1idx = tgtarr[tgtarr < (q1 - 2.22*iqr)].index
                o2idx = tgtarr[tgtarr > (q3 + 2.22*iqr)].index
                if len(o1idx) > 0:
                    for o1 in range(len(o1idx)):
                        cCount += 1
                        t1idx = np.unique(fulldata.tme[(fulldata.tme != fulldata.iloc[o1idx[o1]].tme)])
                        o1arr = fulldata.loc[(fulldata.ind == fulldata.iloc[o1idx[o1]].ind) & (fulldata.tme != fulldata.iloc[o1idx[o1]].tme)][sel[m_i]]
                        z1 = np.polyfit(t1idx, o1arr, deg=2)
                        p1 = np.poly1d(z1)
                    fulldata.iloc[o1idx[o1], m_i+3] = p1(fulldata.iloc[o1idx[o1]].tme)
                if len(o2idx) > 0:
                    for o2 in range(len(o2idx)):
                        cCount += 1
                        t2idx = np.unique(fulldata.tme[(fulldata.tme != fulldata.iloc[o2idx[o2]].tme)])
                        o2arr = fulldata.loc[(fulldata.ind == fulldata.iloc[o2idx[o2]].ind) & (fulldata.tme != fulldata.iloc[o2idx[o2]].tme)][sel[m_i]]
                        z2 = np.polyfit(t2idx, o2arr, deg=2)
                        p2 = np.poly1d(z2)
                        fulldata.iloc[o2idx[o2], m_i+3] = p2(fulldata.iloc[o2idx[o2]].tme)

    print(str(cCount) + " outliers corrected")
    return fulldata


def normalizeData(data, nInd, t_n, data_fid):
    
    test = np.array(data)
    
    for i in range(t_n-1):
        test[(i+1)*nInd:(i+2)*nInd,:] = np.divide(test[(i+1)*nInd:(i+2)*nInd,:], test[:nInd,:])
    test[:nInd,:] = np.divide(test[:nInd,:], test[:nInd,:])
    
    data = pd.DataFrame(test, columns=data_fid)
    
    return data


