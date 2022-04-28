"""
@Authors: Viktor Sambergs & Isabelle Frodé
@Date: Feb 2022
"""


import numpy as np
import pandas as pd
import random


from scipy import interpolate


def move_in_parallel(yield_curve: np.ndarray, offset: float, normal_dist: bool=True):
    """
    Moves yield curve 'offset' in parallel.
    """
    if normal_dist:
        if offset < 0:
            offset = -offset
        offset = np.random.normal(0, offset, 1)[0]
    y0 = yield_curve + offset
    for i in range(0,len(yield_curve)):
        if (y0[i]) < 0 and (y0[i] > offset/4): 
            y0[i] = offset/10
    if (min(y0) < 0) or (max(y0) > 0.06):
        y0 = move_in_parallel(yield_curve, offset/2)
    return y0


def multiply_curves(yield_curve1: np.ndarray, yield_curve2: np.ndarray, ttm1: np.ndarray, 
                         ttm2: np.ndarray):
    """
    Multiplies and takes the square root of two curves (element wise).
    """
    f = interpolate.interp1d(ttm2, yield_curve2, fill_value='extrapolate')
    yield_curve2 = f(ttm1)
    return np.sqrt(np.multiply(yield_curve1, yield_curve2))


def tilt_curve(yield_curve: np.ndarray, tilt: float, offset: float, normal_dist: bool=True):
    """
    Tilts yield curve.
    """
    if normal_dist:
        tilt = np.random.normal(0, np.abs(tilt), 1)[0]
        if tilt < 0:
            tilt = tilt/10
    line = np.linspace(0, tilt, len(yield_curve))
    y0 = yield_curve + line
    for i in range(0,len(yield_curve)):
        if (y0[i]) < 0 and (y0[i] > offset/4): 
            y0[i] = tilt/10
    if (min(y0) <= 0) or (max(y0) > 0.06):
        y0 = tilt_curve(yield_curve, tilt/2)
    return y0


def random_curve(df: pd.DataFrame):
    """
    Returns a random curve from a pd.DataFrame df.
    """
    random_curve = random.randint(0, max(df['curve']))
    y0_r = df[df['curve'] == random_curve]['INTEREST_PCT'].values
    ttm_r = df.loc[df['curve'] == random_curve]['TTM']
    return y0_r, ttm_r
