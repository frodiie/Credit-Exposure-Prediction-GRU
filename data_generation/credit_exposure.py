"""
@Authors: Viktor Sambergs & Isabelle Frodé
@Date: Feb 2021
"""

import pandas as pd
import numpy as np
import QuantLib as ql
import time


def short_rate(nsim: int, seq: ql.GaussianPathGenerator, nbr_gridpoints: int):
    """
    Returns all short rate paths (nbr_gridpoints x nsim).
    
    ARGS:
        nsim (int):                       nbr of MC simulations.
        seq (ql.GaussianPathGenerator):   ql PathGenerator object.
        nbr_gridpoints (int):             nbr of time gridpoints.
    """
    r = np.empty((nsim, nbr_gridpoints))
    for i in range(nsim):
        path = seq.next().value()
        value = [path[j] for j in range(len(path))]
        r[i, :] = np.array(value)
    return np.transpose(r)


def zcb_price(rate: np.ndarray, T: int, zero_rates: list, fwd_rates: list,
              gridpoints: pd.Series, nsim: int, a: float, sigma: float):
    """
    Returns zero coupon bond price P(t, T) paths (T+1 x nsim).

    ARGS:
        rate (np.ndarray):                short rate paths matrix.
        T (int):                          time to maturity (months).
        zero_rates (list):                short rates at t=0.
        fwd_rates (list):                 forward rates at t=0.
        gridpoints (pd.Series):           time gridpoints in [0, T] (years).
        nsim (int):                       nbr of MC simulations.
        a (float):                        HW1F param mean reversion.
        sigma (float):                    HW1F param volatility.
    """
    t = gridpoints[0: T+1]
    short_rates = rate[0: T+1, :]
    B = (1 - np.exp(-a * (t[T] - t[0: T+1]))) / a
    P_T = np.exp(-zero_rates[T] * t[T])
    P_t = np.exp(-np.multiply(zero_rates[0: T+1], t[0: T+1]))
    lnA = np.log(P_T / P_t) + B * fwd_rates[0: T+1] - ((0.25 * sigma ** 2) / (a * a * a)) * (
                (np.exp(-a * t[T]) - np.exp(-a * t[0: T+1])) ** 2) * (np.exp(2 * a * t[0: T+1]) - 1)
    A = np.exp(lnA)
    A = np.transpose(np.tile(A, (nsim, 1)))
    B = np.transpose(np.tile(B, (nsim, 1)))
    s = np.append(0,gridpoints[0:T])
    M = -(sigma**2/a**2)*(1-np.exp(-a*(t-s))) + (sigma**2/(2*a**2))*(np.exp(-a * (t[T] - t[0:T + 1])) - np.exp(-a * (t[T] + t[0:T + 1] - 2*s)))
    M = np.transpose(np.tile(M, (nsim, 1)))
    P_t_T = A * np.exp(-B * (short_rates + M))
    return P_t_T


def irs(rate: np.ndarray, T: int,  zero_rates, fwd_rates, gridpoints: np.ndarray,
        nsim: int, a: float, sigma: float, delta=0.5, fl_freq: int=3, N: int=100):
    """
    Returns values of a recieve fixed play floating interest rate swap (nbr_gridpoints x nsim).

    ARGS:
        rate (np.ndarray):                short rate paths matrix.
        T (int):                          time to maturity (months).
        zero_rates (list):                short rates at t=0.
        fwd_rates (list):                 forward rates at t=0.
        gridpoints (pd.Series):           time gridpoints in [0, T] (years).
        nsim (int):                       nbr of MC simulations.
        a (float):                        HW1F param mean reversion.
        sigma (float):                    HW1F param volatility.
        delta (float):                    time difference between fixed payments (years)
        fl_freq (float):                  months between floating payments
        N (int):                          notional principal (M USD)

    """
    tenor = [i for i in range(0, 12*T+1, fl_freq)]
    tenor_fix = [tenor[i] for i in np.arange(len(tenor))[2::2]]
    tenor_index = tenor
    nbr_payments = len(tenor) - 1
    zcb_prices = [zcb_price(rate,i, zero_rates, fwd_rates, gridpoints, nsim, a, sigma) for i in tenor]
    libor_rates = [(1)/(zcb_prices[i+1][tenor[i],:]) for i in range(len(zcb_prices)-1)]

    all_fix = 0
    for i in np.arange(len(tenor))[2::2]:
        all_fix = all_fix + delta * zcb_prices[i][0, :]
    all_floating = 1  - zcb_prices[-1][0, :]
    
    R = all_floating / all_fix
    V = np.empty(shape=(0, nsim))

    for j in range(0,nbr_payments):
        outstanding_fix = list(filter((tenor[j]).__lt__, tenor_fix))
        outstanding_ind = [tenor.index(i) for i in outstanding_fix]
        outstanding_fl = list(filter((tenor[j]).__le__, tenor))
        outstanding_ind_fl = [tenor.index(i) for i in outstanding_fl][1:]
        l = outstanding_ind_fl[0]
        fix_leg = sum(R*delta*zcb_prices[l][tenor_index[j] :tenor_index[j + 1] , :] for l in outstanding_ind)
        fl_leg = libor_rates[l-1]*zcb_prices[l][tenor_index[j]:tenor_index[j + 1], :] - zcb_prices[-1][tenor_index[j]:tenor_index[j + 1], :]

        V_t = N*(fl_leg - fix_leg)
        V = np.append(V, V_t, axis=0)
    final_price = np.zeros((1, nsim))
    V = np.append(V, final_price, axis=0)
    return V


def exposure(y0: np.ndarray, dates: np.ndarray, T: int, a: float, sigma: float, 
                  nsim: int, dayCounter: ql.ActualActual, nbr_gridpoints: int):
    """
    Generate pair of yield curve & credit exposure.
    
    ARGS:
        y0 (np.ndarray):                  observed yield curve.
        dates (np.ndarray):               observation dates.
        T (int):                          max maturity (years).
        a (float):                        HW1F param mean reversion.
        sigma (float):                    HW1F param volatility.
        nsim (int):                       nbr of MC simulations.
        dayCounter (ql.ActualActual):     ql Actual/Actual daycount object.
        nbr_gridpoints (int):             nbr of time gridpoints.
    """
    dates = pd.Series(dates)
    start_date = pd.Timestamp(dates[0])
    end_date = start_date + pd.DateOffset(years=T)
    all_dates_pd = pd.Series(pd.date_range(start_date- pd.DateOffset(days = start_date.day), end_date, freq = 'MS').tolist()) + pd.DateOffset(days = start_date.day-1)
    all_dates = all_dates_pd.apply(ql.Date().from_date)
    obs_dates = dates.apply(ql.Date().from_date)
    gridpoints = (all_dates_pd - start_date) / np.timedelta64(1, 'Y')
    dayCounter = ql.ActualActual()
    curve = ql.CubicZeroCurve(obs_dates, y0, dayCounter, ql.TARGET())
    curve_handle = ql.YieldTermStructureHandle(curve)
    curve.enableExtrapolation()
    zero_rates = [
        curve.zeroRate(date, dayCounter, ql.Continuous).rate()
        for date in all_dates
    ]
    fwd_rates = [
        curve.forwardRate(date, date + ql.Period('1d'), dayCounter, ql.Simple).rate()
        for date in all_dates
    ]
    hw_process = ql.HullWhiteProcess(curve_handle, a, sigma)
    rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(nbr_gridpoints-1, ql.UniformRandomGenerator()))
    seq = ql.GaussianPathGenerator(hw_process, T, nbr_gridpoints-1, rng, False)
    t0 = time.time()
    rate = short_rate(nsim, seq, nbr_gridpoints)
    t1 = time.time()
    t2 = time.time()
    irs_values = irs(rate, T, zero_rates, fwd_rates, gridpoints, nsim, a, sigma, N=N)
    max_val = np.maximum(irs_values,0)
    exposure = np.mean(max_val, axis = 1)
    t3 = time.time()
    return zero_rates, exposure.tolist()
