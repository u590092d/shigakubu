#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
__all__ = [
    "autocorr",
    "LevinsonDurbin",
    "preEmphasis"
]
def preEmphasis(wave, p=0.97):
    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, wave)
def autocorr(x, nlags=None):
    """自己相関関数を求める
    x:     信号
    nlags: 自己相関関数のサイズ（lag=0からnlags-1まで）
           引数がなければ（lag=0からlen(x)-1まですべて）
    """
    N = len(x)
    if nlags == None: nlags = N
    r = np.zeros(nlags)
    for lag in range(nlags):
        for n in range(N - lag):
            r[lag] += x[n] * x[n + lag]
    return r

def LevinsonDurbin(r, lpcOrder):
    """Levinson-Durbinのアルゴリズム
    k次のLPC係数からk+1次のLPC係数を再帰的に計算して
    LPC係数を求める"""
    # LPC係数（再帰的に更新される）
    # a[0]は1で固定のためlpcOrder個の係数を得るためには+1が必要
    a = np.zeros(lpcOrder + 1)
    e = np.zeros(lpcOrder + 1)

    # k = 1の場合
    a[0] = 1.0
    a[1] = - r[1] / r[0]
    e[1] = r[0] + r[1] * a[1]
    lam = - r[1] / r[0]

    # kの場合からk+1の場合を再帰的に求める
    for k in range(1, lpcOrder):
        # lambdaを更新
        lam = 0.0
        for j in range(k + 1):
            lam -= a[j] * r[k + 1 - j]
        lam /= e[k]

        # aを更新
        # UとVからaを更新
        U = [1]
        U.extend([a[i] for i in range(1, k + 1)])
        U.append(0)

        V = [0]
        V.extend([a[i] for i in range(k, 0, -1)])
        V.append(1)

        a = np.array(U) + lam * np.array(V)

        # eを更新
        e[k + 1] = e[k] * (1.0 - lam * lam)

    return a, e[-1]