"""
Algorithm based on the paper 'Automatic Chord Recognition from
Audio Using Enhanced Pitch Class Profile' by Kyogu Lee
This script computes 12 dimensional chromagram for chord detection
@author ORCHISAMA
"""

from __future__ import division
from scipy.signal import hamming
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt


def nearestPow2(inp):
    power = np.ceil(np.log2(inp))
    return 2**power


"""Function to calculcate Harmonic Power Spectrum from DFT"""


def HPS(dft, M):

    hps_len = int(np.ceil(np.size(dft) / (2**M)))
    hps = np.ones(hps_len)
    for n in range(hps_len):
        for m in range(M + 1):
            hps[n] *= np.absolute(dft[(2**m) * n])
    return hps


"""Function to compute CQT using sparse matrix multiplication, Brown and Puckette 1992- fast"""


def CQT_fast(x, fs, bins, fmin, fmax, M):

    threshold = 0.0054  # for Hamming window
    K = int(bins * np.ceil(np.log2(fmax / fmin)))
    Q = 1 / (2 ** (1 / bins) - 1)
    nfft = np.int32(nearestPow2(np.ceil(Q * fs / fmin)))
    tempKernel = np.zeros(nfft, dtype=np.complex)
    specKernel = np.zeros(nfft, dtype=np.complex)
    sparKernel = []

    # create sparse Kernel
    for k in range(K - 1, -1, -1):
        fk = (2 ** (k / bins)) * fmin
        N = np.int32(np.round((Q * fs) / fk))
        tempKernel[:N] = hamming(N) / N * np.exp(-2 * np.pi * 1j * Q * np.arange(N) / N)
        specKernel = fft(tempKernel)
        specKernel[np.where(np.abs(specKernel) <= threshold)] = 0
        if k == K - 1:
            sparKernel = specKernel
        else:
            sparKernel = np.vstack((specKernel, sparKernel))

    sparKernel = np.transpose(np.conjugate(sparKernel)) / nfft
    ft = fft(x, nfft)
    cqt = np.dot(ft, sparKernel)
    ft = fft(x, nfft * (2**M))
    # calculate harmonic power spectrum
    # harm_pow = HPS(ft,M)
    # cqt = np.dot(harm_pow, sparKernel)
    return cqt


"""Function to compute constant Q Transform, Judith Brown, 1991 - slow"""


def CQT_slow(x, fs, bins, fmin, fmax):

    K = int(bins * np.ceil(np.log2(fmax / fmin)))
    Q = 1 / (2 ** (1 / bins) - 1)
    cqt = np.zeros(K, dtype=np.complex)

    for k in range(K):
        fk = (2 ** (k / bins)) * fmin
        N = int(np.round(Q * fs / fk))
        arr = -2 * np.pi * 1j * Q * np.arange(N) / N
        cqt[k] = np.dot(x[:N], np.transpose(hamming(N) * np.exp(arr))) / N
    return cqt


"""Function to compute Pitch Class Profile from constant Q transform"""


def PCP(cqt, bins, M):
    CH = np.zeros(bins)
    for b in range(bins):
        CH[b] = np.sum(cqt[b + (np.arange(M) * bins)])
    return CH


def compute_chroma(x, fs):

    fmin = 96
    fmax = 5250
    bins = 12
    M = 3
    nOctave = np.int32(np.ceil(np.log2(fmax / fmin)))
    CH = np.zeros(bins)
    # Compute constant Q transform
    cqt_fast = CQT_fast(x, fs, bins, fmin, fmax, M)
    # get Pitch Class Profile
    CH = PCP(np.absolute(cqt_fast), bins, nOctave)
    return CH
