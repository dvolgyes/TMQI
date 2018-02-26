#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.signal import convolve, gaussian
from scipy.ndimage.filters import generic_filter
from scipy.stats import norm, beta
from contracts import contract


@contract(hdrImage='array[NxMx3](float)|array[NxM](float),N>10,M>10',
          ldrImage='array[NxMx3](float)|array[NxM](float)')
def TMQI(hdrImage, ldrImage, window=None):
    """
    Examples:

    >>> test = np.random.normal(size=(100,100),loc=116,scale=28)
    >>> Q, S, N, s_maps, s_local = TMQI(test,test)
    >>> print(S)  # Should be similar to itself
    1.0
    >>> print(N>0.4)
    True

    >>> test2 = np.random.normal(size=(100,100),loc=40,scale=5)
    >>> Q, S, N, s_maps, s_local = TMQI(test2,test2)
    >>> print(S)  # Should be similar to itself
    1.0
    >>> print(N<0.01)
    True

    >>> Q, S, N, s_maps, s_local = TMQI(test,test2)
    >>> print(Q<0.6)
    True
    >>> print(N<0.01)
    True
    >>> print(S<0.3)
    True
    """

    # images must have same dimenions
    assert hdrImage.shape == ldrImage.shape

    if len(hdrImage.shape) == 3 and len(ldrImage.shape) == 3:
        # Processing RGB images
        L_hdr = RGBtoY(hdrImage)
        L_ldr = RGBtoY(ldrImage)
        return TMQI_gray(L_hdr, L_ldr, window)

    # input is already grayscale
    return TMQI_gray(hdrImage, ldrImage, window)


@contract(hdrImage='array[NxM](float)',
          ldrImage='array[NxM](float)',
          window='None|array[UxV],U<N,V<M,U>=2,V>=2')
def TMQI_gray(hdrImage, ldrImage, window=None):
    a = 0.8012
    Alpha = 0.3046
    Beta = 0.7088
    lvl = 5  # levels
    weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    M, N = hdrImage.shape

    if window is None:
        gauss = gaussian(11, 1.5)
        window = np.outer(gauss, gauss)

    # unnecessary, it is just for the sake of parallels with the matlab code
    L_hdr = hdrImage
    L_ldr = ldrImage

    # I think this is a bug:
    # lmin, lmax = L_hdr.min(), L_hdr.max()
    # L_hdr = (L_hdr - lmin) * np.around((2**32 - 1.) / (lmax - lmin))

    S, s_local, s_maps = StructuralFidelity(L_hdr, L_ldr, lvl, weight, window)
    N = StatisticalNaturalness(L_ldr)
    Q = a * (S ** Alpha) + (1 - a) * (N ** Beta)
    return Q, S, N, s_maps, s_local


@contract(RGB='array[NxMx3](float)')
def RGBtoY(RGB):
    M = np.asarray([[0.2126, 0.7152, 0.0722], ])
    Y = np.dot(RGB.reshape(-1, 3), M.T)
    return Y.reshape(RGB.shape[0:2])


@contract(L_hdr='array[NxM](float),N>0,M>0',
          L_ldr='array[NxM](float),N>0,M>0')
def StructuralFidelity(L_hdr, L_ldr, level, weight, window):

    f = 32
    s_local = []
    s_maps = []
    kernel = np.ones((2, 2)) / 4.0

    for _ in range(level):
        f = f / 2
        sl, sm = Slocal(L_hdr, L_ldr, window, f)

        s_local.append(sl)
        s_maps.append(sm)

        # averaging
        filtered_im1 = convolve(L_hdr, kernel, mode='valid')
        filtered_im2 = convolve(L_ldr, kernel, mode='valid')

        # downsampling
        L_hdr = filtered_im1[::2, ::2]
        L_ldr = filtered_im2[::2, ::2]

    S = np.prod(np.power(s_local, weight))
    return S, s_local, s_maps


@contract(img1='array[NxM](float),N>0,M>0',
          img2='array[NxM](float),N>0,M>0',
          sf='float,>0')  # img2=L_ldr
def Slocal(img1, img2, window, sf, C1=0.01, C2=10.):

    window = window / window.sum()

    mu1 = convolve(window, img1, 'valid')
    mu2 = convolve(window, img2, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = convolve(img2 * img2, window, 'valid') - mu2_sq

    sigma1 = np.sqrt(np.maximum(sigma1_sq, 0))
    sigma2 = np.sqrt(np.maximum(sigma2_sq, 0))

    sigma12 = convolve(img1 * img2, window, 'valid') - mu1_mu2

    CSF = 100.0 * 2.6 * (0.0192 + 0.114 * sf) * np.exp(- (0.114 * sf) ** 1.1)
    u_hdr = 128 / (1.4 * CSF)
    sig_hdr = u_hdr / 3.

    sigma1p = norm.cdf(sigma1, loc=u_hdr, scale=sig_hdr)

    u_ldr = u_hdr
    sig_ldr = u_ldr / 3.

    sigma2p = norm.cdf(sigma2, loc=u_ldr, scale=sig_ldr)

    s_map = ((2 * sigma1p * sigma2p + C1) / (sigma1p**2 + sigma2p**2 + C1)
             * ((sigma12 + C2) / (sigma1 * sigma2 + C2)))
    s = np.mean(s_map)
    return s, s_map


@contract(L_ldr='array[NxM](float),N>0,M>0', win='int,>0')
def StatisticalNaturalness(L_ldr, win=11):
    phat1 = 4.4
    phat2 = 10.1
    muhat = 115.94
    sigmahat = 27.99
    u = np.mean(L_ldr)

    # moving window standard deviation using reflected image
    sig = np.mean(generic_filter(L_ldr, np.std, size=win))
    beta_mode = (phat1 - 1.) / (phat1 + phat2 - 2.)
    C_0 = beta.pdf(beta_mode, phat1, phat2)
    C = beta.pdf(sig / 64.29, phat1, phat2)
    pc = C / C_0
    B = norm.pdf(u, muhat, sigmahat)
    B_0 = norm.pdf(muhat, muhat, sigmahat)
    pb = B / B_0
    N = pb * pc
    return N


if __name__ == "__main__":
    import sys
    import doctest
    doctest.testmod()

    if len(sys.argv) > 1:  # there are command line parameters
        from optparse import OptionParser
        from scipy.misc import imread, imsave

        usage = "usage: %prog [options] HDR_image LDR_image"
        parser = OptionParser(usage=usage)

        parser.add_option("-t", "--type",
                          type="string",
                          dest="maptype",
                          help="s_map file type (default: float32)",
                          default="float32")

        parser.add_option("-m", "--smap_file",
                          type="string",
                          dest="smap",
                          help="s_map file name prefix. (default: s_map_)",
                          default="s_map_")

        parser.add_option("-p", "--precision",
                          type="int",
                          dest="precision",
                          help="precision (number of decimals) (default: 4)",
                          default=4)

        parser.add_option("-W", "--width",
                          type="int",
                          dest="width",
                          help="image width (mandatory for RAW files)"
                          " (default: None)",
                          default=None)

        parser.add_option("-H", "--height",
                          type="int",
                          dest="height",
                          help="image height (mandatory for RAW files)"
                          " (default: None)",
                          default=None)

        parser.add_option("-i", "--input_type",
                          type="string",
                          dest="input",
                          help="type of the input images: float32/float64"
                          " for RAW images\n"
                          "None for regular images opening with scipy"
                          " (e.g. png) (default: None)",
                          default=None)

        parser.add_option("-g", "--gray",
                          dest="gray",
                          action="store_true",
                          help="gray input (ligthness/brightness)"
                          "  (default: RGB)",
                          default=False)

        (options, args) = parser.parse_args()

        if len(args) != 2:
            print("Exactly two input files are needed: HDR and LDR.")
            sys.exit(0)

        if options.input is None:
            if options.gray:
                hdr = imread(args[0], "L").astype(np.float)
                ldr = imread(args[1], "L").astype(np.float)
            else:
                hdr = imread(args[0]).astype(np.float)
                ldr = imread(args[1]).astype(np.float)
        else:
            dtype = np.dtype(options.input)
            W, H = options.width, options.height
            hdr = np.fromfile(args[0], dtype=dtype)
            ldr = np.fromfile(args[1], dtype=dtype)
            if options.gray:
                hdr = hdr.reshape(H, W).astype(np.float)
                ldr = ldr.reshape(H, W).astype(np.float)
            else:
                hdr = hdr.reshape(H, W, -1).astype(np.float)
                ldr = ldr.reshape(H, W, -1).astype(np.float)

        Q, S, N, s_maps, s_local = TMQI(hdr, ldr)

        prec = options.precision
        Q, S, N = np.round(Q, prec), np.round(S, prec), np.round(N, prec)
        s_local_str = " ".join(map(str, np.round(s_local, prec)))

        print("Q: %s     S: %s     N: %s" % (Q, S, N))
        print("S_locals: %s" % s_local_str)
        for idx, sm in enumerate(s_maps):
            filename = "%s%i.%s" % (options.smap, idx + 1, options.maptype)

            try:
                out = sm.astype(options.maptype)
                out.tofile(filename)
            except TypeError:
                imsave(filename, sm)
