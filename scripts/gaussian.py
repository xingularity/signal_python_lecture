## This program is to demonstrate gaussian function and its fourier transform
## Author: Zong-han, Xie<icbm0926@gmail.com>
## License: CC-4.0, by-nc-sa

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import sys
import scipy.fftpack as scipyfft

delta_org = 1 #sampling interval
sf_org = 1./delta_org #sampling frequency
signal_length = 400

class Gaussian(object):
    def __init__(self):
        x = np.arange(0,signal_length,delta_org)
        sigma1 = 50
        sigma2 = 10
        k_0 = 2.0*np.pi/10
        y1, fbins1, dft1 = self.gaussian_carrier(x, x.mean(), sigma1, k_0)
        y2, fbins2, dft2 = self.gaussian_carrier(x, x.mean(), sigma2, k_0)
        plt.xlabel(r'x')
        plt.ylabel(r'gaussian with carrier')
        plt.plot(x, y1, 'b')
        plt.plot(x, y2, 'g')
        plt.figure(2)
        plt.xlabel(r'frequency')
        plt.ylabel(r'normalized spectrum')
        #plot normalized amplitude to show the difference between frequency width
        plt.plot(fbins1, (np.abs(dft1)/np.max(np.abs(dft1)))**2/len(dft1)**2, 'b')
        plt.plot(fbins2, (np.abs(dft2)/np.max(np.abs(dft2)))**2/len(dft2)**2, 'g')
        plt.show()
        
    def gaussian(self,x, xcenter, sigma):
        y = np.zeros(x.shape)
        for i in range(len(x)):
            y[i] = np.exp(-(x[i]-xcenter)**2/2.0/sigma**2)
        dft = scipyfft.fftshift(scipyfft.fft(y))
        fc = 1.0/2.0/delta_org
        df = 1.0/int(signal_length/delta_org)/delta_org
        if ((signal_length % 2) == 0):
            fbins = -1.0*fc+np.array(range(signal_length))*df
        else:
            fbins = -1.0*fc+0.5*df+np.array(range(signal_length))*df
        return (y, fbins, dft)

    def gaussian_carrier(self,x, xcenter, sigma, k_0):
        y = np.zeros(x.shape)
        for i in range(len(x)):
            y[i] = np.exp(-(x[i]-xcenter)**2/2.0/sigma**2)*np.exp(-1j*k_0*i)
        dft = scipyfft.fftshift(scipyfft.fft(y))
        fc = 1.0/2.0/delta_org
        df = 1.0/int(signal_length/delta_org)/delta_org
        if ((signal_length % 2) == 0):
            fbins = -1.0*fc+np.array(range(signal_length))*df
        else:
            fbins = -1.0*fc+0.5*df+np.array(range(signal_length))*df
        return (y, fbins, dft)

def main():
    Gaussian()

if __name__ == '__main__':
	main()
