## This program is to plot aliasing demonstration
## Author: Zong-han, Xie<icbm0926@gmail.com>
## License: CC-4.0, by-nc-SA

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import sys
import scipy.fftpack as scipyfft

delta_org = 1 #sampling interval
sf_org = 1./delta_org #sampling frequency
signal_interval = 20
signal_f = 1./signal_interval
sl = 20.0*signal_interval

def addParserOption():
    parser = OptionParser()
    parser.add_option("-d", "--downsample", dest="downInterval", help="down sample interval")
    options = parser.parse_args()[0]
    return options

class SignalSampling(object):
    def __init__(self, downInterval=None):
        x, signal, fbins, dft = self.signalAndDFT()
        downx= None
        downsignal = None
        downfbins = None
        downdft = None
        if (downInterval != None):
            downx, downsignal, downfbins, downdft = self.downSample(x, signal, 5, downInterval)
        plt.plot(x, signal, 'b', label="Original")
        if (downInterval != None):
            plt.plot(downx, downsignal, 'ro--', label="down sampled")
        plt.xlabel(r'x')
        plt.ylabel(r'signal')
        plt.figure(2)
        plt.plot(fbins, np.abs(dft)**2/len(dft)**2, label="Original")
        if (downInterval != None):
            plt.plot(downfbins, np.abs(downdft)**2/len(downdft)**2, 'ro--', label="down sampled")
        plt.xlabel(r'frequency')
        plt.ylabel(r'abs(signal_fft)')
        plt.show()

    def signalAndDFT(self):
        x = np.arange(0, int(sl/delta_org), delta_org)
        signal = np.zeros(x.shape)
        for i in range(len(signal)):
            signal[i] = np.sin(2.0*np.pi*signal_f*x[i])
        dft = scipyfft.fftshift(scipyfft.fft(signal))
        #create shifted frquency bins
        x_len = len(x)
        fc = 1.0/2.0/delta_org
        df = 1.0/int(sl/delta_org)/delta_org
        fbins = np.arange(-1.0*fc+df, fc+df, df)
        if ((x_len % 2) == 0):
            fbins = np.arange(-1.0*fc+df, fc+0.5*df, df)
        else:
            fbins = np.arange(-1.0*fc+0.5*df, fc+0.5*df, df)
        print(fbins.shape)
        print(dft.shape)
        return (x, signal, fbins, dft)

    def downSample(self, x, signal, xstart_index=0, downInterval=None):
        """
        Param:
            downInterval: sampling interval of down sample
        """
        x = np.copy(x)
        signal = np.copy(signal)
        x=x[xstart_index:]
        signal=signal[xstart_index:]
        if (downInterval != None):
            x = x[0:len(x):int(downInterval)]
            signal = signal[0:len(signal):int(downInterval)]

        x_len = len(x)
        fbins=None
        dft = None
        dft = scipyfft.fftshift(scipyfft.fft(signal))
        if ((x_len % 2) == 0):
            fc = 1.0/2.0/int(downInterval)
            df = 1.0/x_len/int(downInterval)
            fbins = np.arange(-1.0*fc+df, fc+0.5*df, df)
        else:
            fc = 1.0/2.0/int(downInterval)
            df = 1.0/x_len/int(downInterval)
            fbins = np.arange(-1.0*fc+0.5*df, fc+0.5*df, df)
        print(fbins.shape)
        print(dft.shape)
        return (x, signal, fbins, dft)

class PlotObject(object):
    def __init__(self):
	    self.plotlist = {}
	    self.plotlist = {0:SignalSampling}
	    self.plothelp = {0:"SignalSampling"}
    def getPlotObject(self, num):
        return self.plotlist[num]

    def showHelpList(self):
    	for i in self.plothelp:
            print(str(i) + ": " + str(self.plothelp[i]))

def main(options):
    generator = PlotObject()
    downInterval = None
    if (options.downInterval != None):
        downInterval = int(options.downInterval)
        generator.getPlotObject(0)(downInterval)
    else:
        generator.getPlotObject(0)()

if __name__ == '__main__':
	main(addParserOption())
