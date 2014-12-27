## This program is to demoinstrate filtering
## Author: Zong-han, Xie<icbm0926@gmail.com>
## License: CC-4.0, by-nc-sa

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
df_org = 1.0/int(sl/delta_org)/delta_org

def addParserOption():
    parser = OptionParser()
    parser.add_option("", "--ftickinterval", dest="ftickinterval", help="frequency plot tick interval")
    parser.add_option("-x", "", dest="exNum", help="signals to plot")
    options = parser.parse_args()[0]
    return options

class RectangularFilter(object):
    def __init__(self, downInterval=None, ftickinterval=0.05):
        x, signal, fbins, dft = self.signalAndDFT()
        window, filteredsignal = self.filtering(fbins, dft, signal_f/4,4*df_org)
        plt.plot(x, signal, 'b', label="Original")
        plt.xlabel(r'x')
        plt.ylabel(r'signal')
        plt.figure(2)
        plt.plot(fbins, np.abs(dft)**2/len(dft)**2, label="dft")
        plt.xticks(np.arange(min(fbins), max(fbins)+0.05, ftickinterval))
        plt.xlabel(r'frequency')
        plt.ylabel(r'abs(signal_fft)^2/N^2')
        plt.twinx()
        plt.plot(fbins, window, 'r',label="filter window")
        plt.ylabel(r'filter window')
        plt.figure(3)
        plt.plot(x, np.real(filteredsignal), 'b', label="Original")
        plt.xlabel(r'x')
        plt.ylabel(r'filteredsignal')      
        plt.show()

    def signalAndDFT(self):
        x = np.arange(0, int(sl/delta_org), delta_org)
        signal = np.zeros(x.shape)
        another_f = signal_f/4.0
        for i in range(len(signal)):
            signal[i] = np.sin(2.0*np.pi*signal_f*x[i]) + np.sin(2.0*np.pi*another_f*x[i])
        dft = scipyfft.fftshift(scipyfft.fft(signal))
        #create shifted frquency bins
        x_len = len(x)
        fc = 1.0/2.0/delta_org
        df = 1.0/int(sl/delta_org)/delta_org
        fbins = np.arange(-1.0*fc+df, fc+df, df)
        if ((x_len % 2) == 0):
            fbins = -1.0*fc+np.array(range(x_len))*df
        else:
            fbins = -1.0*fc+0.5*df+np.array(range(x_len))*df
        print(fbins.shape)
        print(dft.shape)
        return (x, signal, fbins, dft)
        
    def filtering(self, fbins, dft, centerf, window_band):
        """
        return filtered signal
        Param:
            fbins: fftshiftted fbins
            dft: fftshiftted signal spectrum
            centerf: freqnecy to be preserved, others will be filtered
            windowband: full bandwidth of the preserved frequency band
        """
        fbins = np.copy(fbins)
        dft = np.copy(dft)
        # create rectangular window filter
        window = np.zeros(fbins.shape)
        for i in range(len(window)):
            if (np.abs(np.abs(fbins[i])-np.abs(centerf)) <= window_band/2.0):
                window[i] = 1.0
            else:
                window[i] = 0.0
        return (window, scipyfft.ifft(scipyfft.ifftshift(dft*window)))

class PlotObject(object):
    def __init__(self):
	    self.plotlist = {}
	    self.plotlist = {0:RectangularFilter}
	    self.plothelp = {0:"RectangularFilter"}
    def getPlotObject(self, num):
        return self.plotlist[num]
	
    def showHelpList(self):
    	for i in self.plothelp:
            print(str(i) + ": " + str(self.plothelp[i]))

def main(options):
    downInterval = None
    plotObj = PlotObject()
    SignalObj=None
    if (options.exNum == None):
        plotObj.showHelpList()
        sys.exit(0)
    else:
        SignalObj = plotObj.getPlotObject(int(options.exNum))

    if (options.ftickinterval != None):
        ftickinterval = float(options.ftickinterval)
        SignalObj(ftickinterval=ftickinterval)
    else:
        SignalObj()

if __name__ == '__main__':
	main(addParserOption())
