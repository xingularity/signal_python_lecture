import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import sys
import scipy.fftpack as scipyfft

delta_org = 1 #sampling interval
sf_org = 1./delta_org #sampling frequency
signal_interval = 10
signal_f = 1./signal_interval
sl = 10.0*signal_interval

def addParserOption():
    parser = OptionParser()
    parser.add_option("-x", "", dest="exNum", help="要被畫的函數")
    options = parser.parse_args()[0]
    return options

class OriginalSignal(object):
    def __init__(self, downInterval=None):
        if (downInterval == None):
            x, signal, dft = self.signalAndDFT()
            plt.plot(x, signal, label="Original")
            plt.show()

    def signalAndDFT(self):
        x = np.arange(0, int(sl/delta_org), delta_org)
        signal = np.zeros(x.shape)
        for i in range(len(signal)):
            signal[i] = np.sin(2.0*np.pi*signal_f*i*delta_org)
        dft = scipyfft.fftshift(scipyfft.fft(signal))
        return (x, signal, dft)

    def downSample(self, downInterval=None):
        """
        Param:
            downInterval: sampling interval of down sample
        """
        pass

class PlotObject(object):
    def __init__(self):
	    self.plotlist = {}
	    self.plotlist = {0:OriginalSignal}
	    self.plothelp = {0:"OriginalSignal"}
    def getPlotObject(self, num):
        return self.plotlist[num]

    def showHelpList(self):
    	for i in self.plothelp:
            print(str(i) + ": " + str(self.plothelp[i]))

def main(options):
    generator = PlotObject()
    if (options.exNum == None):
        generator.showHelpList()
        sys.exit(0)
    generator.getPlotObject(int(options.exNum))()

if __name__ == '__main__':
	main(addParserOption())
