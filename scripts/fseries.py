import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import sys

def addParserOption():
    parser = OptionParser()
    parser.add_option("-x", "", dest="exNum", help="要被畫的函數")
    options = parser.parse_args()[0]
    return options

class StepFunction(object):
    def __init__(self):
        x = np.arange(-1.0*np.pi, 1.0*np.pi, 0.002*np.pi)
        y = np.zeros(x.shape)
        for i in range(len(x)):
            if(x[i] < 0):
                y[i] = 0
            else:
             y[i] = 1
        f3 = self.fseries(x, 3)
        f5 = self.fseries(x, 5)
        f10 = self.fseries(x, 10)
        f20 = self.fseries(x, 20)
        f100 = self.fseries(x, 100)
        plt.plot(x, y, label="Step Function")
        plt.plot(x, f3, label="Fourier series to n = 3")
        plt.plot(x, f5, label="Fourier series to n = 5")
        plt.plot(x, f10, label="Fourier series to n = 10")
        plt.plot(x, f20, label="Fourier series to n = 20")
        plt.plot(x, f100, label="Fourier series to n = 100")
        plt.legend(loc='upper left')
        plt.show()

    def fseries(self, x, num):
        """
        param:
              x: x coordinate
              num: number of terms in series
        """
        fseries = np.zeros(x.shape)
        for i in range(len(x)):
            j = 1
            fseries[i] = 0.5
            while(j<=num):
                fseries[i] = fseries[i] + (2.0/np.pi)*np.sin(j*x[i])/j
                j = j + 2
        return fseries

class TriangularFunction(object):
    def __init__(self):
        x = np.arange(-1.0*np.pi, 1.0*np.pi, 0.002*np.pi)
        y = np.zeros(x.shape)
        for i in range(len(x)):
            if(x[i] < 0):
                y[i] = -x[i]
            else:
                y[i] = x[i]
        f3 = self.fseries(x, 3)
        f5 = self.fseries(x, 5)
        f10 = self.fseries(x, 10)
        f20 = self.fseries(x, 20)
        plt.plot(x, y, label="Triangular function")
        plt.plot(x, f3, label="Fourier series to n = 3")
        plt.plot(x, f5, label="Fourier series to n = 5")
        plt.plot(x, f10, label="Fourier series to n = 10")
        plt.plot(x, f20, label="Fourier series to n = 20")
        plt.legend(loc='upper left')
        plt.show()

    def fseries(self, x, num):
        """
        param:
              x: x coordinate
              num: number of terms in series
        """
        fseries = np.zeros(x.shape)
        for i in range(len(x)):
            j = 1
            fseries[i] = 0.5*np.pi
            while(j<=num):
                fseries[i] = fseries[i] - (4.0/np.pi)*np.cos(j*x[i])/float(j)**2
                j = j + 2
        return fseries

class FullRectifierFunction(object):
    def __init__(self):
        x = np.arange(-1.0*np.pi, 1.0*np.pi, 0.002*np.pi)
        y = np.zeros(x.shape)
        for i in range(len(x)):
            if(x[i] < 0):
                y[i] = -np.sin(x[i])
            else:
                y[i] = np.sin(x[i])
        f3 = self.fseries(x, 3)
        f5 = self.fseries(x, 5)
        f10 = self.fseries(x, 10)
        f20 = self.fseries(x, 20)
        plt.plot(x, y, label="Full rectifier")
        plt.plot(x, f3, label="Fourier series to n = 3")
        plt.plot(x, f5, label="Fourier series to n = 5")
        plt.plot(x, f10, label="Fourier series to n = 10")
        plt.plot(x, f20, label="Fourier series to n = 20")
        plt.legend(loc='upper left')
        plt.show()

    def fseries(self, x, num):
        """
        param:
              x: x coordinate
              num: number of terms in series
        """
        fseries = np.zeros(x.shape)
        for i in range(len(x)):
            j = 2
            fseries[i] = 2.0/np.pi
            while(j<=num):
                fseries[i] = fseries[i] - (4.0/np.pi)*np.cos(j*x[i])/(float(j)**2-1)
                j = j + 2
        return fseries

class PlotObject(object):
    def __init__(self):
	    self.plotlist = {}
	    self.plotlist = {0:StepFunction, 1:TriangularFunction, 2:FullRectifierFunction}
	    self.plothelp = {0:"StepFuntion", \
                         1:"TriangularFunction", \
        				 2:"FullRectifierFunction"}
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
