import cPickle
import numpy as np
from numpy import zeros, newaxis

def loadRadio():

    # radioml = cPickle.load(open("../rml_data/RML2016.10a_dict.dat",'rb'))
    radioml = cPickle.load(open("../rml_data/2016.04C.multisnr.pkl",'rb'))

    data = {}
    allm = []

    for k in radioml.keys():
        data[k[0]] = {}
        allm.append(k[0])

    mod = sorted(set(allm))

    for m in mod:
        for k in radioml.keys():
            if k[0] == m :
                for sig in range(len(radioml[k])):
                    data[k[0]].setdefault(k[1], [])
                    data[k[0]][k[1]].append(np.expand_dims(radioml[k][sig].T, 2))
    X = []
    Y = []
    x = {}
    y = {} 

    mval = {}
    count = 0

    for m in mod:
        z = np.zeros((len(mod),))
        z[count] = 1     
        mval[m] = z
        for snr in data[m]:
            dat = data[m][snr]
            for d in dat[:int(len(dat)//1.5)]:
                X.append(d)
                Y.append(z)
            for d in dat[int(len(dat)//1.5):]:
                if not snr in x:
                    x[snr] = []
                    y[snr] = []
                x[snr].append(d)
                y[snr].append(z)
        count += 1   

    return X,Y,x,y,mod

if __name__ == '__main__':
    X, Y, x, y, mod = loadRadio()
    print len(X) + len(x)
    print mod
