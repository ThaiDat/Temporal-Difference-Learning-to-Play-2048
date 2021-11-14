from os import path
import numpy as np
from agent import WeightlessNetworkModel
from pickle import dump

FROM_FILE = path.join('bin', 'model.pasrl')
TO_FILE = path.join('bin', 'modelpas.rl')

if __name__=='__main__':
    source = open(FROM_FILE, 'rb')
    model = WeightlessNetworkModel()
    for pattern in model.patterns:
        pattern.table = np.fromfile(source, dtype=np.float32, count=1 << (4 * 6)).reshape([16]*len(pattern.pattern))
    source.close()

    dest = open(TO_FILE, 'wb')
    dump(model, dest)
    dest.close()