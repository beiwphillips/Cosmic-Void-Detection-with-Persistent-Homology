import numpy as np
import json
from voidsfinding import VoidsFinding

if __name__ == '__main__':

    prefixes = ['v6.0.{}_shuffle_'.format(i) for i in range(10)]
    
    for prefix in prefixes:
        print('Processing {}'.format(prefix))
        binary = np.fromfile(prefix+'map.bin')
        with open(prefix+'void.cfg') as cfg:
            lines = cfg.readlines()
            nx = int(lines[4].split()[-1])
            ny = int(lines[5].split()[-1])
            nz = int(lines[6].split()[-1])
        cube = binary.reshape(nx, ny, nz)
        vf = VoidsFinding(cube, reverse=True, diagonal=False)
        vf.findVoidsUF()
        history = vf.getHistory()

        with open(prefix+'.json', 'w') as outfile:  
            json.dump(history, outfile)