import numpy as np
import subprocess
from matplotlib import pyplot as plt
from matplotlib import ticker

from pixel2d import Pixel2D
from pixel3d import Pixel3D
from topologicalunionfind import TopologicalUnionFind

class VoidsFinding:
    
    def __init__(self, data, reverse, diagonal, float_multiplier=1e4):
        self.reverse = reverse
        self.diagonal = diagonal
        self.dims = len(data.shape)
        self.originalData = data
        if data.dtype == np.float_:
            self.multiplier = float_multiplier
        else:
            self.multiplier = 1

    def findVoidsUF(self):
        if self.reverse:
            self.dataUF = self.originalData.max() - self.originalData + 1
        else:
            self.dataUF = self.originalData

        if self.dims == 2:
            self.mat = self._create2DPixelObjects(self.dataUF)
            arr = sorted([e for row in self.mat for e in row])
        elif self.dims == 3:
            self.mat = self._create3DPixelObjects(self.dataUF)
            arr = sorted([e for aslice in self.mat for row in aslice for e in row])

        tuf = TopologicalUnionFind()
        for pixel in arr:
            tuf.add(pixel, pixel.getV())
            if self.dims == 2:
                neighbors = self._get2DNeighbors(pixel, self.diagonal)
            elif self.dims == 3:
                neighbors = self._get3DNeighbors(pixel, self.diagonal)
            for neighbor in neighbors:
                tuf.union(pixel, neighbor, pixel.getV())

        if not self.reverse:
            self.history = tuf.persistence_history()
        else:
            self.history = {}
            for s, d in tuf.persistence_history().items():
                birth = d['birth']
                death = d['death']
                history = d['history']
                newhistory = {}
                for k, v in history.items():
                    newhistory[self.originalData.max() - k + 1] = v
                self.history[s] = ({'birth': self.originalData.max() - birth + 1, 
                                    'death': self.originalData.max() - death + 1, 
                                    'history': newhistory})
        self.persistenceUF = []
        for birth, death in tuf.persistence():
            if self.reverse:
                nbirth = self.originalData.max() - death + 1 if np.isfinite(death) else self.originalData.min() 
                ndeath = self.originalData.max() - birth + 1
            else:
                nbirth = birth
                ndeath = death if np.isfinite(death) else self.originalData.max()
            self.persistenceUF.append((nbirth, ndeath))

    def findVoidsPH(self, perseusPath, name):
        if self.reverse:
            self.dataPH = self.originalData.max() - self.originalData + 1
        else:
            self.dataPH = self.originalData - self.originalData.min() + 1
        self.dataPH = (self.dataPH * self.multiplier).astype(int)

        if self.dims == 2:
            self.write2DSimplicialInputFile(name)
        elif self.dims == 3:
            self.write3DSimplicialInputFile(name)
        print(subprocess.run([perseusPath, 'nmfsimtop', name+'.txt', name]))
        self.readPerseusOutputFile(name)

    def writeCubicalInputFile(self, name):
        with open(name+'.txt', 'w') as f:
            f.write(str(len(self.dataPH.shape))+'\n')
            for l in self.dataPH.shape:
                f.write(str(l)+'\n')
            for i in self.dataPH.flatten():
                f.write(str(i)+'\n')

    def write2DSimplicialInputFile(self, name):
        with open(name+'.txt', 'w') as f:
            f.write(str(len(self.dataPH.shape))+'\n')
            for i in range(0, self.dataPH.shape[0]):
                for j in range(0, self.dataPH.shape[1]):
                    f.write('0 {} {} {}\n'.format(i, j, self.dataPH[i, j]))
                    self.write2DPath(f, i, j, i+1, j)
                    self.write2DPath(f, i, j, i, j+1)
                    if self.diagonal:
                        self.write2DPath(f, i, j, i+1, j+1)
                        self.write2DPath(f, i, j, i+1, j-1)

    def write2DPath(self, f, i, j, ni, nj):
        if ni < 0 or nj < 0:
            return
        try:
            f.write('1 {} {} {} {} {}\n'.format(i, j, ni, nj, max(self.dataPH[i, j], self.dataPH[ni, nj])))
        except:
            pass

    def write3DSimplicialInputFile(self, name):
        with open(name+'.txt', 'w') as f:
            f.write(str(len(self.dataPH.shape))+'\n')
            for i in range(0, self.dataPH.shape[0]):
                for j in range(0, self.dataPH.shape[1]):
                    for k in range(0, self.dataPH.shape[2]):
                        f.write('0 {} {} {} {}\n'.format(i, j, k, self.dataPH[i, j, k]))
                        self.write3DPath(f, i, j, k, i+1, j, k)
                        self.write3DPath(f, i, j, k, i, j+1, k)
                        self.write3DPath(f, i, j, k, i, j, k+1)
                        if self.diagonal:
                            self.write3DPath(f, i, j, k, i, j+1, k+1)
                            self.write3DPath(f, i, j, k, i, j+1, k-1)
                            self.write3DPath(f, i, j, k, i+1, j, k+1)
                            self.write3DPath(f, i, j, k, i+1, j, k-1)
                            self.write3DPath(f, i, j, k, i+1, j+1, k)
                            self.write3DPath(f, i, j, k, i+1, j-1, k)
                            self.write3DPath(f, i, j, k, i+1, j+1, k+1)
                            self.write3DPath(f, i, j, k, i+1, j-1, k+1)
                            self.write3DPath(f, i, j, k, i-1, j+1, k+1)
                            self.write3DPath(f, i, j, k, i-1, j-1, k+1)

    def write3DPath(self, f, i, j, k, ni, nj, nk):
        if ni < 0 or nj < 0 or nk < 0:
            return
        try:
            f.write('1 {} {} {} {} {} {} {}\n'.format(i, j, k, ni, nj, nk, max(self.dataPH[i, j, k], self.dataPH[ni, nj, nk])))
        except:
            pass

    def readPerseusOutputFile(self, name):
        pd = []
        with open(name+'_0.txt') as f:
            for l in f.readlines():
                info = l.split()
                start = int(info[0]) / self.multiplier
                end = int(info[1]) / self.multiplier if int(info[1]) != -1 else -1
                pd.append((start, end))
        sorted_pd = sorted(pd, key=lambda x:x[0])
        self.persistencePH = []
        for birth, death in sorted_pd:
            if self.reverse:
                nbirth = self.originalData.max() - death + 1 if death != -1 else self.originalData.min() 
                ndeath = self.originalData.max() - birth + 1
            else:
                nbirth = birth + self.originalData.min() - 1
                ndeath = death + self.originalData.min() - 1 if death != -1 else self.originalData.max()
            self.persistencePH.append((nbirth, ndeath))

    def _create2DPixelObjects(self, data):
        mat = []
        for i in range(data.shape[0]):
            row = []
            for j in range(data.shape[1]):
                element = data[i][j]
                row.append(Pixel2D(i, j, element))
            mat.append(row)
        return mat

    def _create3DPixelObjects(self, data):
        mat = []
        for i in range(data.shape[0]):
            aslice = []
            for j in range(data.shape[1]):
                row = []
                for k in range(data.shape[2]):
                    element = data[i][j][k]
                    row.append(Pixel3D(i, j, k, element))
                aslice.append(row)
            mat.append(aslice)
        return mat

    def _get2DNeighbors(self, pixel, diagonal=False):
        neighbors = []
        x = pixel.getX()
        y = pixel.getY()
        self._append2DNeighbor(neighbors, x-1, y)
        self._append2DNeighbor(neighbors, x+1, y)
        self._append2DNeighbor(neighbors, x, y-1)
        self._append2DNeighbor(neighbors, x, y+1)
        if diagonal:
            self._append2DNeighbor(neighbors, x-1, y-1)
            self._append2DNeighbor(neighbors, x-1, y+1)
            self._append2DNeighbor(neighbors, x+1, y-1)
            self._append2DNeighbor(neighbors, x+1, y+1)
        return neighbors

    def _get3DNeighbors(self, pixel, diagonal=False):
        neighbors = []
        x = pixel.getX()
        y = pixel.getY()
        z = pixel.getZ()
        self._append3DNeighbor(neighbors, x-1, y, z)
        self._append3DNeighbor(neighbors, x+1, y, z)
        self._append3DNeighbor(neighbors, x, y-1, z)
        self._append3DNeighbor(neighbors, x, y+1, z)
        self._append3DNeighbor(neighbors, x, y, z-1)
        self._append3DNeighbor(neighbors, x, y, z+1)
        if diagonal:
            self._append3DNeighbor(neighbors, x-1, y-1, z-1)
            self._append3DNeighbor(neighbors, x-1, y-1, z)
            self._append3DNeighbor(neighbors, x-1, y-1, z+1)
            self._append3DNeighbor(neighbors, x-1, y, z-1)
            self._append3DNeighbor(neighbors, x-1, y, z+1)
            self._append3DNeighbor(neighbors, x-1, y+1, z-1)
            self._append3DNeighbor(neighbors, x-1, y+1, z)
            self._append3DNeighbor(neighbors, x-1, y+1, z+1)
            self._append3DNeighbor(neighbors, x, y+1, z+1)
            self._append3DNeighbor(neighbors, x, y-1, z+1)
            self._append3DNeighbor(neighbors, x, y+1, z-1)
            self._append3DNeighbor(neighbors, x, y-1, z-1)
            self._append3DNeighbor(neighbors, x+1, y-1, z-1)
            self._append3DNeighbor(neighbors, x+1, y-1, z)
            self._append3DNeighbor(neighbors, x+1, y-1, z+1)
            self._append3DNeighbor(neighbors, x+1, y, z-1)
            self._append3DNeighbor(neighbors, x+1, y, z+1)
            self._append3DNeighbor(neighbors, x+1, y+1, z-1)
            self._append3DNeighbor(neighbors, x+1, y+1, z)
            self._append3DNeighbor(neighbors, x+1, y+1, z+1)
        return neighbors

    def _append2DNeighbor(self, neighbors, x, y):
        if x < 0 or y < 0:
            return
        try:
            neighbors.append(self.mat[x][y])
        except:
            pass

    def _append3DNeighbor(self, neighbors, x, y, z):
        if x < 0 or y < 0 or z < 0:
            return
        try:
            neighbors.append(self.mat[x][y][z])
        except:
            pass

    def getPersistenceUF(self):
        """
        Need finer result
        """
        return self.persistenceUF

    def getPersistencePH(self):
        """
        Need finer result
        """
        return self.persistencePH

    def getHistory(self):
        return self.history

    def plotPersistenceBarcodesUF(self):
        y = np.arange(0, len(self.persistenceUF))
        xmin = [birth for birth, death in self.persistenceUF]
        xmax = [death for birth, death in self.persistenceUF]
        if self.reverse:
            xlim = (max(xmax), min(xmin))
        else:
            xlim = (min(xmin), max(xmax))

        height = 2 + len(self.persistenceUF) // 10
        height = 10 if height > 10 else height
        plt.figure(figsize=(15,height))
        plt.hlines(y=y, xmin=xmin, xmax=xmax)
        # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator())
        if len(self.persistenceUF) < 10:
            plt.gca().yaxis.set_major_locator(ticker.MultipleLocator())
        plt.xlim(xlim[0], xlim[1])
        plt.xlabel('Value')
        plt.ylabel('Connected Component ID')
        plt.show()

    def plotPersistenceDiagramUF(self):
        xmin = [birth for birth, death in self.persistenceUF]
        xmax = [death for birth, death in self.persistenceUF]
        if self.reverse:
            xlim = (max(xmax)+1, min(xmin)-1)
            xmin, xmax = xmax, xmin
        else:
            xlim = (min(xmin)-1, max(xmax)+1)
        plt.figure(figsize=(10,7))
        plt.scatter(xmin, xmax, c=np.array(xmax)-np.array(xmin), cmap='jet_r')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(xlim[0], xlim[1])
        # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator())
        # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator())
        plt.xlabel('Birth Time')
        plt.ylabel('Death Time')
        plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], c='grey')
        plt.show()
        
    def plotPersistenceBarcodesPH(self):
        y = np.arange(0, len(self.persistencePH))
        xmin = [birth for birth, death in self.persistencePH]
        xmax = [death for birth, death in self.persistencePH]
        if self.reverse:
            xlim = (max(xmax), min(xmin))
        else:
            xlim = (min(xmin), max(xmax))
        height = 2 + len(self.persistencePH) // 10
        plt.figure(figsize=(15,height))
        plt.hlines(y=y, xmin=xmin, xmax=xmax)
        plt.xlim(xlim[0], xlim[1])
        # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator())
        if len(self.persistencePH) < 10:
            plt.gca().yaxis.set_major_locator(ticker.MultipleLocator())
        plt.xlabel('Value')
        plt.ylabel('Connected Component ID')
        plt.show()

    def plotPersistenceDiagramPH(self):
        xmin = [birth for birth, death in self.persistencePH]
        xmax = [death for birth, death in self.persistencePH]
        if self.reverse:
            xlim = (max(xmax)+1, min(xmin)-1)
            xmin, xmax = xmax, xmin
        else:
            xlim = (min(xmin)-1, max(xmax)+1)

        plt.figure(figsize=(10,7))
        plt.scatter(xmin, xmax, c=np.array(xmax)-np.array(xmin), cmap='jet_r')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(xlim[0], xlim[1])
        # plt.gca().xaxis.set_major_locator(ticker.MultipleLocator())
        # plt.gca().yaxis.set_major_locator(ticker.MultipleLocator())
        plt.xlabel('Birth Time')
        plt.ylabel('Death Time')
        plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], c='grey')
        plt.show()
