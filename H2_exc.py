import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from scipy import integrate


def column(matrix, i):
    if i == 0 or (isinstance(i, str) and i[0] == 'v'):
        return [row.val for row in matrix]
    if i == 1 or (isinstance(i, str) and i[0] == 'p'):
        return [row.plus for row in matrix]
    if i == 2 or (isinstance(i, str) and i[0] == 'm'):
        return [row.minus for row in matrix]

H2_energy = np.genfromtxt('energy_X_H2.dat', dtype=[('nu', 'i2'), ('j', 'i2'), ('e', 'f8')],
                          unpack=True, skip_header=3, comments='#')
H2energy = np.zeros([max(H2_energy['nu']) + 1, max(H2_energy['j']) + 1])
for e in H2_energy:
    H2energy[e[0], e[1]] = e[2]

stat = [(2 * i + 1) * ((i % 2) * 2 + 1) for i in range(12)]


spcode = {'H': 'n_h', 'H2': 'n_h2',
          'H2j0': 'pop_h2_v0_j0', 'H2j1': 'pop_h2_v0_j1', 'H2j2': 'pop_h2_v0_j2', 'H2j3': 'pop_h2_v0_j3',
          'H2j4': 'pop_h2_v0_j4', 'H2j5': 'pop_h2_v0_j5', 'H2j6': 'pop_h2_v0_j6', 'H2j7': 'pop_h2_v0_j7',
          'C': 'n_c', 'C+': 'n_cp', 'CO': 'n_co',
         }

class model():
    def __init__(self, folder='', name=None, filename=None, species=[]):
        self.folder = folder
        self.sp = {}
        self.species = species
        self.filename = filename
        if filename is not None:
            if name is None:
                name = filename.replace('.hdf5', '')

            self.name = name
            self.read()

    def read(self, showMeta=False):
        """
        Read model data from hdf5 file

        :param
            -  showMeta         : if True show Metadata table

        :return: None
        """

        self.file = h5py.File(self.folder + self.filename, 'r')

        # >>> model input parameters
        self.z = self.par('metal')
        self.P = self.par('gas_pressure_input')
        self.uv = self.par('radm_ini')

        # >>> profile of physical quntities
        self.x = self.par('distance')

        self.av = self.par('av')
        self.tgas = self.par('tgas')
        self.Pgas = self.par('Pgas')
        self.n = self.par('ntot')

        self.readspecies()

        if 0:
            self.plot_phys_cond()
            self.plot_profiles()

        if showMeta:
            self.showMetadata()

        self.file.close()

    def par(self, par=None):
        """
        Read parameter or array from the hdf5 file, specified by Metadata name

        :param:
            -  par          :  the name of the parameter to read

        :return: x
            - x             :  corresponding data (string, number, array) correspond to the parameter
        """
        meta = self.file['Metadata/Metadata']
        if par is not None:
            ind = np.where(meta[:, 3] == par.encode())[0]
            if len(ind) > 0:
                attr = meta[ind, 0][0].decode() + '/' + meta[ind, 1][0].decode()
                x = self.file[attr][:, int(meta[ind, 2][0].decode())]
                if len(x) == 1:
                    typ = {'string': str, 'real': float, 'integer': int}[meta[ind, 5][0].decode()]
                    return typ(x[0].decode())
                else:
                    return x

    def showMetadata(self):
        """
        Show Metadata information in the table
        """

        self.w = pg.TableWidget()
        self.w.show()
        self.w.resize(500, 900)
        self.w.setData(self.file['Metadata/Metadata'][:])

    def readspecies(self, species=None):
        """
        Read the profiles of the species

        :param
            -  species       : the list of the names of the species to read from the file

        :return: None

        """
        if species is None:
            species = self.species

        for s in species:
            self.sp[s] = self.par(spcode[s])

        self.species = species

    def plot_phys_cond(self, pars=['tgas', 'n', 'av'], logx=True, ax=None, legend=True):
        """
        Plot the physical parameters in the model

        :parameters:
            -  pars         :  list of the parameters to plot
            -  logx         :  if True x in log10
            -  ax           :  axes object to plot
            -  legend       :  show Legend

        :return: ax
            -  ax           :  axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        if logx:
            mask = self.x > 0
            x = np.log10(self.x[mask])
            xlabel = 'log(Distance), cm'
        else:
            mask = self.x > -1
            x = self.x
            xlabel = 'Distance, cm'
        ax.set_xlim([x[0], x[-1]])
        ax.set_xlabel(xlabel)

        lines = []
        for i, p in enumerate(pars):
            if i == 0:
                axi = ax
            else:
                axi = ax.twinx()

            color = plt.cm.tab10(i / 10)
            axi.set_ylabel(p, color=color)
            line, = axi.plot(x, getattr(self, p)[mask], color=color, label=p)
            lines.append(line)

            if i > 0:
                axi.spines['right'].set_position(('outward', 60*(i-1)))
                axi.set_frame_on(True)
                axi.patch.set_visible(False)

            for t in axi.get_yticklabels():
                t.set_color(color)

        if legend:
            ax.legend(handles=lines, loc='best')


    def plot_profiles(self, species=None, ax=None, legend=True):
        """
        Plot the profiles of the species

        :param:
            -  species       :  ist of the species to plot
            -  ax            :  axis object to plot in, if None, then figure is created
            -  legend        :  show legend

        :return: ax
            -  ax            :  axis object
        """
        if species is None:
            species = self.species

        if ax is None:
            fig, ax = plt.subplots()

        for s in species:
            ax.plot(np.log10(self.x[1:]), np.log10(self.sp[s][1:]), '-', label=s)

        if legend:
            ax.legend()

        return ax

    def calc_cols(self, species=[], logN=None, sides=2):
        """
        Calculate column densities for given species

        :param:
            -  species       :  list of the species to plot
            -  logN          :  column density threshold, dictionary with species and logN value
            -  side          :  make calculation to be one or two sided

        :return: sp
            -  sp            :  dictionary of the column densities by species
        """

        if logN is not None:
            logN[list(logN.keys())[0]] -= np.log10(sides)
            self.set_mask(logN=logN)

        cols = {}
        for s in species:
            cols[s] = np.log10(np.trapz(self.sp[s][self.mask], x=self.x[self.mask]) * sides)

        return cols

    def set_mask(self, logN={'H': None}):
        """
        Calculate mask for a given threshold

        :param:
            -  logN          :  column density threshold

        :return: None
        """
        cols = np.insert(np.log10(integrate.cumtrapz(self.sp[list(logN.keys())[0]], x=self.x)), 0, 0)
        if logN is not None:
            self.mask = cols < logN[list(logN.keys())[0]]
        else:
            self.mask = cols > -1
        #return np.searchsorted(cols, value)

class H2_exc():
    def __init__(self, folder=''):
        self.folder = folder
        self.model = {}
        self.species = ['H', 'H2', 'C', 'C+', 'CO', 'H2j0', 'H2j1', 'H2j2', 'H2j3', 'H2j4', 'H2j5', 'H2j6', 'H2j7']
        self.readH2database()

    def readH2database(self):
        import sys
        sys.path.append('C:/science/python/')
        import H2_summary

        self.H2 = H2_summary.load_QSO()

    def readmodel(self, filename=None):
        """
        Read one model by filename
        """
        if filename is not None:
            m = model(folder=self.folder, filename=filename, species=self.species)
            self.model[m.name] = m
            self.current = m.name

    def readfolder(self):
        """
        Read list of models from the folder
        """
        for f in os.listdir(self.folder):
            self.readmodel(f)

    def compare(self, objects=[], model='current'):
        if model == 'current':
            m = self.model[self.current]
        else:
            m = self.model[model]

        obj = {}
        for o in objects:
            for qso in self.H2:
                if o in qso.name:
                    for q in qso.comp:
                        species = [s for s in q.e.keys() if 'H2j' in s]
                        sp = m.calc_cols(species, logN={'H2': q.e['H2'].col.val})
                        print(sp)
                        for s in sp.keys():
                            sp[s] = (sp[s], q.e[s].col)
            obj[o] = sp

        return obj

    def plot_H2exc(self, objects=[], ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        objects = self.compare(objects=objects)

        for sp in objects.values():
            j = np.sort([int(s[3:]) for s in sp.keys()])
            x = [H2energy[0, i] for i in j]
            y = [sp['H2j'+str(i)][1] / stat[i] for i in j]
            mod = [sp['H2j'+str(i)][0] - np.log10(stat[i]) for i in j]
            typ = [sp['H2j'+str(i)][1].type for i in j]
            if len(y) > 0:
                color = plt.cm.tab10(list(objects.values()).index(sp)/10)

                ax.plot(x, mod, marker='o', ls='--', lw=2, color='k', label='model', zorder=0)

                if 0:
                    ax.plot(x, column(y, 'v'), marker='o', ls='-', lw=2, color=color, label='obs')
                else:
                    for k in range(len(y)):
                        if typ[k] == 'm':
                            ax.errorbar([x[k]], [column(y, 0)[k]], yerr=[[column(y, 1)[k]], [column(y, 2)[k]]],
                                        fmt='o', lw=0, elinewidth=1, color=color)
                        if typ[k] == 'u':
                            ax.errorbar([x[k]], [column(y, 0)[k]], yerr=[[0.4], [0.4]],
                                        fmt='o', uplims=0.2, lw=1, elinewidth=1, color=color)

if __name__ == '__main__':
    import sys

    app = QtGui.QApplication([])

    H2 = H2_exc(folder='data/')
    #H2.readmodel(filename='test_s_20.hdf5')
    H2.readfolder()
    #H2.compare(['J0643'])
    H2.plot_H2exc(objects=['J0643'])

    plt.tight_layout()
    plt.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()


