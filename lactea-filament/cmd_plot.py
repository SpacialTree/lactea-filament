import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        pass

    def band(self, band):
        pass

    def color(self, band1, band2):
        pass

    def plot_color_color(self, band1, band2, band3, band4, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.color(band1, band2), self.color(band3, band4), **kwargs)
        ax.set_xlabel(f'{band1} - {band2}')
        ax.set_ylabel(f'{band3} - {band4}')
        return ax

    def plot_color_mag(self, band1, band2, band3, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.color(band1, band2), self.band(band3), **kwargs)
        ax.set_xlabel(f'{band1} - {band2}')
        ax.set_ylabel(f'{band3}')
        plt.gca().invert_yaxis()
        return ax

    def plot_mag_mag(self, band1, band2, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.band(band1), self.band(band2), **kwargs)
        ax.set_xlabel(f'{band1}')
        ax.set_ylabel(f'{band2}')
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        return ax

    def plot_color_color_color(self, band1, band2, band3, band4, band5, band6, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        im = ax.scatter(self.color(band1, band2), self.color(band3, band4), c=self.color(band5, band6), **kwargs)
        ax.set_xlabel(f'{band1} - {band2}')
        ax.set_ylabel(f'{band3} - {band4}')
        return ax