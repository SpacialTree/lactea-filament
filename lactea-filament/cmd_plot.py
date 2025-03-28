import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        pass

    def band(self, band):
        pass

    def color(self, band1, band2):
        pass

    def plot_CCD(self, band1, band2, band3, band4, ax=None, **kwargs):
        # Color-Color Diagram
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.color(band1, band2), self.color(band3, band4), **kwargs)
        ax.set_xlabel(f'[{band1.upper()}] - [{band2.upper()}]')
        ax.set_ylabel(f'[{band3.upper()}] - [{band4.upper()}]')
        return ax

    def plot_CMD(self, band1, band2, band3, ax=None, **kwargs):
        # Color-Magnitude Diagram
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.color(band1, band2), self.band(band3), **kwargs)
        ax.set_xlabel(f'[{band1.upper()}] - [{band2.upper()}]')
        ax.set_ylabel(f'[{band3.upper()}]')
        plt.gca().invert_yaxis()
        return ax

    def plot_MMD(self, band1, band2, ax=None, **kwargs):
        # Magnitude-Magnitude Diagram
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.band(band1), self.band(band2), **kwargs)
        ax.set_xlabel(f'[{band1.upper()}]')
        ax.set_ylabel(f'[{band2.upper()}]')
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        return ax

    def plot_CCCD(self, band1, band2, band3, band4, band5, band6, ax=None, **kwargs):
        # Color-Color-Color Diagram
        if ax is None:
            ax = plt.gca()
        im = ax.scatter(self.color(band1, band2), self.color(band3, band4), c=self.color(band5, band6), **kwargs)
        ax.set_xlabel(f'[{band1.upper()}] - [{band2.upper()}]')
        ax.set_ylabel(f'[{band3.upper()}] - [{band4.upper()}]')
        plt.colorbar(im, ax=ax, label=f'[{band5.upper()}] - [{band6.upper()}]')
        return ax