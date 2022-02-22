import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def connect_data_points(p1, p2, dash=False):

    # c = cm.rainbow(np.linspace(0, 1, 0))

    if dash:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', color= 'k')
    else:
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=  'k')

    """

    Draws a line from point p1 to point p2.

    Parameters
    ----------
    p1 : ndarray
        Point 1.
    p2 : ndarray
        Point 2.
    dash : bool
        True to plot dash line.

    """