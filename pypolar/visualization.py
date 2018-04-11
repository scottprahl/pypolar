#/usr/local/bin/python

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation

__all__ = ['drawPropagatingWave',
           'drawPhaseDiagram',
           'showPolarization',
           'showPolarizationAnimation']


def drawAxis(v, ax, last=4 * np.pi):
    Ha, Va = np.abs(v)
    the_max = max(Ha, Va) * 1.1

    ax.plot([0, last], [0, 0], [0, 0], 'k')
    ax.plot([0, 0], [-the_max, the_max], [0, 0], 'g')
    ax.plot([0, 0], [0, 0], [-the_max, the_max], 'b')
    return


def drawHwave(v, ax, last=4 * np.pi, offset=0):
    Hamp = abs(v[0])
    Hshift = np.angle(v[0])

    t = np.linspace(0, last, 100) + offset
    x = t - offset
    y = Hamp * np.cos(t - Hshift)
    z = 0 * t
    ax.plot(x, y, z, ':g')
    return


def drawVwave(v, ax, last=4 * np.pi, offset=0):
    Vamp = abs(v[1])
    Vshift = np.angle(v[1])

    t = np.linspace(0, last, 100) + offset
    x = t - offset
    y = 0 * t
    z = Vamp * np.cos(t - Vshift)
    ax.plot(x, y, z, ':b')
    return


def drawSumwave(v, ax, last=4 * np.pi, offset=0):
    Hamp, Vamp = np.abs(v)
    Hshift, Vshift = np.angle(v)

    t = np.linspace(0, last, 100) + offset
    x = t - offset
    yH = 0 * t
    yV = Hamp * np.cos(t - Hshift)
    zH = 0 * t
    zV = Vamp * np.cos(t - Vshift)
    y = yH + yV
    z = zH + zV
    ax.plot(x, y, z, 'r')
    return


def drawVectorSum(v, ax, offset=0):
    Hamp, Vamp = np.abs(v)
    Hshift, Vshift = np.angle(v)

    t = offset
    yH = 0 * t
    yV = Hamp * np.cos(t - Hshift)
    zH = 0 * t
    zV = Vamp * np.cos(t - Vshift)
    y = yH + yV
    z = zH + zV

    x1 = 0
    y1 = y
    z1 = 0
    x2 = 0
    y2 = y
    z2 = z
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'g--')

    x1 = 0
    y1 = 0
    z1 = z
    x2 = 0
    y2 = y
    z2 = z
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'b--')

    x1 = 0
    y1 = 0
    z1 = 0
    x2 = 0
    y2 = y
    z2 = z
    ax.plot([x1, x2], [y1, y2], [z1, z2], 'r')
    ax.plot([x2], [y2], [z2], 'ro')
    return


def drawPropagatingWave(v, ax, off=0):

    drawAxis(v, ax)
    drawHwave(v, ax, offset=off)
    drawVwave(v, ax, offset=off)
    drawSumwave(v, ax, offset=off)
    drawVectorSum(v, ax, offset=off)

    ax.grid(False)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return


def drawPhaseDiagram(v, ax, off=0):
    Ha, Va = np.abs(v)
    Hoffset, Voffset = np.angle(v)
    the_max = max(Ha, Va) * 1.1

    xmax = the_max
    ymax = the_max
    ax.plot([-xmax, xmax], [0, 0], 'g')
    ax.plot([0, 0], [-ymax, ymax], 'b')

    t = np.linspace(0, 2 * np.pi, 100) + off
    x = Ha * np.cos(t - Hoffset)
    y = Va * np.cos(t - Voffset)
    ax.plot(x, y, 'k')

    t = off
    x = Ha * np.cos(t - Hoffset)
    y = Va * np.cos(t - Voffset)
    ax.plot(x, y, 'ro')
    ax.plot([x, x], [0, y], 'g--')
    ax.plot([0, x], [y, y], 'b--')
    ax.plot([0, x], [0, y], 'r')

    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return


def showPolarization(v, offset=0):
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0], projection='3d')
    ax2 = plt.subplot(gs[1])
    drawPropagatingWave(v, ax1, off=offset)
    drawPhaseDiagram(v, ax2, off=offset)
    return plt


def ani_update(startAngle, v, ax1, ax2):
    ax1.clear()
    ax2.clear()
    drawPropagatingWave(v, ax1, off=startAngle)
    drawPhaseDiagram(v, ax2, off=startAngle)
    return ax1, ax2


def showPolarizationAnimation(v):
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1 = plt.subplot(gs[0], projection='3d')
    ax2 = plt.subplot(gs[1])

    startAngle = 0
    drawPropagatingWave(v, ax1, off=startAngle)
    drawPhaseDiagram(v, ax2, off=startAngle)

    ani = animation.FuncAnimation(fig, ani_update, frames=np.linspace(
        0, 2 * np.pi, 64), fargs=(v, ax1, ax2))
    return ani
