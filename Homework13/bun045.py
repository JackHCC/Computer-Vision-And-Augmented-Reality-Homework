#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
from plyfile import *
from mpl_toolkits.mplot3d import Axes3D

plydata = PlyData.read('./bun045.ply')
# plydata = PlyData.read('./ellipsoid.ply')
# plydata = PlyData.read('./dragon.ply')

xlist = plydata['vertex']['x']
ylist = plydata['vertex']['y']
zlist = plydata['vertex']['z']

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xlist, ylist, zlist)
plt.show()

