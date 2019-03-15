#!/usr/bin/env python
# coding: utf-8

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

X_input1 = np.load("X_input1.npy")

fig = plt.figure()
ims = [];
title = []
for i in range(10):
    im = plt.imshow(X_input1[i, :, :, 0], animated=True)
    title = plt.title('Frame %f' % (i))
    ims.append([im, title])
ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
ani.save("movie.mp4")
plt.show()
