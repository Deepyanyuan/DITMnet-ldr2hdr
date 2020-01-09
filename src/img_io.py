from __future__ import division

# -*- encoding: utf-8 -*-
"""
@File    : img_io.py
@Time    : 2019/11/13 18:21
@Author  : Beny Liang
@Email   : 18203416228@163.com
@Software: PyCharm
"""
'''
script description:

'''
# ----- Load libraries -------------------------------------
import numpy as np
import scipy.misc
import OpenEXR, Imath, cv2

# ----- Custom functions -----
class IOException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


# Read and prepare 8-bit image in a specified resolution
def readLDR(file, height, width):
    try:
        x_buffer = cv2.imread(file)
        x_buffer = cv2.cvtColor(x_buffer, cv2.COLOR_BGR2RGB)
        x_buffer = (x_buffer / 255.).astype(np.float32)
        res = cv2.resize(x_buffer, (height, width))
        res = np.reshape(res, (1, height, width, -1))

        return res

    except Exception as e:
        raise IOException("Failed reading LDR image: %s" % e)


# Read and prepare 8-bit image in a specified resolution
def readHDR(file, height, width):
    try:

        x_buffer = cv2.imread(file, flags=cv2.IMREAD_ANYDEPTH)
        x_buffer = cv2.cvtColor(x_buffer, cv2.COLOR_BGR2RGB)
        x_buffer = x_buffer.astype(np.float32)
        res = cv2.resize(x_buffer, (height, width))

        return res

    except Exception as e:
        raise IOException("Failed reading HDR image: %s" % e)


# Write exposure compensated 8-bit image
def writeLDR(img, file, exposure=0):

    # Convert exposure fstop in linear domain to scaling factor on display values
    sc = np.power(np.power(2.0, exposure), 0.5)

    try:
        scipy.misc.toimage(sc * np.squeeze(img), cmin=0.0, cmax=1.0).save(file)
    except Exception as e:
        raise IOException("Failed writing LDR image: %s" % e)


# Write HDR image using OpenEXR
def writeEXR(img, file):
    try:
        img = np.squeeze(img)
        sz = img.shape
        header = OpenEXR.Header(sz[1], sz[0])
        half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        out = OpenEXR.OutputFile(file, header)
        # # original code
        R = (img[:,:,0]).astype(np.float16).tostring()
        G = (img[:,:,1]).astype(np.float16).tostring()
        B = (img[:,:,2]).astype(np.float16).tostring()

        out.writePixels({'R': R, 'G': G, 'B': B})
        out.close()
    except Exception as e:
        raise IOException("Failed writing EXR: %s" % e)