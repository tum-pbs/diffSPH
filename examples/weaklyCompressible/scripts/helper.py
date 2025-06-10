from sphMath.util import postProcess
import os

exportName = f'18_flowPastSphere'
imagePrefix = f'./images/{exportName}/'
postProcess(imagePrefix, fps = 50,  exportName = 'flowPastSphere')