

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import color
import torch as t
from PIL import Image as im


#Calculation function
def Mandelbrot(zeroShift,visibleRange,sizex,sizey,maxIt):
    
    
    #defining sets for calculations, format pytorch
    x = t.linspace(-visibleRange[0],visibleRange[0], sizex, dtype=t.float64) #w = window's width in pixels; xmin, xmax = left and right borders of the window
    y = t.linspace(-visibleRange[1], visibleRange[1], sizey, dtype=t.float64)#.type(t.half) #h = window's height in pixels; ymin, ymax = bottom and up borders of the window
    x=x+zeroShift[0]
    y=y+zeroShift[1]
    #print(x[1] - x[0]) #Pixel resolution, check if 64bit precision has been reached
    cx, cy = t.meshgrid([x,y],indexing='xy' )
    zx = t.zeros(sizex*sizey, dtype=t.float64).resize_as_(cx)
    zy = t.zeros(sizex*sizey, dtype=t.float64).resize_as_(cy)

    zx2=zx
    zy2=zy
    
    it= t.zeros(sizex*sizey, dtype=t.int32).reshape(sizey,sizex)
    #
    for i in range(0,maxIt):

        '''y= 2 * x * y + cy
        x= x2 - y2 + cx
        x2= x * x
        y2= y * y
        it= it + 1'''
        
        zy=2*zx*zy+cy
        zx= zx2- zy2+cx

        zx2=zx*zx
        zy2=zy*zy
        inf= (zx2 +zy2 )>=100
        it[inf]=i
        #divergence criteria
    return it
#Picturer renderer
def RenderMandelbrot(zeroShift, visibleRange,sizex,sizey,maxIt,c_index,name,loc):
    it=Mandelbrot(zeroShift,visibleRange,sizex,sizey,maxIt).numpy()
    #print(it)
    pixels=np.zeros((sizey,sizex,3),dtype='uint8')
    for i in range(0,sizex):
        for j in range(0,sizey):
          
            colour=c_index[it[j,i]]
            pixels[j,i]=colour



    pixels[int(pixels.shape[0]/2),int(pixels.shape[1]/2)]=[255,0,0]
    #OLD CV Method, does not produce color accurate pixels, needs work
    #pixels=np.array(pixels* 255, dtype = np.uint8)
    #pixels=np.array(pixels,dtype=)
    #vis2 = cv.cvtColor(cv.cvtColor(pixels, cv.COLOR_GRAY2BGR),cv.COLOR_BGR2HSV)
    #cv.imwrite(loc+'/'+name+'_cv.jpg', pixels)
    # cv.imshow("WindowNameHere", vis2)
    #cv.waitKey(0)

    #PIL method, probably slower, but creates the right result
    data = im.fromarray(pixels)
    data.save(loc+'/'+name+'_PIL.png')
    print("done")

#splits the array into ranges according to given percentages
def splitArray(n,percentages):
    parts=len(percentages)
    if (np.sum(percentages)!=100):
        print("Error:sums of percentages must equal 100")
        return np.zeros(parts)
    res=[]
    for percentage in percentages:
        res.append(int(n*percentage/100))
    res[-1]=res[-1]+n-np.sum(res)
    print(res)

    return np.array(res)
    
#defining parameters for the render
shift=4.3299e-15 *7
zeroShift=(-0.749997723532+ 60*(2.0266e-06)-5*(4.5781e-09)+ 20.0075 * 5.6353e-12  -shift  ,0.022448167983 + 20*(2.0266e-06) +7.6261e-13 * 4.001)
sizex,sizey=(1920,1080)
r=2
visibleRange0=(r,r*sizey/sizex)


colors= [[ 0,   7, 100], [32, 107, 203], [237, 255, 255],[255, 170,   0], [0,   2,   0]]
colors.reverse()
loc='images'


#renders multiple images to 
for i in range(0,1000):
    #magnification factor per image
    expFactor=1/12
    magnification=np.exp(i*expFactor)
    n=int(np.exp(i*expFactor/9))
    maxIt=1000*n +2000
    print(magnification)
    print(i)
    visibleRange=(visibleRange0[0]/magnification,visibleRange0[1]/magnification)
    widths=splitArray(maxIt,[25,25,25,25])
    c_index=color.multiGradient(np.array(colors),widths,maxIt,color.cuberp,showGraph=False)

    RenderMandelbrot(zeroShift,visibleRange,sizex,sizey,maxIt,c_index,'mandelbrot '+str(i)+'x',loc)
