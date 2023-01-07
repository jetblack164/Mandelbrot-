#Companding functions and algorithms taken from: https://stackoverflow.com/questions/22607043/color-gradient-algorithm

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
def SrgbCompanding(c):
    #Convert color from 0..255 to 0..1
    r = c[0] #/ 255
    g = c[1] #/ 255
    b = c[2] #/ 255
    
    #Apply companding to Red, Green, and Blue
    if (r > 0.0031308): r = 1.055*np.power(r, 1/2.4)-0.055 
    else: r = r * 12.92
    if (g > 0.0031308): g = 1.055*np.power(g, 1/2.4)-0.055 
    else: g = g * 12.92
    if (b > 0.0031308): b = 1.055*np.power(b, 1/2.4)-0.055 
    else: b = b * 12.92

    #//return new color. Convert 0..1 back into 0..255
    result=np.zeros(3)
    result[0] = r*255
    result[1] = g*255
    result[2] = b*255

    return result


def InverseSrgbCompanding(c):

    #Convert color from 0..255 to 0..1
    r = c[0] / 255
    g = c[1] / 255
    b = c[2] / 255

    #Inverse Red, Green, and Blue
    if (r > 0.04045): r = np.power((r+0.055)/1.055, 2.4) 
    else: r = r / 12.92
    if (g > 0.04045): g = np.power((g+0.055)/1.055, 2.4)
    else: g = g / 12.92
    if (b > 0.04045): b = np.power((b+0.055)/1.055, 2.4) 
    else: b = b / 12.92

    #return new color. Convert 0..1 back into 0..255
    result=np.zeros(3)
    result[0] = r#*255
    result[1] = g#*255
    result[2] = b#*255

    return result

'''def lerp(c1, c2, frac):
    return c1 * (1 - frac) + c2 * frac'''
#Interpolation functions used for the gradients
def lerp(x,y,maxIt):
    result=np.zeros((x[-1],3))
    for i in range(1,len(x)):
        #temp=np.arange(x[i-1],x[i])
        fracs=np.linspace(0,1,x[i]-x[i-1])
        
        c1=y[i-1]
        c2=y[i]
        temp=np.zeros((x[i]-x[i-1],3))
        for j,frac in enumerate(fracs):
            result[j+x[i-1]] =c1*(1-frac)+c2*(frac)
    return result
        

def cuberp(x,y,maxIt):

    fit=CubicSpline(x,y, bc_type='clamped')
    x=np.arange(maxIt)
    return fit(x)
#OLD gradient code, did not allow more than 2 colours
'''def gradient(c1,c2,interp,mix):
    c1=InverseSrgbCompanding(c1)
    c2=InverseSrgbCompanding(c2)
    c=interp(c1,c2,mix)
    gamma=0.43
    #lerp the brightness
    b1=np.power(np.sum(c1),gamma)
    b2=np.power(np.sum(c2),gamma)
    b=interp(b1,b2,mix)
    intensity=np.power(b,1/gamma)
    if(np.sum(c)!=0):
        c*intensity/np.sum(c)
    return SrgbCompanding(c)
'''

def multiGradient(colors,widths,maxIt,interp,showGraph):
    if(np.sum(widths)!=maxIt):
        print("ERROR: sum of Widths and maxIt do not coincide "+str(np.sum(widths))+"=/="+str(maxIt))
        return c_index
    gamma=0.43
    colors_original=colors
    colors=np.array([InverseSrgbCompanding(color) for color in colors])
    b=[np.power(np.sum(color),gamma) for color in colors]
    intensity=np.power(b,1/gamma)
    boundaries=[]
    i=0
    for width in widths:
        boundaries.append(i)
        i=i+width
    boundaries.append(i)

    intensity=interp(boundaries,intensity,maxIt)
    c_index=interp(boundaries,colors,maxIt)


    c_return=[c*intensity[i]/np.sum(c) for i,c in enumerate(c_index) if(np.sum(c_index[i])!=0) ]
    c_return=np.array([SrgbCompanding(c) for c  in c_return]).clip(0,255)
    #allows visualisation of plots 
    if(showGraph):
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(c_return[:,0],c_return[:,1],c_return[:,2])
        colors=colors_original
        ax.scatter(colors[:,0],colors[:,1],colors[:,2],color='red')
        print(c_return[15])
        plt.show()

    return c_return



'''

LEGACY CODE
maxIt=256
colors= [[ 0,   7, 100], [32, 107, 203], [237, 255, 255],[255, 170,   0], [0,   2,   0]]
colors.reverse()
widths=[16,16,128,64+32]
c_index=multiGradient(np.array(colors),widths,maxIt,cuberp,True)

boundaries=[]
i=0
for width in widths:
    boundaries.append(i)
    i=i+width
boundaries.append(i)
print(lerp(np.array(boundaries),np.array(colors),255))'''

"""def gradient(c1,c2,maxIt):
        #Create Gradient Array
    c_index=[]
    for i in range(0,maxIt+1):
        c_index.append(color.gradient(c1,c2,i/maxIt))
    #c_index[0]=[0,0,0]
    return c_index

def multiGradient(colors,widths,maxIt):
    c_index=np.zeros((maxIt+1,3))
    if(np.sum(widths)!=maxIt):
        print("ERROR: sum of Widths and maxIt do not coincide "+str(np.sum(widths))+"=/="+str(maxIt))
        return c_index
    it=0
    color=colors[0]
    for bandIt in range(1,len(colors)):
        bandGradient=gradient(color,colors[bandIt],widths[bandIt-1])

        for i in range(0,len(bandGradient)):
            c_index[it+i]=bandGradient[i]
        #print(it)
        it=it+widths[bandIt-1]
        color=colors[bandIt]

    return c_index"""