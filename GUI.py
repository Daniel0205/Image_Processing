import tkinter as tk                # python 3
import numpy as np
import matplotlib.pyplot as plt
import Gaussian as filtros
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk 
from tkinter import font  as tkfont  # python 3
from tkinter import ttk # python 3
from PIL import Image
from math import fabs


import cv2
import math

LARGE_FONT = ("Verdana",12)

ds = Image.open("MRI Images/lenna.png")
columns,rows=ds.size
data = np.array(ds) 

#Function that apply the convolution filter from a given convolution matrix (3x3)
def aplicarFiltro(image,matriz,scalar):
    copy=image.copy()

    for i in range(rows):
        for j in range(columns):
            if i==0 or i==rows-1 or j==0 or j==columns-1:
                copy[i,j]=0
            else:
                aux=image[i-1,j-1]*matriz[0][0]
                aux+=image[i-1,j]*matriz[0][1]
                aux+=image[i-1,j+1]*matriz[0][2]
                aux+=image[i,j-1]*matriz[1][0]
                aux+=image[i,j]*matriz[1][1]
                aux+=image[i,j+1]*matriz[1][2]
                aux+=image[i+1,j-1]*matriz[2][0]
                aux+=image[i+1,j]*matriz[2][1]
                aux+=image[i+1,j+1]*matriz[2][2]
                aux=aux/scalar
                aux=int(aux)
                copy[i,j]=aux
    return copy
###################################################################################

#### Calculate Within Class Variance####
def varianzaClase(intensidad,gradiente):
    
    total = rows*columns

    sum = 0
    for t in range (0,np.amax(gradiente)+1):
        sum += t * intensidad[t]

    sumB = 0
    wB = 0
    wF = 0

    varMax = 0
    threshold = 0

    for t in range (0,256):
        wB += intensidad[t]               # Weight Background
        if (wB == 0): continue

        wF = total - wB                 # Weight Foreground
        if (wF == 0): break

        sumB += float(t * intensidad[t])

        mB = sumB / wB            # Mean Background
        mF = (sum - sumB) / wF    # Mean Foreground

        #Calculate Between Class Variance
        varBetween = float(wB) * float(wF) * (mB - mF) * (mB - mF)

        #Check if new maximum found
        if (varBetween > varMax) :
            varMax = varBetween
            threshold = t
    
    return threshold


def aplicarBordes(fig,canvas):
 
    matrizSobelX=[[-1,0,1],[-2,0,2],[-1,0,1]] #Gradient matrix in X
    matrizSobelY=[[-1,-2,-1],[0,0,0],[1,2,1]] #Gradient matrix in Y

    imagenBordes = data.copy()
    gradienteX= cv2.filter2D(imagenBordes, -1,  np.asarray(matrizSobelX))#aplicarFiltro(imagenBordes,matrizSobelX,1)
    gradienteY= cv2.filter2D(imagenBordes, -1,  np.asarray(matrizSobelY))#aplicarFiltro(imagenBordes,matrizSobelY,1)

    gradiente=crearMatrizGradiente(gradienteX,gradienteY)
    intensidadesGradiente=llenarHistograma(gradiente)

    umbralBordes = varianzaClase(intensidadesGradiente,gradiente)#Otsu Thresholding

    imagenBordes = definirBordes(gradiente,umbralBordes)

    if(len(fig.get_axes())!=1):
        fig.get_axes()[1].cla()
    bordes = fig.add_subplot(122)
    bordes.imshow(gradiente, cmap=plt.cm.gray)
    canvas.draw()

def crearMatrizGradiente(matrizX,matrizY):

    image = np.zeros((rows,columns),dtype=int)
    for i in range(rows):
        for j in range(columns):
            image[i,j]=fabs(matrizX[i,j])+fabs(matrizY[i,j])#int(((matrizX[i,j])**2+(matrizY[i,j])**2)**(1/2))

    return image

def llenarHistograma(imagen):
    array = []

    for i in range(np.amax(imagen)+1):
     array.append(0)  

    for i in range(rows):
        for j in range(columns):
            #intensidades[ds.pixel_array[i,j]-1]=intensidades[ds.pixel_array[i,j]-1]+1
            array[imagen[i,j]]=array[imagen[i,j]]+1
    return array


#Funtion that fills a binary matrix that defines the image's borders
def definirBordes(gradiente,umbral):
    image = np.zeros((rows,columns),dtype=int)
    for i in range(rows):
        for j in range(columns):
            if (gradiente[i,j]>umbral):
                image[i,j]=1
            else:
                image[i,j]=0
    return image



def aplicarFiltroGau(fig,canvas):
    matrizGau,scalarGau = filtros.get_gaussian_filter()

    filterImageGau = data.copy() 
    filterImageGau = aplicarFiltro(filterImageGau,matrizGau,scalarGau)

    if(len(fig.get_axes())!=1): 
        fig.get_axes()[1].cla()
    filtroGaus = fig.add_subplot(122)
    filtroGaus.imshow(filterImageGau, cmap=plt.cm.gray)
    canvas.draw()


def aplicarFiltroRay(fig,canvas):
    matrizRay,scalarRay = filtros.get_rayleigh_filter()

    filterImageRay = data.copy() 
    filterImageRay = aplicarFiltro(filterImageRay,matrizRay,scalarRay)

    if(len(fig.get_axes())!=1):
        fig.get_axes()[1].cla()
    filtroRay = fig.add_subplot(122)
    filtroRay.imshow(filterImageRay, cmap=plt.cm.gray)
    canvas.draw()


def mostrarHistograma(fig,canvas):

    intensidades=llenarHistograma(data)

    if(len(fig.get_axes())!=1):
        fig.get_axes()[1].cla()
    histograma = fig.add_subplot(122)
    histograma.plot(intensidades)
    canvas.draw()


    

####Configurating the GUI##################

"""
#interfaz grafica
interfaz = tk.Tk()

w = interfaz.winfo_screenwidth() 
h = interfaz.winfo_screenheight()
x = w/2 - 500/2
y = h/2 - 500/2
"""

class ImageProgram(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)


        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, ImagePage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = ttk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(side="top", fill="x", pady=10)

        buttonImage = ttk.Button(self, text="Select an image",
                            command=lambda: controller.show_frame("ImagePage"))

        buttonImage.pack()
        
 #Funtion that fill the histogram's vector array

class ImagePage(tk.Frame):
    
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        
        label = ttk.Label(self,text="Image Page", font=LARGE_FONT)
        label.pack(side="top", fill="x", pady=10)

        fig = plt.Figure(figsize=(5,5), dpi=100)            
        subPlot = fig.add_subplot(121)
        subPlot.imshow(data, cmap=plt.cm.gray)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar =NavigationToolbar2Tk(canvas,self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)        
                
        buttonBack = ttk.Button(self, text="Go to the start page", command=lambda: controller.show_frame("StartPage"))        
        buttonHist = ttk.Button(self, text="Make histogram", command=lambda:mostrarHistograma(fig,canvas))
        buttonFilGau = ttk.Button(self, text="Apply Gauss filter",command=lambda:aplicarFiltroGau(fig,canvas))        
        buttonFilRay = ttk.Button(self, text="Apply Ray filter",command=lambda:aplicarFiltroRay(fig,canvas)) 
        buttonBordes = ttk.Button(self, text="Apply border",command=lambda:aplicarBordes(fig,canvas))        

        buttonBack.pack(side=tk.TOP)
        buttonHist.pack(side=tk.TOP)
        buttonFilGau.pack(side=tk.TOP)
        buttonFilRay.pack(side=tk.TOP)
        buttonBordes.pack(side=tk.TOP)


if __name__ == "__main__":
    app = ImageProgram()
    app.mainloop()