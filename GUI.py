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

matrizSobelX=[[-1,0,1],[-2,0,2],[-1,0,1]] #Gradient matrix in X
matrizSobelY=[[-1,-2,-1],[0,0,0],[1,2,1]] #Gradient matrix in Y

#Function that apply the convolution filter from a given convolution matrix (3x3)
def aplicarFiltro(image,kernel,scalar):
	copy = image.copy()
	s = len(kernel)#Tamaño del Kernel
	n = int((s-1)/2)#numero de vecinos

	for i in range(rows):
		for j in range(columns):
			if(i<n or i>((rows-1)-n) or j<n or j>((columns-1)-n)):
				copy[i,j]=0
			else:
				px=0
				py=0
				aux=0.0
				for kx in range(i-n,i+n+1):
					
					for ky in range(j-n,j+n+1):

						img = image[kx][ky]
						krn = kernel[px][py]

						aux = aux + (img*krn) 

						py += 1

					#end for ky
					px += 1
					py = 0
				#end for kx

				aux = aux/scalar
				aux = int(aux)
				copy[i][j] = aux
						
		#end for j
	#end for i

	return copy
###################################################################################

#### Calculate Within Class Variance####


def llenarHistograma(imagen):
    array = []

    for i in range(np.amax(imagen)+1):
     array.append(0)  

    for i in range(rows):
        for j in range(columns):
            #intensidades[ds.pixel_array[i,j]-1]=intensidades[ds.pixel_array[i,j]-1]+1
            array[imagen[i,j]]=array[imagen[i,j]]+1
    return array


def aplicarFiltroGau(tamano):
    matrizGau,scalarGau = filtros.get_gaussian_filter(int((tamano-1)/2),1)

    filterImageGau = data.copy() 
    filterImageGau = aplicarFiltro(filterImageGau,matrizGau,scalarGau)

    return filterImageGau


def aplicarFiltroRay(tamano):
    matrizRay,scalarRay = filtros.get_rayleigh_filter(int((tamano-1)/2),1)

    filterImageRay = data.copy() 
    filterImageRay = aplicarFiltro(filterImageRay,matrizRay,scalarRay)

    return filterImageRay

    
def crearMatrizGradiente(matrizX,matrizY):

    image = np.zeros((rows,columns),dtype=int)
    for i in range(rows):
        for j in range(columns):
            image[i,j]=fabs(matrizX[i,j])+fabs(matrizY[i,j])#int(((matrizX[i,j])**2+(matrizY[i,j])**2)**(1/2))

    return image


def aplicarSobel():
 
    imagenBordes = data.copy()
    gradienteX= aplicarFiltro(imagenBordes,matrizSobelX,100)#cv2.filter2D(imagenBordes, -1,  np.asarray(matrizSobelX))
    gradienteY= aplicarFiltro(imagenBordes,matrizSobelY,100)#cv2.filter2D(imagenBordes, -1,  np.asarray(matrizSobelY))

    gradiente=crearMatrizGradiente(gradienteX,gradienteY)
    
    return gradiente


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

def aplicarOtsu(gradiente):
    intensidadesGradiente=llenarHistograma(gradiente)

    umbralBordes = varianzaClase(intensidadesGradiente,gradiente)#Otsu Thresholding

    imagenBordes = definirBordes(gradiente,umbralBordes)

    return imagenBordes


def ubicarCentroides():
    contador=0

    centroides= data.copy()

    c1=0
    c2=int(np.amax(data)/2)
    c3=np.amax(data)

    
    while(contador<2):
        arrayC1 = []
        arrayC2 = []
        arrayC3 = []
        for i in range(rows):
            for j in range(columns):
                distanciaC1 = fabs(data[i,j]-c1)
                distanciaC2 = fabs(data[i,j]-c2)
                distanciaC3 = fabs(data[i,j]-c3)

                if(distanciaC1<=distanciaC2 and distanciaC1<=distanciaC3):
                    centroides[i,j]=1
                    arrayC1.append(data[i,j])
                elif(distanciaC2<=distanciaC1 and distanciaC2<=distanciaC3):
                    centroides[i,j]=150
                    arrayC2.append(data[i,j])
                else: 
                    centroides[i,j]=250
                    arrayC3.append(data[i,j])
            #print("row: " + str(i)+" columns: "+str(j))
                
        aux1=int(np.mean(arrayC1))
        aux2=int(np.mean(arrayC2))
        aux3=int(np.mean(arrayC3))

        if(c1==aux1 and c2==aux2 and c3==aux3):
            contador=contador+1
        c1=aux1
        c2=aux2
        c3=aux3

    return centroides
        





    

####Configurating the GUI##################

def aplicarFiltros(fig,canvas,filtro,size):
    imagenF=[]

    tamano=int(size[:size.index("x")])

    if(filtro=="Gaussiano"):
        imagenF=aplicarFiltroGau(tamano)       

    elif(filtro=="Rayleigh"):
        imagenF=aplicarFiltroRay(tamano)

    elif(filtro=="Sobel"):
        imagenF=aplicarSobel()

    elif(filtro=="Otsu"):
        imagenF=aplicarSobel()
        imagenF=aplicarOtsu(imagenF)

    

    if(len(fig.get_axes())!=1):
        fig.get_axes()[1].cla()
    filtroRay = fig.add_subplot(122)
    filtroRay.imshow(imagenF, cmap=plt.cm.gray)
    canvas.draw()

def aplicarKMeans(fig,canvas):
    
    centroides=ubicarCentroides()

    if(len(fig.get_axes())!=1): 
        fig.get_axes()[1].cla()
    filtroGaus = fig.add_subplot(122)
    filtroGaus.imshow(centroides)
    canvas.draw()

def mostrarHistograma(fig,canvas):

    intensidades=llenarHistograma(data)

    if(len(fig.get_axes())!=1):
        fig.get_axes()[1].cla()
    histograma = fig.add_subplot(122)
    histograma.plot(intensidades)
    canvas.draw()


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

        cb = ttk.Combobox(self, values=("Gaussiano", "Rayleigh", "Mediana", "Sobel", "Otsu"),state="readonly")
        cb.set("Select a filter")

        size = ttk.Combobox(self, values=("3x3", "5x5", "7x7", "9x9", "11x11"),state="readonly")
        size.set("3x3")
        #cb.bind('<<ComboboxSelected>>', lambda x:self.asignarTamano(size,cb.get()))
                        
        buttonBack = ttk.Button(self, text="Go to the start page", command=lambda: controller.show_frame("StartPage"))        
        buttonHist = ttk.Button(self, text="Make Histogram", command=lambda:mostrarHistograma(fig,canvas))
        buttonFiltros = ttk.Button(self, text="Apply Filter",command=lambda:aplicarFiltros(fig,canvas,cb.get(),size.get()))          
        buttonKMeans = ttk.Button(self, text="Apply k-means",command=lambda:aplicarKMeans(fig,canvas))        
        
        cb.pack(side=tk.TOP)
        size.pack(side=tk.TOP)
        buttonFiltros.pack(side=tk.TOP)
        buttonBack.pack(side=tk.TOP)
        buttonHist.pack(side=tk.TOP)        
        buttonKMeans.pack(side=tk.TOP)
    
    def asignarTamano(self,size,filtro):

        if(filtro=="Gaussiano"):
            size = ttk.Combobox(self, values=("3x3", "5x5", "7x7", "9x9", "11x11"),state="readonly")    
            print("entro0")

        elif(filtro=="Rayleigh"):
            size = ttk.Combobox(self, values=("3x3", "5x5", "7x7", "9x9", "11x11"),state="readonly")
            print("entro1")        

        elif(filtro=="Sobel"):
            size = ttk.Combobox(self, values=("3x3"),state="readonly")
            print("entro2")

        elif(filtro=="Otsu"):
            size = ttk.Combobox(self, values=("3x3"),state="readonly")
            print("entro3")
        
        size.set("holasdf")
        

        

if __name__ == "__main__":
    app = ImageProgram()
    app.mainloop()