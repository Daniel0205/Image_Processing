import tkinter as tk                # python 3
import numpy as np
import matplotlib.pyplot as plt
import Gaussian as filtros
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk 
from tkinter import font  as tkfont  # python 3
from tkinter import ttk # python 3
from PIL import Image
from math import fabs

import pydicom

import cv2
import math

LARGE_FONT = ("Verdana",12)

ds = pydicom.dcmread("Brain Images/brain_001.dcm")
columns= int(ds.Rows)
rows=int(ds.Columns)
data = ds.pixel_array.copy()

matrizSobelX=[[-1,0,1],[-2,0,2],[-1,0,1]] #Gradient matrix in X
matrizSobelY=[[-1,-2,-1],[0,0,0],[1,2,1]] #Gradient matrix in Y

#Function that apply the convolution filter from a given convolution matrix (3x3)
def aplicarFiltro(image,kernel,scalar):
	copy = image.copy()
	tamano = len(kernel)#Tamaño del Kernel
	vecinos = int((tamano-1)/2)#numero de vecinos

	for i in range(rows):
		for j in range(columns):
			if(i<vecinos or i>((rows-1)-vecinos) or j<vecinos or j>((columns-1)-vecinos)):
				copy[i,j]=0
			else:
				px=0
				py=0
				aux=0.0
				for kx in range(i-vecinos,i+vecinos+1):					
					for ky in range(j-vecinos,j+vecinos+1):

						aux = aux + (image[kx][ky]*kernel[px][py]) 
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


def filtroMediana(imagen, vecinos = 1):
    copy = imagen.copy()
    for i in range(rows):
    	for j in range(columns):
    		if i<vecinos or i>((rows-1)-vecinos) or j<n or j>((columns-1)-vecinos):
    			copy[i][j]=imagen[i][j]
    		else:
    			lista =[]*9
    			mid = 4

    			for x in range(i-vecinos,i+vecinos+1):
    				for y in range(j-vecinos,j+vecinos+1):
    					lista.append(imagen[x][y])
    			lista = ordenar(lista)

    			copy[i][j] = lista[4]

    return copy

    
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

def calcularDirecciones():

    direcciones=data.copy()
    
    gradienteX= aplicarFiltro(data.copy(),matrizSobelX,100)#cv2.filter2D(imagenBordes, -1,  np.asarray(matrizSobelX))
    gradienteY= aplicarFiltro(data.copy(),matrizSobelY,100)#cv2.filter2D(imagenBordes, -1,  np.asarray(matrizSobelY))

    for i in range(rows):
        for j in range(columns):
            angulo=np.arctan2(gradienteY[i,j], gradienteX[i,j])* 180 / np.pi

            angulo=normalizarAngulo(angulo)
            direcciones[i,j]=angulo

    return direcciones


def normalizarAngulo(angulo):
    if (int(angulo)==0):
        angulo==0
    elif(angulo<=45):
        angulo=45
    else:
        angulo=90

    return int(angulo)


def calcularDistancia(cn,direcciones,i,j,n):

    aux=(data[i,j]-cn[n][0])**2
    aux+=(direcciones[i,j]-cn[n][1])**2
    aux=aux**(1/2)

    return aux


def ubicarCentroides(k,direcciones):
    contador=0

    centroides= data.copy()

    cn = []

    for i in range(k):
        cn.append([int(255/k)*i,normalizarAngulo(i)])

    
    while(contador<2):
        arrayCn = []
        
        for n in range(k):
            arrayCn.append([])
        for i in range(rows):
            for j in range(columns):
                distancias=[]
                for n in range(k):
                    distancias.append(calcularDistancia(cn,direcciones,i,j,n))

                index=distancias.index(np.amin(distancias))
                arrayCn[index].append([data[i,j],direcciones[i,j]])

                centroides[i,j]=index*int(255/k)
            #print("row: " + str(i)+" columns: "+str(j))

        iguales=True                
        for n in range(k):
        	if(cn[n][0]!=int(np.mean(arrayCn[n][0])) or cn[n][1]!=normalizarAngulo(int(np.mean(arrayCn[n][1])))):
        		cn[n]=[int(np.mean(arrayCn[n][0])),normalizarAngulo(int(np.mean(arrayCn[n][1])))]
        		iguales=False
        
        if(iguales):
            contador+=1
        print("Contador:"+str(contador))

    return centroides
        

####Configurating the GUI##################

def aplicarFiltros(fig,canvas,filtro,size):
    imagenF=[]

    tamano=int(size[:size.index("x")])

    if(filtro=="Gaussiano"):
        imagenF=aplicarFiltroGau(tamano)       

    elif(filtro=="Rayleigh"):
        imagenF=aplicarFiltroRay(tamano)
    
    elif (filtro=="Mediana"):
        imagenF=aplicarFiltroMe(data.copy())

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


def aplicarKMeans(fig,canvas,k):
    if(k!="Select number of centroids"):

        direccionesGrad=calcularDirecciones()

        centroides=ubicarCentroides(int(k),direccionesGrad)

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

        centroidsNum = ttk.Combobox(self, values=("2", "3", "4", "5"),state="readonly")
        centroidsNum.set("Select number of centroids")
                        
        buttonBack = ttk.Button(self, text="Go to the start page", command=lambda: controller.show_frame("StartPage"))        
        buttonHist = ttk.Button(self, text="Make Histogram", command=lambda:mostrarHistograma(fig,canvas))
        buttonFiltros = ttk.Button(self, text="Apply Filter",command=lambda:aplicarFiltros(fig,canvas,cb.get(),size.get()))          
        buttonKMeans = ttk.Button(self, text="Apply k-means",command=lambda:aplicarKMeans(fig,canvas,centroidsNum.get()))        
        
        cb.pack(side=tk.TOP)
        size.pack(side=tk.TOP)
        buttonFiltros.pack(side=tk.TOP)
        buttonBack.pack(side=tk.TOP)
        buttonHist.pack(side=tk.TOP)  
        centroidsNum.pack(side=tk.TOP)      
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