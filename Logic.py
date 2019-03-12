#Modules imported
import pydicom 
import matplotlib.pyplot as plt
import Gaussian as filtros
import tkinter as tk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk 
from matplotlib.figure import Figure
from math import fabs
from PIL import Image
from tkinter import ttk


LARGE_FONT = ("Verdana",12)



"""########Read image##############
filename = "MRI Images/MRI01.dcm"

ds = pydicom.dcmread(filename)

rows = int(ds.Rows)
columns = int(ds.Columns)



################################
"""


	

	
###############################################	

def aplicarFiltroGau():
	matrizGau,scalarGau = filtros.get_gaussian_filter()

	filterImageGau = data.copy() 
	filterImageGau = filtro(filterImageGau,matrizGau,scalarGau)

	for widget in ventanaFiltro.winfo_children():
		widget.destroy()

	figure = plt.Figure()
	subPlot = figure.add_subplot(111)
	subPlot.imshow(filterImageGau, cmap=plt.cm.gray)
	imagesTemp = FigureCanvasTkAgg(figure, master=ventanaFiltro)
	imagesTemp.draw()
	imagesTemp.get_tk_widget().pack(padx=5, pady=15)

def aplicarFiltroRay():
	matrizRay,scalarRay = filtros.get_rayleigh_filter()

	filterImageRay = data.copy() 
	filterImageRay = filtro(filterImageRay,matrizRay,scalarRay)

	for widget in ventanaFiltro.winfo_children():
		widget.destroy()

	figure = plt.Figure()
	subPlot = figure.add_subplot(111)
	subPlot.imshow(filterImageRay, cmap=plt.cm.gray)
	imagesTemp = FigureCanvasTkAgg(figure, master=ventanaFiltro)
	imagesTemp.draw()
	imagesTemp.get_tk_widget().pack(padx=5, pady=15)



v

#Funtion that fills a binary matrix that defines the image's borders
def definirBordes(image,matrizX,matrizY):
	for i in range(rows):
		for j in range(columns):
			if ((((matrizX[i,j])**2+(matrizY[i,j])**2)**(1/2))>umbralBordes):
				image[i,j]=0
			else:
				image[i,j]=1
	return image
#####################################################################

######calcular peso ############

def peso(t1,t2):
	aux=0
	for i in range(t1,t2):
		aux+=intensidades[i]
	
	
	return aux

######calcular media ############
def media(t1,t2, pixels):
	aux=0
	for i in range(t1,t2):
		aux+=intensidades[i]*i
	
	aux=aux/pixels

	return aux


######calcular varianza############
def varianza( t1,t2, pixels, media):
	aux=0
	for i in range(t1,t2):
		aux+=((i-media)**2)*intensidades[i]
	
	aux=aux/pixels

	return aux


#### Calculate Within Class Variance####
def varianzaClase(minValue,maxValue):
	varianzas=[]
	
	for i in range (minValue+1,maxValue-1):
		###Background##
		pesoBack=peso(minValue,i)
		print("for: "+str(pesoBack))
		mediaBack=media(minValue,i,pesoBack)
		varianzaBack=varianza(minValue,i,pesoBack,mediaBack)
		pesoBack=pesoBack/(rows*columns)
		###Foreground##
		pesoFor=peso(i,maxValue+1)
		mediaFor=media(i,maxValue+1,pesoFor)

		print("for: "+str(pesoFor))
		varianzaFor=varianza(i,maxValue+1,pesoFor,mediaFor)
		pesoFor=pesoFor/(rows*columns)

		varianzas.append(pesoBack*varianzaBack+pesoFor*varianzaFor)

	
	return varianzas.index(min(varianzas))
		
		
def aplicarBordes():
	umbralBordes = varianzaClase(np.amin(filterImage),np.amax(filterImage))#Otsu Thresholding

	matrizSobelX=[[-1,0,1],[-2,0,2],[-1,0,1]] #Gradient matrix in X
	matrizSobelY=[[-1,-2,-1],[0,0,0],[1,2,1]] #Gradient matrix in Y

	imagenBordes = filterImage.copy()
	gradienteX=filtro(imagenBordes,matrizSobelX,1)
	gradienteY=filtro(imagenBordes,matrizSobelY,1)

	imagenBordes = definirBordes(imagenBordes,gradienteX,gradienteY)



#interfaz grafica
interfaz = tk.Tk()

w = interfaz.winfo_screenwidth() 
h = interfaz.winfo_screenheight()
x = w/2 - 500/2
y = h/2 - 500/2

imagen = tk.Toplevel(interfaz)



#Funtion that fill the histogram's vector array
def realizarHistograma():
	for i in range(np.amax(data)+1):
		intensidades.append(0)

	for i in range(rows):
		for j in range(columns):
			#intensidades[ds.pixel_array[i,j]-1]=intensidades[ds.pixel_array[i,j]-1]+1
			intensidades[data[i,j]]=intensidades[data[i,j]]+1


	f = plt.Figure(figsize=(5,5), dpi=100)
	histograma = f.add_subplot(111)
	histograma.plot(intensidades)

	canvas = FigureCanvasTkAgg(f,imagen)
	canvas.draw()
	canvas.get_tk_widget().pack(side=TOP, fill=tk.BOTH, expand=True)

	toolbar =NavigationToolbar2Tk(canvas,imagen)
	toolbar.update()
	canvas._tkcanvas.pack(side=TOP, fill=tk.BOTH, expand=True)
	print("entro")




"""

class ImageProgram(tk.Tk):

	def __init__(self, *args, **kwargs):
		tk.Tk.__init__(self, *args, **kwargs)
		container = tk.Frame(self)

		container.pack(side="top", fill="both",expand=True)

		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)

		self.frames = {}

		frame = StartPage(container,self)

		self.frames[StartPage] = frame

		frame.grid(row=0, column=0,sticky="nsew")

		self.show_frame(StartPage)


	def show_frame(self,cont):

		print( cont)
		frame = self.frames[cont]
		frame.tkraise()

class StartPage (tk.Frame):
	
	def __init__(self, parent, controller):
		
		tk.Frame.__init__(self, parent)
		label = tk.Label(self,text="Start Page", font=LARGE_FONT)
		label.pack(pady=10,padx=10)

		buttonImage = tk.Button(self,text="Seleccionar imagen", command=lambda: controller.show_frame(PageImage))
		buttonImage.pack()
		


class PageImage(tk.Frame):
	
	def __init__(self, parent, controller):
		tk.Frame.__init__(self,parent)
		label = tk.Label(self,  text = "Image Page", font=LARGE_FONT)
		label.pack(pady=10,padx=10)
		
		buttonBack = tk.Button(self,text="Selectionar imagen", command=lambda: controller.show_frame(StartPage))
		buttonBack.pack()
	
		
		self.resizable(width=False, height=False)
		self.configure(background = 'dark slate gray')
		self.geometry("500x500+%d+%d" %  (x, y))

		
		#combobox para escoger filtro
		filtros = ttk.Combobox(imagen,values=["Gauss","Ray"])
		filtros.pack(padx=5, pady=5, side = tk.TOP)
		filtros.place(x = 200, y =70)
		
		#boton para mostrar histograma
		botonHis = Button(imagen, text = "Histograma", command=realizarHistograma)
		botonHis.pack(padx=5, pady=5, side = tk.TOP)
		botonHis.place(x = 200, y =70)
		


app = ImageProgram()
app.mainloop()

"""

"""
#titulo de la ventana
interfaz.title('Dicom')

#medidas de la ventana, cuestiones de tamano y fondo
interfaz.resizable(width=False, height=False)
interfaz.configure(background = 'dark slate gray')
interfaz.geometry("500x500+%d+%d" %  (x, y)) 


#boton para seleccionar imagen
boton = Button(interfaz, text="Seleccionar imagen", command=lambda: controller.show_frame(PageImage))
boton.pack(padx=5, pady=5,side = tk.TOP)

interfaz.mainloop()
"""



"""


#boton para filtro gaussiano
botonGau = Button(ventana, text = "Filtro gaussiano", command = aplicarFiltroGau)
botonGau.pack(padx=5, pady=5)
botonGau.place(x = 60, y =70)

#boton para rayleight
botonGau = Button(ventana, text = "Rayleigh", command = aplicarFiltroRay)
botonGau.pack(padx=5, pady=5)
botonGau.place(x = 310, y =70)

#boton para mediana
botonMed =  Button(ventana, text = "Bordes", command = aplicarBordes)
botonMed.pack(padx=5, pady=5)
botonMed.place(x = 400, y =70)

#sub-ventana imagen
ventanaImagen=tk.Frame(ventana, bg="dark slate gray")
ventanaImagen.pack(side = tk.RIGHT)

#mostrar informacion imagen
ventanaInfo=tk.Frame(ventana, bg="dark slate gray")
ventanaInfo.pack(side = tk.LEFT, padx = 8, pady = 8)

#mostrar imagen con filtros
ventanaFiltro = tk.Frame(ventana, bg = "dark slate gray")
ventanaFiltro.pack(side = tk.RIGHT, padx = 8, pady = 8)

labelInfo = tk.Label(ventanaInfo, text = "", bg = "dark slate gray", fg = "white", height=30, width=35)
labelInfo.pack()
"""