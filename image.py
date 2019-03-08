#Modules imported
import pydicom 
import matplotlib.pyplot as plt
import Gaussian as filtros
from math import fabs
from PIL import Image
import numpy as np

"""########Read image##############
filename = "MRI Images/MRI01.dcm"

ds = pydicom.dcmread(filename)

rows = int(ds.Rows)
columns = int(ds.Columns)



################################
"""


ds = Image.open("MRI Images/lenna.png")
rows,columns=ds.size
data = np.array(ds) 

intensidades = [] ##Histogram Matrix


#Funtion that fill the histogram's vector array
def realizarHistograma(matriz):
	for i in range(np.amax(matriz)+1):
		intensidades.append(0)

	for i in range(rows):
		for j in range(columns):
			#intensidades[ds.pixel_array[i,j]-1]=intensidades[ds.pixel_array[i,j]-1]+1
			intensidades[matriz[i,j]]=intensidades[matriz[i,j]]+1

	
###############################################	


#Function that apply the convolution filter from a given convolution matrix (3x3)
def filtro(image,matriz,scalar):
	copy=image.copy()

	for i in range(rows):
		for j in range(columns):
			if i==0 or i==rows-1 or j==0 or j==columns-1:
				copy[i,j]=image[i,j]
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
		
		

############Aplying the filter############
matriz = filtros.get_gaussian_filter()[0]
scalar = filtros.get_gaussian_filter()[1]

filterImage = data.copy() 
filterImage = filtro(filterImage,matriz,scalar)


###########################################

####Creating the borders matrix#########
realizarHistograma(filterImage)### Create the histogram
umbralBordes = varianzaClase(np.amin(filterImage),np.amax(filterImage))#Otsu Thresholding

print(umbralBordes)


matrizSobelX=[[-1,0,1],[-2,0,2],[-1,0,1]] #Gradient matrix in X
matrizSobelY=[[-1,-2,-1],[0,0,0],[1,2,1]] #Gradient matrix in Y

imagenBordes = filterImage.copy()
gradienteX=filtro(imagenBordes,matrizSobelX,1)
gradienteY=filtro(imagenBordes,matrizSobelY,1)

imagenBordes = definirBordes(imagenBordes,gradienteX,gradienteY)
########################################


####Configurating the GUI##################
plt.subplot(1,7,1)
plt.title('DICOM IMAGE')
plt.imshow(data,cmap=plt.get_cmap('gray'))


plt.subplot(1,7,3)
plt.title('HISTOGRAM')
plt.plot(intensidades)

plt.subplot(1,7,5)
plt.title('IMAGE-RAYLEIGH FILTER') 
plt.imshow(filterImage,cmap=plt.get_cmap('gray'))

plt.subplot(1,7,7)
plt.title('IMAGE-BORDERS') 
plt.imshow(imagenBordes,cmap=plt.get_cmap('gray'))

plt.show()
###########################################


