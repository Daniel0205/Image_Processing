#Modules imported
import pydicom 
import matplotlib.pyplot as plt
import Gaussian as filtros
from math import fabs

########Read image##############
filename = "MRI Images/MRI01.dcm"

ds = pydicom.dcmread(filename)

rows = int(ds.Rows)
columns = int(ds.Columns)

intensidades = [] ##Histogram Matrix

realizarHistograma()### Create the histogram

umbralBordes = varianzaClase()


################################

#Funtion that fill the histogram's vector array
def realizarHistograma():
	for i in range((2**16)):
		intensidades.append(0)

	for i in range(rows):
		for j in range(columns):
			intensidades[ds.pixel_array[i,j]]=intensidades[ds.pixel_array[i,j]]+1
###############################################	


#Function that apply the convolution filter from a given convolution matrix (3x3)
def filtro(image,matriz,scalar):

	for i in range(rows-1):
		for j in range(columns-1):
			if i==0 or i==rows-1 or j==0 or j==columns-1:
				image[i,j]=0
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
				image[i,j]=aux
	return image
###################################################################################

#Funtion that fills a binary matrix that defines the image's borders
def definirBordes(image,matrizX,matrizY):
	for i in range(rows):
		for j in range(columns):
			if (fabs(matrizX[i,j])+fabs(matrizY[i,j])>umbralBordes):
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
		aux+=(((i-media)**2)*intensidades[i]
	
	aux=aux/pixels

	return aux


#### Calculate Within Class Variance####
def varianzaClase():
	varianzas=[]
	largoVector=len(intensidades)
	for i in range (2**16):
		###Background##
		pesoBack=peso(0,i)
		mediaBack=media(0,i,pesoBack)
		varianzaBack=varianza(0,i,pesoBack,mediaBack)
		pesoBack=pesoBack/(rows*columns)
		###Foreground##
		pesoFor=peso(i,largoVector)
		mediaFor=media(i,largoVector,pesoBack)
		varianzaFor=varianza(i,largoVector,pesoFor,mediaFor)
		pesoFor=pesoBack/(rows*columns)

		varianzas.append(pesoBack*varianzaBack+pesoFor*varianzaFor)

	return min(varianzas)
		
		

############Aplying the filter############
matriz = filtros.get_rayleigh_filter()[0]
scalar = filtros.get_rayleigh_filter()[1]

filterImage = ds.pixel_array.copy() 
filterImage = filtro(filterImage,matriz,scalar)
###########################################

####Creating the borders matrix#########
matrizSobelX=[[-1,0,1],[-2,0,2],[-1,0,1]] #Gradient matrix in X
matrizSobelY=[[-1,-2,-1],[0,0,0],[1,2,1]] #Gradient matrix in Y

imagenBordes = ds.pixel_array.copy()
gradienteX=filtro(imagenBordes,matrizSobelX,1)
gradienteY=filtro(imagenBordes,matrizSobelY,1)

imagenBordes = definirBordes(imagenBordes,gradienteX,gradienteY)
########################################


####Configurating the GUI##################
plt.subplot(1,7,1)
plt.title('DICOM IMAGE')
plt.imshow(ds.pixel_array,cmap=plt.get_cmap('gray'))


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


