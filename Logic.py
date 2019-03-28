import numpy as np
import Gaussian as filtros
from math import fabs

#import cv2
#import math

ds = None
columns= None
rows= None
data =  None

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


#Funtion that consulte the patient's information
def consultarInformacion():
    inf = "INFORMACION DEL PACIENTE:\n"
    inf += "-Patient's Name:  " + str(ds.PatientName)+"\n"
    inf += "-Patient ID: "+ str(ds.PatientID)+"\n"
    inf += "-Patient's Birth Date: "+ str(ds.PatientBirthDate)+"\n"
    inf += "-Patient's Sex:  " + str(ds.PatientSex)+"\n"
    inf += "-Patient's Age : "+ str(ds.PatientAge )+"\n"
    inf += "-Patient's Weight: "+ str(ds.PatientWeight)+"\n"
   # inf += "-Patient Position: "+ str(ds.PatientPosition)+"\n"    
    inf += "-Additional Patient History:  "+ str(ds.AdditionalPatientHistory)+"\n\n"
    inf += "INFORMACION DE LA IMAGEN\n"
    inf += "-Acquisition Date: "+ str(ds.AcquisitionDate) +"\n"
    inf += "-Slide Name:  " + str(ds.StationName )+"\n"
    inf += "-Study Description:  " + str(ds.StudyDescription) +"\n"
    inf += "-MR Acquisition Type:  " + str(ds.MRAcquisitionType) +"\n"
    inf += "-Repetition Time:  " + str(ds.RepetitionTime) +"\n"

    return inf

#funtion that fill the histogram with the image given 
def llenarHistograma(imagen):
    array = []

    for i in range(np.amax(imagen)+1):
     array.append(0)  

    for i in range(rows):
        for j in range(columns):
            array[imagen[i,j]]=array[imagen[i,j]]+1
    return array


#Funtion that apply the Gaussian filter and return the image filtered
def aplicarFiltroGau(tamano):
    matrizGau,scalarGau = filtros.get_gaussian_filter(int((tamano-1)/2),1)

    filterImageGau = data.copy() 
    filterImageGau = aplicarFiltro(filterImageGau,matrizGau,scalarGau)

    return filterImageGau

#Funtion that apply the Rayleigh filter and return the image filtered
def aplicarFiltroRay(tamano):
    matrizRay,scalarRay = filtros.get_rayleigh_filter(int((tamano-1)/2),1)

    filterImageRay = data.copy() 
    filterImageRay = aplicarFiltro(filterImageRay,matrizRay,scalarRay)

    return filterImageRay

#Auxiliar funtion that sort a list given
def ordenar(lista):
	n = len(lista)

	for i in range (n):
		for j in range (0, n-i-1):
			if lista[j] > lista[j+1]:
				aux = lista[j]
				lista[j] = lista[j+1]
				lista[j+1] = aux
	return lista

#Funtion that apply the Median filter and return the image filtered
def aplicarFiltroMe(imagen, vecinos = 1):
    copy=imagen.copy()
    
    tamKrn = (2*vecinos)+1
    tamArr = tamKrn**2
    for i in range(rows):
        for j in range(columns):
            if i<vecinos or i>((rows-1)-vecinos) or j<vecinos or j>((columns-1)-vecinos):
                copy[i][j]=0
            else:
                lista =[]*tamArr
                for x in range(i-vecinos,i+vecinos+1):
                    for y in range(j-vecinos,j+vecinos+1):
                        lista.append(imagen[x][y])
                lista = ordenar(lista)
                
                mid = int((len(lista)-1)/2)
                copy[i][j] = lista[mid]
    return copy

#Funtion that fill the gradient matrix with the gradient X and the gradient Y
def crearMatrizGradiente(matrizX,matrizY):

    image = np.zeros((rows,columns),dtype=int)
    for i in range(rows):
        for j in range(columns):
            image[i,j]=fabs(matrizX[i,j])+fabs(matrizY[i,j])

    return image

#Funtion that apply the Sobel filter and return the border image 
def aplicarSobel():
 
    imagenBordes = data.copy()
    gradienteX= aplicarFiltro(imagenBordes,matrizSobelX,100)#cv2.filter2D(imagenBordes, -1,  np.asarray(matrizSobelX))
    gradienteY= aplicarFiltro(imagenBordes,matrizSobelY,100)#cv2.filter2D(imagenBordes, -1,  np.asarray(matrizSobelY))

    gradiente=crearMatrizGradiente(gradienteX,gradienteY)
    
    return gradiente

#Funtion that find a threshold from a gradient matrix and the intensities of that matrix
def umbral(intensidad,gradiente):
    
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

#Funtion that apply the otsu filter from a gradient matrix and return the image filtered
def aplicarOtsu(gradiente):
    intensidadesGradiente=llenarHistograma(gradiente)

    umbralBordes = umbral(intensidadesGradiente,gradiente)#Otsu Thresholding

    imagenBordes = definirBordes(gradiente,umbralBordes)

    return imagenBordes

#Funtion that segment the image finding the value of k centroids
def ubicarCentroides(k,data):

    colores=[[255, 255, 255],[0, 0, 0],[0, 0, 255],[255, 0, 0],[255, 255, 0],[0, 255, 0],[255, 0, 255]]
    contador=0

    centroides= np.zeros((rows,columns,3))

    cn = []

    for i in range(k):
        cn.append(int(255/k)*i)

    
    while(contador<2):
        arrayCn = []
        
        for n in range(k):
            arrayCn.append([])
        for i in range(rows):
            for j in range(columns):
                distancias=[]
                for n in range(k):
                    
                    distancias.append(fabs(cn[n]-data[i][j]))

                index=distancias.index(np.amin(distancias))
                arrayCn[index].append(data[i][j])

                centroides[i,j]=colores[index]

        iguales=True                
        for n in range(k):
            if(cn[n]!=int(np.mean(arrayCn[n]))):
                cn[n]=int(np.mean(arrayCn[n]))
                iguales=False
        
        if(iguales):
            contador+=1

    return centroides
        

