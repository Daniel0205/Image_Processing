import tkinter as tk                # python 3
import Logic as Lg
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk 
from tkinter import font  as tkfont  # python 3
from tkinter import ttk # python 3
from tkinter import *
from PIL import Image


LARGE_FONT = ("Verdana",12)#font

####Configurating the GUI##################

#Funtion where the image is selected
def seleccionarImagen(parent,controller):
    global ds,columns,rows,data

    filename = filedialog.askopenfilename()   

    if(filename.find(".dcm")!=-1):
        Lg.ds = pydicom.dcmread(filename) 
        Lg.data =  Lg.ds.pixel_array.copy()
        Lg.columns=  int(Lg.ds.Rows)
        Lg.rows= int(Lg.ds.Columns)

    else:
        Lg.ds = Image.open(filename)
        Lg.columns,rows=Lg.ds.size
        Lg.data = np.array(Lg.ds) 

    page_name = ImagePage.__name__
    frame = ImagePage(parent=parent, controller=controller,filename=filename)
    controller.frames[page_name] = frame

    frame.grid(row=0, column=0, sticky="nsew")
    controller.show_frame("ImagePage")


#Funtion that decide wich filter apply and draw the image in the GUI
def aplicarFiltros(fig,canvas,filtro,size):
    imagenF=[]

    tamano=int(size[:size.index("x")])

    if(filtro=="Gaussiano"):
        imagenF=Lg.aplicarFiltroGau(tamano)       

    elif(filtro=="Rayleigh"):
        imagenF=Lg.aplicarFiltroRay(tamano)
    
    elif (filtro=="Mediana"):
        imagenF=Lg.aplicarFiltroMe(Lg.data.copy(),int((tamano-1)/2))

    elif(filtro=="Sobel"):
        imagenF=Lg.aplicarSobel()

    elif(filtro=="Otsu"):
        imagenF=Lg.aplicarSobel()
        imagenF=Lg.aplicarOtsu(imagenF)

    
    if(len(fig.get_axes())!=1):
        fig.get_axes()[1].cla()
    filtroRay = fig.add_subplot(122)
    filtroRay.imshow(imagenF, cmap=plt.cm.gray)
    canvas.draw()


#Funtion that apply and show the segmented image
def aplicarKMeans(fig,canvas,k):
    if(k!="Select number of centroids"):

        centroides=Lg.ubicarCentroides(int(k))

        if(len(fig.get_axes())!=1): 
            fig.get_axes()[1].cla()
        filtroGaus = fig.add_subplot(122)
        filtroGaus.imshow(centroides)
        canvas.draw()

#Funtion that build and show the histogram
def mostrarHistograma(fig,canvas):

    intensidades=Lg.llenarHistograma(Lg.data)

    if(len(fig.get_axes())!=1):
        fig.get_axes()[1].cla()
    histograma = fig.add_subplot(122)
    histograma.plot(intensidades)
    canvas.draw()

#Controller Frame 
class ImageProgram(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=500)
        container.grid_columnconfigure(0, weight=500)

        self.frames = {}
        page_name = StartPage.__name__
        frame = StartPage(parent=container, controller=self)
        self.frames[page_name] = frame

        w = self.winfo_screenwidth() 
        h = self.winfo_screenheight()
        x = w/2 - 500/2
        y = h/2 - 500/2
        self.geometry("500x500+%d+%d" %  (x, y))

        # put all of the pages in the same location
        # the one on the top of the stacking order
        # will be the one that is visible.
        frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()

#Main frame - start page
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = ttk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(side=tk.TOP)

        buttonImage = ttk.Button(self, text="Select an image",
                            command=lambda: seleccionarImagen(parent,controller))#controller.show_frame("ImagePage"))

        buttonImage.pack(side=tk.TOP)
        
 #Frame where is showed the image
class ImagePage(tk.Frame):
    
    def __init__(self, parent, controller,filename):
        tk.Frame.__init__(self, parent)
        self.controller = controller


        w = self.winfo_screenwidth() 
        h = self.winfo_screenheight()
        x = w/2 - 500/2
        y = h/2 - 500/2
        controller.geometry("%dx%d+%d+%d" %  (w-5,h-50,x, y))
        
        label = ttk.Label(self,text="Image Page", font=LARGE_FONT)
        label.pack(side=tk.TOP, fill="x", pady=10)

        fig = plt.Figure(figsize=(5,5), dpi=100)            
        subPlot = fig.add_subplot(121)
        subPlot.imshow(Lg.data, cmap=plt.cm.gray)

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

        if(filename.find(".dcm")!=-1):
            T = Text(self, width=50)
            T.insert(END, Lg.consultarInformacion())        
            T.config(state=DISABLED,font=("Verdana",9))
            T.pack(side=tk.RIGHT)
                        
        buttonBack = ttk.Button(self, text="Go to the start page", command=lambda: controller.show_frame("StartPage"))        
        buttonHist = ttk.Button(self, text="Make Histogram", command=lambda:mostrarHistograma(fig,canvas))
        buttonFiltros = ttk.Button(self, text="Apply Filter",command=lambda:aplicarFiltros(fig,canvas,cb.get(),size.get()))          
        buttonKMeans = ttk.Button(self, text="Apply k-means",command=lambda:aplicarKMeans(fig,canvas,centroidsNum.get()))        

        
        buttonBack.pack(side=tk.LEFT,padx=15)
        buttonHist.pack(side=tk.LEFT,padx=10)         
        cb.pack(side=tk.LEFT,padx=5)
        size.pack(side=tk.LEFT,padx=5)
        buttonFiltros.pack(side=tk.LEFT,padx=5)
        centroidsNum.pack(side=tk.LEFT,padx=10)      
        buttonKMeans.pack(side=tk.LEFT,padx=10)
    

    """
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
        
"""

#Start the GUI
if __name__ == "__main__":
    app = ImageProgram()
    app.mainloop()