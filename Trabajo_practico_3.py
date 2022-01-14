import cv2
import numpy as np
import matplotlib.pyplot as plt
# # Archivo donde voy a poner funciones homologas a matlab
# import matlab

# Gracias a Murtaza's Workshop - Robotics and AI, sin su ayuda este
# archivo quiza no existiria

## ----------------------------- Funciones ----------------------------- #

# Funcion imshow
def imshow(Img,title = None):
    plt.figure()
    if len(Img.shape) == 3:
        plt.imshow(Img)
    else:
        plt.imshow(Img, cmap='gray')

    plt.title(title)
    plt.xticks([]), plt.yticks([])
    plt.show()
    # plt.show() espera a que la imagen se cierre para proseguir


# Funcion imclearborder (equivalente a imclearborder de MATLAB)
def imclearborder(Img_BW):
    # Es importante insertar una imagen blanco y negro, pero voy a
    # tratar de adaptarlo asi no me complico la vida
    Blanco = np.max(Img_BW)

    Img_clear = Img_BW.copy()       # Copio la imagen
    Alto,Ancho = Img.shape[:2]      # Obtengo las dimensiones

    # El tamaño debe ser dos pixeles mayor al de la imagen original
    mask = np.zeros((Alto+2, Ancho+2), np.uint8)

    # Recorro los extremos de la imagen, aplicando floodfill (llenado
    # por difusion) siempre que encuentro un punto blanco

    # IMPORTANTE: El formato del punto semilla en floodfill (tercer 
    # argumento) va en el formato (x,y) y no en (fila,columna)
    for x in range(Ancho - 1):
        # Extremo superior
        if Img_clear[0,x] == Blanco:
            # Llena por desde el punto (0,x)
            cv2.floodFill(Img_clear, mask, (x,0), 0)

        # Extremo inferior
        if Img_clear[Alto-1,x] == Blanco:
            # Llena desde el punto (Alto,x)
            cv2.floodFill(Img_clear, mask, (0,Alto-1), 0)

    for y in range(Alto - 1):
        # Extremo izquierdo
        if Img_clear[y,0] == Blanco:
            # Llena desde el punto (y,0)
            cv2.floodFill(Img_clear, mask, (0,y), 0)

        # Extremo derecho
        if Img_clear[y,Ancho-1] == Blanco:
            # Llena desde el punto (y,Ancho)
            cv2.floodFill(Img_clear, mask, (Ancho-1,y), 0)

    return Img_clear


# Funcion imfill (equivalente a imfill de MATLAB)
def imfill(Img):
# Gracias a Satya Mallick por el script
# https://learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

    # Copia la imagen
    Img_floodfill = Img.copy()

    # Mascara usada para el llenado por difusion
    # El tamaño debe ser dos pixeles mayor al de la imagen original
    Ancho, Alto = Img.shape[:2]
    mask = np.zeros((Ancho+2, Alto+2), np.uint8)

    # Llena por difusion (floodfill) desde el punto (0, 0)
    cv2.floodFill(Img_floodfill, mask, (0,0), 255)

    # Imagen llenada invertida
    Img_floodfill_inv = cv2.bitwise_not(Img_floodfill)

    # La imagen final es la union entre la imagen original y la
    # imagen llenada invertida
    Salida = Img | Img_floodfill_inv
    return Salida


# Funcion label2rgb
def label2rgb(labels,N_label,color_fondo = (0,0,0),colormap = 2):
    # Mascara logica con los pixeles correspondientes al fondo
    Fondo = labels == 0

    # Convierte la matriz etiqueta a RGB
    labels = np.uint8( (255*labels)/N_label )
    Img_color = cv2.applyColorMap(labels, colormap)

    # Usa la mascara Fondo para cambiar el color de fondo
    Img_color[Fondo] = color_fondo

    return Img_color


def Get_Fourier_desc(Img_BW,m,n):
# GFD is a function to calculate the generic fourier descriptors of an
# object in an binary image
#     FD = GFD(BW,M,N) gets an NxX binary image BW as input with a single
#     object centered to the image center. M denotes the radial frequency, N
#     the angular frequency. Returns and vector FD, containing the fourier
#     descriptors. The length of the output vector FD is (m*n+n+1). FD(end)
#     contains the FD(0,0), the rest follow the order of FD(rad*n+ang) with
#     0 <= rad <= m and 0 <= ang <= n. 


#     Function that centers an object to the image center with its Centroid:
#     http://www.mathworks.com/matlabcentral/fileexchange/52560-centerobject-bw

#     This is an implementation of:
#     "Shape-based image retrieval using generic Fourier descriptors", D.Zhang,
#     G. Lu, 2002

#     by Frederik Kratzert 24. Aug 2015
#     contact f.kratzert(at)gmail.com

#     Implementado a Python por Agustin Mattiazzi, si esto sirve, Dios tambien habra 
#     tenido algo que ver

    # if type(Img_BW) != type(True):
    #     raise NameError('The input image must be of type "logical"')

    if (m % 1) > 0 | (n % 1) > 0 | m < 0 | n < 0:
        # Con % se obtiene el resto de una division
        raise NameError('Input arguments M and N must be an integer greater or equal to zero')
    
    Output = cv2.connectedComponentsWithStats(Img_BW, 8, cv2.CV_32S)

    # Resultados
    num_labels =    Output[0]   # Cantidad de elementos
    labels =        Output[1]   # Matriz con etiquetas
    stats =         Output[2]   # Matriz de stats
    centroids =     Output[3]   # Centroides de elementos

    if num_labels > 2:  # El fondo cuenta como un objeto
        raise NameError('Image contains more than one object')
    
    Size = np.asarray(Img_BW.shape)
    # Img_BW.shape es una tupla. np.asarray la convierte en array

    Centro = Size/2
    Centroide = centroids[1]
    # Diferencia = Centro - Centroide
    # if sum(abs(Diferencia) > 0.5) > 0:
    #     raise NameError('Object is not centered to the image center. See "help gfd"')

    contour, hierarchy = cv2.findContours(Img_BW,cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    
    # Veo que onda con el extrema
    # Img_esq = Img_BW.copy()
    # Img_esq = cv2.cvtColor(Img_esq, cv2.COLOR_GRAY2RGB)
    # Img_esq = cv2.drawContours(Img_esq,contour,-1,(0,0,255),1)
    # imshow(Img_esq, title = 'Extrema')

    # Trato de calcular el radio maximo a partir de los puntos extremos
    contour_array = contour[0][:,0,:]
    # print(contour_array.shape)
    # print(contour_array,contour_array.dtype)
    x = (contour_array[:,0]-Centroide[0])
    y = (contour_array[:,1]-Centroide[1])
    # print(x.shape, y.shape)
    Radio_max = np.sqrt( x**2 + y**2)
    Radio_max = np.max(Radio_max)
    # print(Radio_max)

    N = Size[1]
    x = np.linspace(-N/2,N/2,N)
    y = x
    X,Y = np.meshgrid(x,y)

    Radio = np.sqrt(X**2+Y**2)/Radio_max

    pi = 3.14159265359  # Definicion de pi
    Theta = np.arctan2(Y,X)
    # print(Radio.shape)
    
    # Aux = Theta < 0
    Theta[Theta < 0] = Theta[Theta < 0]  + 2*pi
    # print("Theta: \n", Theta.shape)

    FR = np.zeros( (m+1,n+1) )
    FI = np.zeros( (m+1,n+1) )
    FD = np.zeros(  ((m+1)*(n+1),1) )
    
    i = 0
    for Rad in range(0,m):
    # loop over all angular frequencies
        for Ang in range(0,n):
            # calculate FR and FI for (rad,ang)

            R_temp = Img_BW*np.cos(2*pi*Radio*Rad + Ang*Theta)
            I_temp = -1*Img_BW*np.sin(2*pi*Radio*Rad + Ang*Theta)
            FR[Rad,Ang] = np.sum(R_temp[:])
            FI[Rad,Ang] = np.sum(I_temp[:])

            # calculate FD, where FD(end)=FD(0,0) --> rad == 0 & ang == 0
            FD_00 = np.sqrt( FR[0,0]**2 + FR[0,0]**2 )
            if Rad == 0 & Ang == 0:
                # normalized by circle area
                Area_circ = pi*Radio_max**2
                FD[i] = FD_00 / Area_circ

            else:
                # normalized by |FD(0,0)|
                FD[i] = np.sqrt( ( FR[Rad,Ang]**2 + FI[Rad,Ang]**2) ) / FD_00

            i = i+1

    return FD

# Funcion Get_desc_Fourier
# Obtiene los descriptores de Fourier de una figura individual
def Get_desc_Fourier(Img_BW,orden):
    # Gracias a timfeirg por la ayuda
    # https://github.com/timfeirg/Fourier-Descriptors
    contour, hierarchy = cv2.findContours(Img_BW,cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE)
    contour_array = contour[0][:, 0, :]
    # Guarda los puntos del contorno en un array (no entiendo bien la 
    # sintaxis), que carajo es ese 0 en el medio?

    # Crea un array de numeros complejos donde se almacenaran los puntos
    # del contorno expresados como numeros complejos
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]

    # Calcula la serie de fourier del contorno
    Desc = np.fft.fft(contour_complex)
    print(Desc.shape)

    # Centra el espectro? transformado
    Desc = np.fft.fftshift(Desc)

    # Trunca los descriptores desde el centro para mantener simetria
    centro = len(Desc) / 2
    Izq = np.floor( (centro-orden)/2 )
    Der = np.ceil( (centro+orden)/2 )
    Desc = Desc[ int(Izq) : int(Der) ]

    # Vuelve al espectro original (no centrado)
    Desc = np.fft.ifftshift(Desc)
    return Desc


# Funcion centerobject
def centerobject(Img_BW):
    if len(Img_BW) < 9: # Puse un numero chico al azar
        print("Image is too little")
        return 0,False  # Operacion fallida

    if len(Img_BW.shape) == 3:
        print("Color image is not allowed")
        return 0,False  # Operacion fallida

    Output = cv2.connectedComponentsWithStats(Img_BW, 8, cv2.CV_32S)

    # Resultados
    num_labels =    Output[0]   # Cantidad de elementos
    centroids =     Output[3]   # Centroides de elementos

    if num_labels != 2:  # El fondo cuenta como un objeto
        print("Image does not contain one object")
        return 0,False  # Operacion fallida
    
    Size_BW = np.asarray(Img_BW.shape)
    # Img_BW.shape es una tupla, np.asarray la convierte en array

    Centroide = np.uint8( centroids[1] )
    Diff = np.round( abs( Size_BW/2 - Centroide ) )
    Diff = np.uint8( Diff )
    # Se usa un valor absoluto porque siempre va a ser necesario agregar
    # filas y columnas a la imagen original

    R = np.max(Size_BW + Diff )
    Salida = np.zeros( (R+13,R+13) )    # Hay que dar margen
    
    Size_O = np.asarray(Salida.shape)
    Diff = np.uint8( Size_O/2 - Centroide )    # Esto deberia dar positivo

    for x in range(0, Size_BW[1] - 1):  # Columnas, eje x
        for y in range(0, Size_BW[0] - 1):  # Filas, eje y
            Salida[ y+Diff[1], x+Diff[0] ] = Img_BW[ y,x ]

    return np.uint8(Salida),True

## --------------------------------------------------------------------- #

## --------------------- Comandos basicos de Python -------------------- #
# Abrir Imagen
Img = cv2.imread('foto_1.jpg')

# Conversion a escala de grises
Img_gris = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)

# Umbralado
# ret,Img_BW1 = cv2.threshold(Img_gris,120,255,cv2.THRESH_BINARY)
ret,Img_BW1 = cv2.threshold(Img_gris,30,255,cv2.THRESH_OTSU)
Img_BW1 = cv2.bitwise_not(Img_BW1)  # Negativo

# Limpiar bordes
Img_BW2 = imclearborder(Img_BW1)

# Mostrar Imagen
imshow(Img_BW2, title = 'Bordes Limpios')

## ---------------------- Componentes conectadas ----------------------- #
# img = cv2.imread('Letras y Figuras.tif', cv2.IMREAD_GRAYSCALE)
Output = cv2.connectedComponentsWithStats(Img_BW2, 8, cv2.CV_32S)
# https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f

# Resultados
num_labels =    Output[0]   # Cantidad de elementos
labels =        Output[1]   # Matriz con etiquetas
stats =         Output[2]   # Matriz de stats
centroids =     Output[3]   # Centroides de elementos

# # Coloreamos los elementos
# Img_color = label2rgb(labels,num_labels)

# # Agregamos Bounding Box
# for i in range(1,num_labels):
#     Bbox = stats[i,]
#     cv2.rectangle(Img_color, (Bbox[0], Bbox[1]), (Bbox[0]+Bbox[2],
#         Bbox[1]+Bbox[3]), (255,255,255), 2)

# imshow(Img_color, title = 'Matriz Etiqueta RGB')

## ---------------------- Descriptores de Fourier ---------------------- #
D_Fourier = []
for i in range(1,num_labels):   # El elemento 0 es el fondo

    # Recorta cada una de las letras
    Bbox = stats[i,]
    # Aplica una mascara para evitar la superposicion de letras
    Mask = labels == i
    Crop = Mask[ Bbox[1]:Bbox[1]+Bbox[3] , Bbox[0]:Bbox[0]+Bbox[2] ]
    Crop = 255*np.uint8(Crop)   # Convierte de bool a gris

    # imshow(Crop, title = 'Recortada')
    # print(Crop.shape, Crop.dtype)
    Crop,Flag = centerobject(Crop)
    if Flag == True:
        Output = cv2.connectedComponentsWithStats(Crop, 8, cv2.CV_32S)
        # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f

        # Resultados
        centroids =     Output[3]   # Centroides de elementos

        Crop_color = cv2.cvtColor(Crop,cv2.COLOR_GRAY2RGB)
        Centro = np.uint8( np.asarray(Crop.shape)/2 )
        Centroide = np.uint8( centroids[1] )

        # print(Centro,Centroide)
        # imshow(Crop_color, title = 'Centrada')
        # print(i, i.dtype)
        if i == 1:
            D_Fourier = Get_Fourier_desc(Crop,3,3)
        else:
            D_Fourier = np.hstack((D_Fourier,
                Get_Fourier_desc(Crop,3,3)))

## ------------------------------ K-means ------------------------------ #
# Para aplicar el K-means de open-cv se debe tener el conjunto de vectores
# a clasificar, el criterio (cuando frena), el maximo de iteraciones, el
# epsilon (la diferencia maxima entre una iteracion y otra), la cantidad
# de categorias y la cantidad de repeticiones

criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
# Con este criterio se detiene el algoritmo si se llega a cumplir la
# precision deseada, o bien, si se llego al numero maximo de iteraciones,
# lo que llegue primero

criteria = (criteria_type, 10, 1.0)
# El maximo de iteraciones es 10 y el epsilon es 1.0

flag = cv2.KMEANS_RANDOM_CENTERS
# Determina que los centroides iniciales sean aleatorios

D_Fourier = np.float32( np.transpose(D_Fourier) )
# Ojo aca, el K-means de open cv agarra los datos por fila, si los 
# vectores de datos son columnas y no filas, asegurate de hacer la 
# transpuesta, ademas de eso agarra solo arrays de float32, si sera
# caprichoso

ret, Letras, center = cv2.kmeans(D_Fourier, 27, None, criteria, 10, flag)
print( Letras )

L_Clasificado = labels.copy()
# Separo clusters
for i in Letras:
    # Busca las letras agrupadas en un conjunto
    Mask = L_Clasificado == Letras[i]
    # Reasinga su valor al valor obtenido del K-means
    L_Clasificado[Mask] = -i


L_Clasificado = np.abs( L_Clasificado )
Img_Clasificado = label2rgb(L_Clasificado,27)
imshow(Img_Clasificado, title = 'Resultado')





# ---------------------------------------------------------------------- #