import cv2
import numpy as np
import matplotlib.pyplot as plt

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

    # Esto puede salir tan, pero tan mal
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
def label2rgb(labels,n_labels,color_fondo = (0,0,0),colormap = 2):
    # Mascara logica con los pixeles correspondientes al fondo
    Fondo = labels == 0

    # Convierte la matriz etiqueta a RGB
    labels = np.uint8( (255*labels)/n_labels )
    Img_color = cv2.applyColorMap(labels, cv2.COLORMAP_JET)

    # Usa la mascara Fondo para cambiar el color de fondo
    Img_color[Fondo] = color_fondo

    return Img_color


def Get_Fourier_desc(Img_BW,M,N):
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

    # if type(Img_BW) != type(True):
    #     raise NameError('The input image must be of type "logical"')

    if (M % 1) > 0 | (N % 1) > 0 | M < 0 | N < 0:
        # Con % se obtiene el resto de una division
        raise NameError('Input arguments M and N must be an integer greater or equal to zero')
    
    # contour, hierarchy = cv2.findContours(Img_BW,cv2.RETR_EXTERNAL,
    #     cv2.CHAIN_APPROX_NONE)

    # if len(contour) > 1:
    #     raise NameError('Image contains more than one object')



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

## --------------------------------------------------------------------- #
