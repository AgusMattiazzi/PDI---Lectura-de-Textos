# Librerias por defecto
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Funciones extra
import Funciones as func

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
    Alto,Ancho = Img_BW.shape[:2]      # Obtengo las dimensiones

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


# Funcion label2rgb (equivalente a label2rgb de MATLAB)
def label2rgb(labels,N_label,color_fondo = (0,0,0),colormap = 2):
    # Mascara logica con los pixeles correspondientes al fondo
    Fondo = labels == 0

    # Convierte la matriz etiqueta a RGB
    labels = np.uint8( (255*labels)/N_label )
    Img_color = cv2.applyColorMap(labels, colormap)

    # Usa la mascara Fondo para cambiar el color de fondo
    Img_color[Fondo] = color_fondo

    return Img_color

# Funcion de rotacion (opencv no tiene funciones para rotar a cualquier angulo)
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# Funcion Deteccion_ang
def Deteccion_ang(Imagen_BW):
    Contours, Jerarquia = cv2.findContours(Imagen_BW,
                        cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # Con esta funcion se obtienen los contornos a partir de los
    # bordes obtenidos por Canny
    Suma_ang = 0;       cont = 0

    for cnt in Contours:
        Elipse = cv2.fitEllipse(cnt)
        print(Elipse,'\n')
        # # center, axis_length and orientation of ellipse
        (Centro,Ejes,Orientacion) = Elipse
        Suma_ang = Orientacion + Suma_ang   # Sumatoria
        cont += 1                           # Contador

    print("Angulo fitEllipse:", Suma_ang/cont)
    return ( 90 - Suma_ang/cont )

# Funcion Deteccion_texto
def Deteccion_texto(Imagen_BW, Ancho_prom, Alto_prom, proporcion):
    Imagen_BW = imfill(Imagen_BW)
    
    Ancho_kernel = int( Ancho_prom*proporcion )
    Alto_kernel = int( Alto_prom*proporcion )

    kernel = np.ones((Ancho_kernel,Alto_kernel),np.uint8)    # square (5,5)
    Deteccion = cv2.dilate(Imagen_BW, kernel, iterations = 1)
    Deteccion = imfill(Deteccion)
    # El tercer argumento es opcional y por defecto es 1

    Angulo = Deteccion_ang(Deteccion)
    print("Angulo:", Angulo)

    Output = cv2.connectedComponentsWithStats(Deteccion, 8, cv2.CV_32S)

    # Resultados
    num_labels =    Output[0]   # Cantidad de elementos
    labels =        Output[1]   # Matriz con etiquetas

    # Coloreamos los elementos
    Img_color = label2rgb(labels,num_labels)
    return Img_color


## --------------------------------------------------------------------- #














## --------------------- Comandos basicos de Python -------------------- #
# Abrir Imagen
Img = cv2.imread('foto_4.jpg')

# Conversion a escala de grises
Img_gris = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY)

# Umbralado
# ret,Img_BW1 = cv2.threshold(Img_gris,120,255,cv2.THRESH_BINARY)
ret,Img_BW1 = cv2.threshold(Img_gris,25,255,cv2.THRESH_OTSU)
Img_BW1 = cv2.bitwise_not(Img_BW1)  # Negativo

# Limpiar bordes
Img_BW = imclearborder(Img_BW1)

# Mostrar Imagen
# imshow(Img_BW, title = 'Bordes Limpios')

## --------------------------------------------------------------------- #

## ---------------------------- Dilatacion ----------------------------- #
# Para dilatar, hay que crear una estructura llamada kernel
kernel = np.ones((3,3),np.uint8)    # square (5,5)
Img_BW = cv2.dilate(Img_BW, kernel, iterations = 1)
# El tercer argumento es opcional y por defecto es 1

# imshow(Img_BW, title = 'Canny + Dilatacion')

## --------------------------------------------------------------------- #

## ---------------------- Componentes conectadas ----------------------- #
# img = cv2.imread('Letras y Figuras.tif', cv2.IMREAD_GRAYSCALE)
Output = cv2.connectedComponentsWithStats(Img_BW, 8, cv2.CV_32S)
# https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f

# Resultados
num_labels =    Output[0]   # Cantidad de elementos
labels =        Output[1]   # Matriz con etiquetas
stats =         Output[2]   # Matriz de stats
centroids =     Output[3]   # Centroides de elementos

# Coloreamos los elementos
Img_color = label2rgb(labels,num_labels)

Suma_Alto = 0;    Suma_Ancho = 0

# Agregamos Bounding Box
for i in range(1,num_labels):
    Bbox = stats[i,]
    # print (Bbox)
    # Bbox[0] : Coordenada x punto superior izquierdo
    # Bbox[1] : Coordenada y punto superior izquierdo
    # Bbox[2] : Ancho
    # Bbox[3] : Alto

    cv2.rectangle(Img_color, (Bbox[0], Bbox[1]), (Bbox[0]+Bbox[2],
        Bbox[1]+Bbox[3]), (255,255,255), 2)
    
    Suma_Ancho = Suma_Ancho + Bbox[2]
    Suma_Alto = Suma_Alto + Bbox[3]

# Bbox = stats[45,]
# print (Bbox)
# Crop = Img_color2[ Bbox[1]:Bbox[1]+Bbox[3], Bbox[0]:Bbox[0]+Bbox[2] ]
# imshow(Crop, title = 'Imagen Recortada')

Ancho_prom = Suma_Ancho/num_labels
Alto_prom = Suma_Alto/num_labels

print(Ancho_prom, Alto_prom)    # Valor preliminar

## Esto se puede modificar para detectar los puntos, CONSERVAR
# Suma_Alto = 0;    Suma_Ancho = 0
# # Segundo ciclo
# for i in range(1,num_labels):
#     Bbox = stats[i,]
#     if (Bbox[2] < 2*Ancho_prom and Bbox[2] > 0.5*Ancho_prom):
#         if (Bbox[3] < 2*Alto_prom and Bbox[3] > 0.5*Alto_prom):
#             # print (Bbox)
#             Suma_Ancho = Suma_Ancho + Bbox[2] 
#             Suma_Alto = Suma_Alto + Bbox[3]
#     # Si el ancho o alto estan muy alejados del promedio, se ignoran

# Ancho_prom = Suma_Ancho/num_labels
# Alto_prom = Suma_Alto/num_labels

# print(Ancho_prom, Alto_prom)    # Valor final

imshow(Img_color, title = 'Matriz Etiqueta RGB')

## ------------------------ Deteccion de Texto ------------------------- #

dil_palabra = 0.3;  dil_parrafo = 4

Img_palabra = Deteccion_texto(Img_BW, Ancho_prom, Alto_prom, dil_palabra)
imshow(Img_palabra, title = 'Palabras detectadas')# 

Img_parrafo = Deteccion_texto(Img_BW, Ancho_prom, Alto_prom, dil_parrafo)
imshow(Img_parrafo, title = 'Parrafos detectados')

# # Funcion Deteccion_texto
# def Deteccion_texto(Imagen_BW, Ancho_prom, Alto_prom, proporcion):

proporcion = dil_parrafo

Imagen_BW = imfill(Img_BW)

Ancho_kernel = int( Ancho_prom*proporcion )
Alto_kernel = int( Alto_prom*proporcion )

kernel = np.ones((Ancho_kernel,Alto_kernel),np.uint8)    # square (5,5)
Deteccion = cv2.dilate(Imagen_BW, kernel, iterations = 1)
Deteccion = imfill(Deteccion)
# El tercer argumento es opcional y por defecto es 1

Angulo = Deteccion_ang(Deteccion)
print("Angulo:", Angulo)

Output = cv2.connectedComponentsWithStats(Deteccion, 8, cv2.CV_32S)

# Resultados
num_labels =    Output[0]   # Cantidad de elementos
labels =        Output[1]   # Matriz con etiquetas

# Coloreamos los elementos
Img_color = label2rgb(labels,num_labels)
imshow(Img_color)
# return Img_color

# ## ----------------------------- Rotacion ------------------------------ #
# Ang_rotacion = -Angulo
# Rotada = rotate_image(Img_BW, Ang_rotacion)
# imshow(Rotada, title = 'Imagen Rotada')

## ------------------------ Aproximar contorno ------------------------- #
Img_gris2 = cv2.cvtColor(Img_color,cv2.COLOR_BGR2GRAY)

# Umbralado
# ret,Img_BW1 = cv2.threshold(Img_gris,120,255,cv2.THRESH_BINARY)
ret,Area_Texto = cv2.threshold(Img_gris2,25,255,cv2.THRESH_OTSU)
Ancho,Alto = Area_Texto.shape

Contours, Hierarchy = cv2.findContours(Area_Texto,cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

for cnt in Contours:
    area = cv2.contourArea(cnt)
    # Calcula el area de cada contorno
    if area > 0.2*Ancho*Alto:
        # Si el area elegida abarca un 10% de la imagen
        # Se obtienen las esquinas del poligono aproximante
        per = cv2.arcLength(cnt,True)
        Puntos = cv2.approxPolyDP(cnt,0.01*per,True,)
        cv2.polylines(Img_color, [Puntos], True, (255,0,0), 1 )

        # Bounding box
        Bbox = cv2.boundingRect(cnt)
        cv2.rectangle(Img_color, (Bbox[0], Bbox[1]), (Bbox[0]+Bbox[2],
        Bbox[1]+Bbox[3]), (0,255,0), 1)

        # Minimo Rectangulo contenedor de la figura
        Rect = cv2.minAreaRect(cnt)
        points = cv2.boxPoints(Rect)
        Box = np.int32(points)
        # Dibujar Rectangulo de minima area que abarca al texto
        cv2.drawContours(Img_color,[Box],0,(0,0,255),1)

# Imprimir resultados para la caja, funca relativamente bien, faltan ajustes
print(Box);     print(Puntos);      print(Bbox)
imshow(Img_color, title = 'Imagen Marcada')

## --------------------------------------------------------------------- #
