import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
print(np.shape(img))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")
print("Type de l'image :",img.dtype)

#Début du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
print("Shape de la matrice de autocorrelation: ",np.shape(Theta))
# Mettre ici le calcul de la fonction d'intérêt de Harris

#----------

#we have to make sure all results are in the [0,255] range, so
def adjust_values(f_result):

    f_result_norm = 255*(f_result - f_result.min()) / (f_result.max() - f_result.min())
    return f_result_norm

#On commence en calculant la matrice de autocorrelation


def calc_auto(in_matrix,shape,alpha):
    len_window = shape[1] - 1
    w_space = len_window//2
    size=len(in_matrix)
    size_y=np.shape(in_matrix)[1]
    f_result=np.zeros((size,size_y))
    kernel_dx=np.array([[0,0,0],[-1,0,1],[0,0,0]])
    kernel_dy=np.array([[0,-1,0],[0,0,0],[0,1,0]])

    matrix_dx=cv2.filter2D(in_matrix,-1,kernel_dx)
    matrix_dx = adjust_values(matrix_dx)
    matrix_dy=cv2.filter2D(in_matrix,-1,kernel_dy)
    matrix_dy=adjust_values(matrix_dy)
    #we also pad the original image's derivatives with 0s to make the loop
    matrix_copy_dx = cv2.copyMakeBorder(matrix_dx, w_space, w_space, w_space, w_space,cv2.BORDER_CONSTANT)
    matrix_copy_dy = cv2.copyMakeBorder(matrix_dy, w_space, w_space, w_space, w_space,cv2.BORDER_CONSTANT)
    #onc commence le boucle:
    for i in range(1,size + 1):
        for j in range(1,size_y+1):
            #on cree deux matrices pour calculer l autocorrelation
            window_dx = np.array([[matrix_copy_dx[k][p] for p in range(j-w_space,j+1+w_space)]
                                   for k in range(i-w_space,i+1+w_space)])#wrong
            window_dy = np.array([[matrix_copy_dy[k][p] for p in range(j-w_space,j+1+w_space)]
                                   for k in range(i-w_space,i+1+w_space)])
            auto_corr = np.zeros((2,2))#la matrice qu'on sauvegardera ls valeurs
            auto_corr[0][0]=np.sum(window_dx**2)
            auto_corr[1][1]=np.sum(window_dy**2)
            auto_corr[0][1]=np.sum(np.multiply(window_dx,window_dy))
            auto_corr[1][0]=np.sum(np.multiply(window_dx,window_dy))
            f_result[i-1][j-1] =np.linalg.det(auto_corr) - alpha*(np.trace(auto_corr)**2)
    return f_result


Theta=calc_auto(Theta,(3,3),0.05)
print('Shape of Theta after operation: ',np.shape(Theta))

# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)#padding with balck border
seuil_relatif = 0.01
d_maxloc = 3
se = np.ones((d_maxloc,d_maxloc),np.uint8)
Theta_dil = cv2.dilate(Theta,se)
#Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0
#On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Mon calcul des points de Harris :",time,"s")
print("Nombre de cycles par pixel :",(t2 - t1)/(h*w),"cpp")

plt.subplot(131)
plt.imshow(img,cmap = 'gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta,cmap = 'gray')
plt.title('Fonction de Harris')

se_croix = np.uint8([[1, 0, 0, 0, 1],
[0, 1, 0, 1, 0],[0, 0, 1, 0, 0],
[0, 1, 0, 1, 0],[1, 0, 0, 0, 1]])
print('Shape of max_loc: ',np.shape(Theta_maxloc))
Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
print('Shape of ml_dil: ',np.shape(Theta_ml_dil))
#Relecture image pour affichage couleur
Img_pts=cv2.imread('Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
(h,w,c) = Img_pts.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes x",c,"canaux")
print("Type de l'image :",Img_pts.dtype)
#On affiche les points (croix) en rouge
Img_pts[Theta_ml_dil > 0] = [255,0,0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points de Harris')

window_sizes = np.array([2*i+1 for i in range(1,4)],dtype=np.int16)

alpha_values = np.linspace(0.04,0.06,3)

#different plots for window size
fig, axs = plt.subplots(3,2)
fig.suptitle('Résultats pour différentes fenêtres')
img=np.float64(cv2.imread('Image_Pairs/Graffiti0.png',cv2.IMREAD_GRAYSCALE))
Theta_loop = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for size in range(len(window_sizes)):
    Theta=calc_auto(Theta_loop,(window_sizes[size],window_sizes[size]),0.05)
    Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
    Theta_dil = cv2.dilate(Theta,se)
    Theta_maxloc[Theta < Theta_dil] = 0.0
    Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
    Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
    Img_pts=cv2.imread('Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
    Img_pts[Theta_ml_dil > 0] = [255,0,0]
    axs[size][0].imshow(Theta,cmap = 'gray')
    axs[size][0].tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
    axs[size][0].set_ylabel('(' + 'i'*(size+1)+')')
    if (size==0):
        axs[size][0].set_title('Fonction de Harris')
        axs[size][1].set_title('Points de Harris')
    axs[size][1].imshow(Img_pts)
    axs[size][1].tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
    
fig1, axs1 = plt.subplots(3,2)
fig1.suptitle(r'Résultats pour différentes valeurs de $\alpha$')
for i in range(3):
    Theta=calc_auto(Theta_loop,(3,3),alpha_values[i])
    Theta_maxloc = cv2.copyMakeBorder(Theta,0,0,0,0,cv2.BORDER_REPLICATE)
    Theta_dil = cv2.dilate(Theta,se)
    Theta_maxloc[Theta < Theta_dil] = 0.0
    Theta_maxloc[Theta < seuil_relatif*Theta.max()] = 0.0
    Theta_ml_dil = cv2.dilate(Theta_maxloc,se_croix)
    Img_pts=cv2.imread('Image_Pairs/Graffiti0.png',cv2.IMREAD_COLOR)
    Img_pts[Theta_ml_dil > 0] = [255,0,0]
    axs1[i][0].imshow(Theta,cmap = 'gray')
    axs1[i][0].tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
    axs1[i][0].set_ylabel('(' + 'i'*(i+1)+')')
    if (i==0):
        axs1[i][0].set_title('Fonction de Harris')
        axs1[i][1].set_title('Points de Harris')
    axs1[i][1].imshow(Img_pts)
    axs1[i][1].tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)

plt.show()
