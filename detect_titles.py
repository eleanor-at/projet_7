import cv2
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt


def detect_titles(img_number):

     #création du path
    number_string=f'{img_number}'
    char=len(number_string)
    for i in range(4-char):
        number_string='0'+number_string
    path='hisoma/images/image-'+number_string+'.png'

    #lecture de l'image
    image=cv2.imread(path)
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (thresh, img) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #rognage
    img=img[120:-400,100:-100]
    height,width=img.shape

    #detection de la zone de texte
    kernel_text=np.ones((10,1),np.uint8)
    text=cv2.erode(img, kernel_text, iterations=3)
    hist_text=np.sum(text//255,axis=0)
    # plt.plot(hist_text)
    # plt.show()
    text_begin=False
    blanc=True
    for c in range(width):
        if blanc and hist_text[c]<0.6*height:
            if not text_begin:
                text_start=c-10
                text_begin=True
            blanc=False
        if not blanc and hist_text[c]>0.6*height:
            text_end=c+10
            blanc=True

    #morph de l'image
    kernelv = np.ones((2,10),np.uint8)
    erodeh = cv2.erode(img, kernelv, iterations=3)
    dilatev=cv2.dilate(erodeh,kernelv,iterations=3)

    #somme des pixels selon l'axe horizontal
    histw= np.sum(dilatev//255, axis=1)
    mw=max(histw)   
    histb=mw-histw

    #lissage
    conv=np.ones((10))
    histb=np.convolve(histb,conv)
    
    #détection des lignes de texte
    peaksb=scipy.signal.find_peaks(histb, prominence=100, width=len(histb)*1//500, distance=10)[0]
    #calcul écart entre les lignes
    mb=max(histb)
    #plt.plot(histb, color='black')
    #plt.vlines(peaksb,0,mb,color='g',lw=0.5)

    n=len(peaksb)
    linestep=[49]
    for i in range(n-2):
        linestep.append(peaksb[i+2]-peaksb[i])
        #plt.vlines(peaksb[i],0,linestep[i]*mb/50,color='r')

    #plt.show()

    #identification des titres
    titles=scipy.signal.find_peaks(linestep, prominence=0.08*max(linestep))[0]
    titles=np.concatenate((np.array([0]),titles))
    title_peaks=peaksb[titles]

    #délimitation des titres avec des lignes horizontales
    left_ips_h=scipy.signal.peak_widths(histb,title_peaks,rel_height=0.9)[2]-10
    right_ips_h=scipy.signal.peak_widths(histb,title_peaks,rel_height=0.9)[3]
    nb_titles=len(titles)
    borders_h=np.zeros(shape=(nb_titles,2),dtype=int)
    for i in range (nb_titles):
        borders_h[i][0]=int(left_ips_h[i])
        borders_h[i][1]=int(right_ips_h[i])


    #création de boites
    kernel1=np.ones((5,2))
    kernel2=np.ones((15,5))
    kernel3=np.ones((10,6))
    kernel4=np.ones((3,3))
    borders_v=np.zeros(shape=(nb_titles,2),dtype=int)
    versio_or_vulgata=np.zeros((nb_titles),dtype=int)
    notae=np.zeros((nb_titles),dtype=int)
    for i in range(nb_titles):
        left_ip,right_ip=borders_h[i][0],borders_h[i][1]
        title_height=right_ip-left_ip
        title_img=img[left_ip:right_ip,:]
        if i>=1:
            title_img=cv2.dilate(title_img,kernel1)
            title_img=cv2.erode(title_img,kernel2,iterations=3)
        if i==0:
            title_img=cv2.dilate(title_img,kernel4, iterations=1)
            title_img=cv2.erode(title_img,kernel3,iterations=3)
        hist0=np.sum(title_img//255,axis=0)
        Title_begin=False
        Title_end=False
        blanc=True
        if i>=1:
            for c in range(text_start+80,text_end-80):
                if blanc and hist0[c]<0.05*title_height:
                    if not Title_begin:
                        borders_v[i][0]=c-5
                        Title_begin=True
                    blanc=False
                if not blanc and hist0[c]>0.95*title_height:
                    end=c+5
                    blanc=True
                    Title_end=True
            if Title_end:
                borders_v[i][1]=end
            else:
                borders_v[i][1]=text_end
        if i==0:
            for c in range(80, text_end-text_start-80):
                if not Title_begin and hist0[text_end-c]<0.1*title_height:
                    borders_v[i][1]=text_end-c+50
                    Title_begin=True
                if Title_begin and hist0[text_end-c]>0.95*title_height and not Title_end:
                    borders_v[i][0]=text_end-c-5
                    Title_end=True
            # cv2.imshow('titre',title_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


        
        #distinction des titres versio antiqua/vulgata nova
        title_img=title_img[:,borders_v[i][0]:borders_v[i][1]]
        middle=int((borders_v[i][1]-borders_v[i][0])/2)
        middle_zone=np.sum(np.sum(title_img[:,middle-75:middle+75],axis=0),axis=0)
        if middle_zone>1000000:
            versio_or_vulgata[i]=1

    #detection des notae
    for i in range(nb_titles):
        if versio_or_vulgata[i]==1:
            j=i+1
            is_notae=False
            while not is_notae:
                if j>nb_titles-1:
                    break
                if versio_or_vulgata[j]==1:
                    break
                if borders_v[i][0]<borders_v[j][0] and borders_v[i][1]>borders_v[j][1]:
                    notae[j]=1
                    is_notae=True
                else:
                    j+=1
            



    borders=np.concatenate((borders_h,borders_v),axis=1)


    


    
    #tracé des boîtes correspondants aux titres
    pb=False
    if np.sum(versio_or_vulgata)==0:
        #print("'Versio Antiqua' missing")
        pb=True
    elif np.sum(notae)!=np.sum(versio_or_vulgata):
        #print("'Notae' missing")
        pb=True

    #delimitation du paragraphe d'interet et du numero du chapitre
    kernel_para=np.ones((20,3))
    paragraphe=[]
    chapitre=img[borders[0][0]:borders[0][1],borders[0][2]:borders[0][3]]
    if not pb:
        for i in range(nb_titles):
            if versio_or_vulgata[i]==1:
                j=i+1
                while notae[j]==0:
                    j+=1
                para_haut=borders_h[i][0]
                para_bas=borders_h[j][0]
                para_height=para_bas-para_haut
                para_nova_antiqua=img[para_haut:para_bas,text_start+20:text_end-20]
                erode_nova_antiqua=cv2.erode(para_nova_antiqua,kernel_para,iterations=3)
                hist_para=np.sum(erode_nova_antiqua//255, axis=0)
                hist_para=np.convolve(conv,hist_para)
                if len(scipy.signal.find_peaks(hist_para,prominence=6*para_height)[0])==0:
                    pb=True
                else:
                    middle_blank=scipy.signal.find_peaks(hist_para,prominence=6*para_height)[0][0]+20
                    nb_para=len(paragraphe)
                    if img_number%2==0:
                        paragraphe.append(img[para_haut:para_bas,text_start+middle_blank:text_end])
                    else:
                        paragraphe.append(img[para_haut:para_bas,text_start:text_start+middle_blank])
                    
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

    thickness=2
    for i in range(nb_titles):
        title=borders[i]
        if versio_or_vulgata[i]==1:
            line_colour=(0,255,0)
        elif notae[i]==1:    
            line_colour=(255,0,0)
        else:
            line_colour=(0,0,255)
        h1,h2,v1,v2=title[0],title[1],title[2],title[3]
        p1=(v1,h1)
        p2=(v2,h1)
        p3=(v1,h2)
        p4=(v2,h2)
        img = cv2.line(img, p1, p2, line_colour, thickness)
        img = cv2.line(img, p1, p3, line_colour, thickness)
        img = cv2.line(img, p3, p4, line_colour, thickness)
        img = cv2.line(img, p2, p4, line_colour, thickness)

    #cv2.imshow(f'Image no {img_number}', img)
    # if len(chapitre)>0:
    #     cv2.imshow(f'chapitre',chapitre)
    # else:
    #     print('chapter number missing')
    # for i,para in enumerate(paragraphe):
    #     cv2.imshow(f'paragraphe {i}',para)
    if pb:
        return img_number
    else:
        return 0

current_img = 48
max_img = 1157
min_img = 1
page_pb=[]
for nb in range (48,1104):
    print(nb)
    page_pb.append(detect_titles(nb))
print(f'on a le paragraphe pour {int(100*len(page_pb)/(1104-48))}')

# while True:
#     # Detect and show titles on current image
#     detect_titles(current_img)

#     # Wait for key press
#     key = cv2.waitKey(0) & 0xFF

#     # Right arrow key → go to next image
#     if key == ord('d'):
#         if current_img < max_img:
#             current_img += 1
#             cv2.destroyAllWindows()
#         else:
#             print("Already at last image.")

#     # Left arrow key → go to previous image
#     elif key == ord('q'):
#         if current_img > min_img:
#             current_img -= 1
#             cv2.destroyAllWindows()
#         else:
#             print("Already at first image.")

#     # ESC key to exit
#     elif key == 27:
#         cv2.destroyAllWindows()
#         break

    