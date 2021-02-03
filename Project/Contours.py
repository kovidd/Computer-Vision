import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob, os

#== Parameters           
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # In BGR format

def t1(name):
    #-- Read image
    img = cv2.imread(name)
    #kovid
    # plt.imshow(img, cmap = 'gray')
    # plt.show()
    img = cv2.GaussianBlur(img,(1,1),0)
    # img = cv2.medianBlur(img,3)
    # img = cv2.bilateralFilter(img,11,5,5)
    # plt.imshow(img, cmap = 'gray')
    # plt.show()
    
    # edges = cv2.Canny(img,100,200)
    # print('\nCanny Edges')
    # plt.imshow(edges, cmap = 'gray')
    # plt.show()
    
    #histogram equalisation
    print(f'Histrogram')
    equ = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    res = np.hstack((img,equ)) #stacking images side-by-side
    # plt.imshow(res, cmap = 'gray')
    # plt.show()
    
    
    print('Enhanced')
    # plt.imshow(equ, cmap = 'gray')
    # plt.show()
    
    
    print(f'Lowest value in image = {np.min(equ)}')
    print(f'Highest value in image = {np.max(equ)}')
    print('Enhancing')
    norma = equ
    norma[norma > 30] = 255
    norma[norma < 20] = 0
    # plt.imshow(norma, cmap = 'gray')
    # plt.show()
    
    #histogram equalisation
    print(f'Final')
    res1 = np.hstack((img,norma)) #stacking images side-by-side
    # plt.imshow(res1, cmap = 'gray')
    # plt.show()
    
    #histogram equalisation
    print(f'Final')
    equ = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    res2 = np.hstack((equ, norma)) #stacking images side-by-side
    plt.figure(figsize=(10,5))
    plt.imshow(res2, cmap = 'gray')
    plt.show()
    
    # plt.imshow(equ, cmap = 'gray')
    # plt.show()

# t1('1.tif')
t1('t052.tif')
# sequence = ['Sequence 1']
# for subfolder in sequence:
#     os.chdir('Fluo-N2DL-HeLa/'+subfolder+'/')
#     for file in glob.glob('*.tif'):
#     #         print(os.getcwd(), 'File: ',file)
#         print('Sequence:', subfolder,' File:',file)
#         os.getcwd()
#         t1(file)

# img = equ
# # for _ in range(10,100,10):
# #     print(f'Histrogram Alpha={_}')
# #     equ = cv2.normalize(img, None, alpha=_, beta=255, norm_type=cv2.NORM_MINMAX)
# #     res = np.hstack((img,equ)) #stacking images side-by-side
# #     plt.imshow(res, cmap = 'gray')
# #     plt.show()
    
# # for _ in range(255,10,-10):
# #     print(f'Histrogram Beta={_}')
# #     equ = cv2.normalize(img, None, alpha=0, beta=_, norm_type=cv2.NORM_MINMAX)
# #     res = np.hstack((img,equ)) #stacking images side-by-side
# #     plt.imshow(res, cmap = 'gray')
# #     plt.show()

# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# #-- Edge detection 
# edges = equ
# edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
# edges = cv2.dilate(edges, None)
# edges = cv2.erode(edges, None)
# print('\nCanny Edges 1')
# plt.imshow(edges, cmap = 'gray')
# plt.show()

# #kovid
# kernel = np.ones((1,1), np.uint8)
# edges = cv2.dilate(edges, kernel, iterations=1)
# edges = cv2.erode(edges, kernel, iterations=1)

# # print('\nCanny Edges 2')
# plt.imshow(edges, cmap = 'gray')
# plt.show()

# # -- Find contours in edges, sort by area 
# contour_info = []
# contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# for c in contours:
#     contour_info.append((
#         c,
#         cv2.isContourConvex(c),
#         cv2.contourArea(c),
#     ))
# contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
# max_contour = contour_info[0]

# #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# # Mask is black, polygon is white
# mask = np.zeros(edges.shape)
# cv2.fillConvexPoly(mask, max_contour[0], (255))

# # #-- Smooth mask, then blur it
# mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
# mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
# mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
# mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

# #-- Blend masked img into MASK_COLOR background
# mask_stack  = mask_stack.astype('float32') / 255.0         
# img         = img.astype('float32') / 255.0    
# masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)  
# masked = (masked * 255).astype('uint8')                    

# print('\nMask_stack')
# plt.imshow(mask_stack, cmap = 'gray')
# plt.show()
# print('\nMasked')
# plt.imshow(masked, cmap = 'gray')
# plt.show()



# # print('------------Skimage-------------')
# # import skimage
# # import skimage.feature
# # import skimage.viewer 

# # import imagecodecs
# # from skimage import io
# # from socket import socket

# # sigma = 12
# # low_threshold = 0.1
# # high_threshold = 0.9

# # image = skimage.io.imread(fname="1.tif", as_gray=True)
# # io.imshow(image)

# # # # load and display original image as grayscale
# # # image = skimage.io.imread(fname="t000.tif", as_gray=True)
# # # viewer = skimage.viewer(image=image)
# # # viewer.show()


# # edges = skimage.feature.canny(
# #     image=image,
# #     sigma=sigma,
# #     low_threshold=low_threshold,
# #     high_threshold=high_threshold,
# # )
# # io.imshow(edges)

# # for _ in range(9):
# #     edges = skimage.feature.canny(image=image, sigma=_, low_threshold=low_threshold, high_threshold=high_threshold, )
# #     io.imshow(edges)
# #     io.imshow(image)
# #     print(f'Sigma = {_}')

# # #gaussian laplacian preprocessing

# # # display edges
# # # viewer = skimage.viewer.ImageViewer(edges)
# # # viewer.show()


# # # # Create the plugin and give it a name
# # # canny_plugin = skimage.viewer.plugins.Plugin(image_filter=skimage.feature.canny)
# # # canny_plugin.name = "Canny Filter Plugin"

# # # # Add sliders for the parameters
# # # canny_plugin += skimage.viewer.widgets.Slider(
# # #     name="sigma", low=0.0, high=7.0, value=2.0
# # # )
# # # canny_plugin += skimage.viewer.widgets.Slider(
# # #     name="low_threshold", low=0.0, high=1.0, value=0.1
# # # )
# # # canny_plugin += skimage.viewer.widgets.Slider(
# # #     name="high_threshold", low=0.0, high=1.0, value=0.2
# # # )

# # # # add the plugin to the viewer and show the window
# # # viewer += canny_plugin
# # # # viewer.show()