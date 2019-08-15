import cv2
from matplotlib import pyplot as plt
im = cv2.imread("reclassified/train/Disturbing/workspace_1_90_2018-06-19_11-00-33.bmp",cv2.IMREAD_ANYDEPTH)
print (im.dtype)
print (im.shape)
# gray= cv2.imread("fisheye/Perceptible/workspace_1_90_20171017_101500.hdr")
# im1 = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
# # print (im1.dtype)
# # print (im1.shape)
# #cv2.imshow('image',im1)
# plt.imshow(im1,cmap='gray')
# plt.show()
# #
#rows,cols,channels = im.shape
rows,cols= im.shape
for i in range(rows):
  for j in range(cols):
      p = im[i, j]
      if p>255:
        print(i, j,p)
      #for k in range(channels):
         # p = im[i,j,k]
         # print(i, j, k, p)
         # if p>1 :
         #     p=p-1
         #    #print (i,j,k,p)
         # else :
         #     p=0
         # if p>2 :
         #     print(i, j, k, p*255)


#print (im.dtype)
# plt.imshow(im)
# plt.show()