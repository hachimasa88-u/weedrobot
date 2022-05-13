import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import glob
from keras.preprocessing.image import load_img,img_to_array

def showImageArray(img_arry):
  n_cols = 8
  n_rows = ((len(img_arry)-1)//n_cols)+1
  fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(10, 1.25*n_rows))
  fig.patch.set_facecolor('white')
  for i,ax in enumerate( ax.flatten() ):
    if i < len(img_arry):
      ax.imshow(img_arry[i])
      ax.set_xticks([])
      ax.set_yticks([])
      img = np.clip(img_arry[i] * 255, 0, 255).astype(np.uint8)
      cv2.imwrite("./export3/"+str(i+650)+".jpg", img)
    else :
      ax.axis('off') # 余白処理
  plt.show()
  plt.close()

#image  array size
img_size = (224,224)
#load images Folder
dir_name = 'orikata'
#File type
file_type  = 'jpg'

#load images and image to array
img_list = glob.glob('./' + dir_name + '/*.' + file_type)

temp_img_array_list = []

for img in img_list:
    temp_img = load_img(img,grayscale=False,target_size=(img_size))
    temp_img_array = img_to_array(temp_img) /255
    temp_img_array_list.append(temp_img_array)

temp_img_array_list = np.array(temp_img_array_list)
print(temp_img_array_list.shape)

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
image_data_generator = ImageDataGenerator(
  rotation_range=20,       # ランダムに±20度範囲で回転 
  width_shift_range=8,     # ランダムに±8px範囲で左右方向移動
  height_shift_range=4,    # ランダムに±4px範囲で上下方向移動
  zoom_range=(0.8, 1.2),   # ランダムに0.8～1.2倍の範囲でズーム
  horizontal_flip=True,    # ランダムで左右反転する
  channel_shift_range=0.2) # チャンネル値（明度）のランダムシフト範囲

org_img = temp_img_array_list[13].copy()
ex_img = np.empty([50, 224, 224, 3])
ex_img[0,:,:,:] = org_img
for i in range(1,50):
  ex_img[i,:,:,:] = image_data_generator.flow( org_img.reshape(1,224,224,3), batch_size=1)[0][0]
showImageArray(ex_img)
