import cv2
import numpy as np
import os

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense

#-----------------------------------------------------------------------------------------------------
result_dir = 'results'

classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
           'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
           'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
           'Windflower', 'Pansy','katabami']
nb_classes = len(classes)

img_height, img_width = 150, 150
channels = 3

# VGG16
input_tensor = Input(shape=(img_height, img_width, channels))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC
fc = Sequential()
fc.add(Flatten(input_shape=vgg16.output_shape[1:]))
fc.add(Dense(256, activation='relu'))
fc.add(Dropout(0.5))
fc.add(Dense(nb_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(vgg16.input,fc(vgg16.output))

# 学習済みの重みをロード
model.load_weights(os.path.join(result_dir, 'finetuning.h5'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


img=cv2.imread('img/7.jpg')
img=cv2.resize(img,(640,480))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#土色のHSV範囲指定。
lower = np.array([5,0,0], dtype=np.uint8)
upper = np.array([20,255,255], dtype=np.uint8)
#土色のRGB範囲指定。
rgblower = np.array([0,50,80], dtype=np.uint8)
rgbupper = np.array([60,110,220], dtype=np.uint8)

#土色を白色に、その他を黒に
mask = cv2.inRange(hsv, lower, upper)
rgbmask=cv2.inRange(img, rgblower, rgbupper)
img_grand = cv2.bitwise_or(mask,rgbmask)
kernel=np.ones((19,19),np.uint8)
closing=cv2.morphologyEx(img_grand,cv2.MORPH_CLOSE,kernel)
closing=cv2.bitwise_not(closing)

# ラベリング結果書き出し用に二値画像をカラー変換
color_src01 = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
# ラベリング処理（白色のオブジェクトを抽出する）
label = cv2.connectedComponentsWithStats(closing)

# オブジェクト情報を項目別に抽出
#ラベルの数
n = label[0] - 1
#x,y,width high
data = np.delete(label[2], 0, 0)
#重心座標
center = np.delete(label[3], 0, 0)

tuning=0
x0=1
y0=1
x1=0
y1=0

if n>0:
    max=0
# オブジェクト情報を利用してラベリング結果を画面に表示
    for i in range(n):
        men=data[i][4]
        label_group_index=np.where(label[1]==i+1)
        if 255 in closing[label_group_index]:
            if max<men:
                max=men
                maxdata=i

    #print(maxdata)
    while 1:
        if x0<=0 or y0<=0 or x1>=640 or y1>=480:
            break
        elif tuning==50:
            break
        x0 = data[maxdata][0]-tuning
        y0 = data[maxdata][1]-tuning
        x1 = data[maxdata][0] + data[maxdata][2]+tuning
        y1 = data[maxdata][1] + data[maxdata][3]+tuning
        tuning=tuning+1

    color_src01[y0:y1,x0:x1]=img[y0:y1,x0:x1]
    cv2.rectangle(color_src01, (x0, y0), (x1, y1), (0, 0, 255),5)
    #---vgg16による画像認識--------------------------------------------------------------
    dst = color_src01[y0:y1,x0:x1]
    # 画像を読み込んで4次元テンソルへ変換
    img = cv2.resize(dst,(img_height,img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！
    # これを忘れると結果がおかしくなるので注意
    x = x / 255.0

    # クラスを予測
    # 入力は1枚の画像なので[0]のみ
    pred = model.predict(x)[0]

    # 予測確率が高いトップ5を出力
    top = 5
    top_indices = pred.argsort()[-top:][::-1]
    result = [(classes[i], pred[i]) for i in top_indices]
    for x in result:
        print(x)
        if x[0]=='katabami' and x[1]>0.6:
            print(center[maxdata][0],center[maxdata][1])
            # 各オブジェクトの重心座標をに黄文字で表示
            cv2.putText(color_src01, "X: " + str(int(center[maxdata][0])), (x1 - 30, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
            cv2.putText(color_src01, "Y: " + str(int(center[maxdata][1])), (x1 - 30, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
            # 各オブジェクトの重心座標をに黄文字で表示
            cv2.putText(color_src01, "katabami", (x1 - 30, y1 + 45), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
            cv2.putText(color_src01, str(x[1]*100)+"%", (x1 - 30, y1 + 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))
else:
    print("オブジェクトが抽出できませんでした")

cv2.imwrite("./img/mask_img.jpg", color_src01)
cv2.imwrite("./img/mask_.jpg", closing)
cv2.imshow("Foreground", closing)
cv2.imshow("Fore", color_src01)
cv2.waitKey(0)
cv2.destroyAllWindows()



