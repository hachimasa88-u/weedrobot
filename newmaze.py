import cv2
import numpy as np
from numpy.lib.type_check import real_if_close
from matplotlib import pyplot as pyp
import serial
import time

#ser=serial.Serial('/dev/ttyACM0',9600)

forward="forward"
right="right"
left="left"
stop="stop"
BLOCK_LENGTH=0 #1区画の長さ
STEP_LENGTH=0 #1周で進む距離
TREAD_CIRCUIT=0 #1旋回するのに必要な距離(トレッド幅*pi)
pals=374



def drv_mtr(direction):
    if direction=="forward":
        ser.write(str.encode('f'))
    elif direction=="right":
        ser.write(str.encode('r'))
    elif direction=="left":
        ser.write(str.encode('l'))
    elif direction=="stop":
        ser.write(str.encode('s'))

def ctrl_mtr_wait(around):
    data=ser.read()
    while data<around:
        print(data)


def ctrl_mtr_forward(i_block):
    stepcount=i_block*(BLOCK_LENGTH/STEP_LENGTH)
    drv_mtr(forward)
    ctrl_mtr_wait(stepcount)
    drv_mtr(stop)

def ctrl_mtr_turn(i_angle):
    stepcount=(TREAD_CIRCUIT*(i_angle/360))/STEP_LENGTH
    if i_angle>0:
        drv_mtr(right)
        ctrl_mtr_wait(stepcount)
        drv_mtr(stop)
    else:
        drv_mtr(left)
        ctrl_mtr_wait(stepcount)
        drv_mtr(stop)


def printMaze(a,width,height):
    """迷路を描画"""
    for i in range(height):
        row = a[i * width : (i + 1) * width]
        print(" ".join(row))


def at(x, y, a,width):
    """迷路の(x, y)座標のオブジェクトを返す"""
    return a[y * width + x]


def setChar(x, y, char, a,width):
    """迷路の(x, y)座標のオブジェクトを設定する"""
    a[y * width + x] = char

def red_detect(img):
    """障害物を#に変換"""
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 赤色のHSVの値域1
    hsv_min = np.array([0,127,0])
    hsv_max = np.array([30,255,255])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)

    # 赤色のHSVの値域2
    hsv_min = np.array([150,127,0])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(hsv, hsv_min, hsv_max)
    

    if (mask1+mask2).any():
        return "#"
    else:
        return " "

def app_mode1():
        #画像読み込み
        img = cv2.imread('une2.jpg')
        img=cv2.resize(img,(640,480))
    
        #解像度設定
        height = 16  # height
        width = 12  # width

        #画像分割
        chunks = []
        for row_img in np.array_split(img, height, axis=0):
            for chunk in np.array_split(row_img, width, axis=1):
                chunks.append(chunk)
    
        """初期値"""
        ss=width*(height-2)+1 #start
        sg=width+1            #goal
        count=-1 #方向
        acount=1
        b=1
        while True:
            """迷路作成"""
            #障害物→#
            a=[]
            for i, chunk in enumerate(chunks):
                a.append(red_detect(chunk))
            #壁→#
            for i in range(height*width):
                if i<=width:
                    a[i]="#"
                elif i%width==0 or i%width==width-1:
                    a[i]="#"
                elif i>width*(height-1):
                    a[i]="#"

            a[ss]="S"
            a[sg]="G"

            """1次元から2次元に変換"""
            sx=ss%width
            sy=(ss-sx)//width
            gx=sg%width
            gy=(sg-gx)//width
            start = [sx, sy]
            goal = [gx, gy]

            visited = [False] * width * height #探索済み配列
            costs = [999999] * width * height  #コスト配列
    
            # スタート位置をqueueに入れる
            queue = [start]
            costs[start[1] * width + start[0]] = 0
    
            while len(queue) > 0:
                # queueから位置を取り出す
                p, queue = queue[0], queue[1:]
                visited[p[1] * width + p[0]] = True

                # 上下左右を検証
                for i in [[0, -1], [0, 1], [-1, 0], [1, 0]]:
                    x, y = p[0] + i[0], p[1] + i[1]
                    if x < 0 or y < 0 or x > width - 1 or y > height - 1:
                        continue

                    if at(x, y, a, width) == " " and visited[y * width + x] == False:
                        queue.append([x, y])
                        costs[y * width + x] = costs[p[1] * width + p[0]] + 1
                    if at(x, y, a,width) == "G":
                        queue = []
                        costs[y * width + x] = costs[p[1] * width + p[0]] + 1
                        break
    
            # costsをゴールから辿り、最短経路を求める
            point = goal
            cost = costs[point[1] * width + point[0]]
            while cost != 1:
                for i in [[0, -1], [0, 1], [-1, 0], [1, 0]]:
                    x, y = point[0] + i[0], point[1] + i[1]
                    if x < 0 or y < 0 or x > width - 1 or y > height - 1:
                        continue
                    if costs[y * width + x] == cost - 1:
                        cost = costs[y * width + x]
                        point = [x, y]
                        setChar(x, y, ".", a,width)
                        grax.append(x*40)
                        gray.append(y*40)

                        break
    
            printMaze(a,width,height) #迷路表示

            flag=1
            pp=start
            while flag:
                for i in [[0, -1], [1, 0], [0, 1], [-1, 0]]:
                    x, y = pp[0] + i[0], pp[1] + i[1]
                    count=count+1
                    if at(x, y, a, width) == "." or at(x, y, a,width) == "G":
                        if at(x, y, a,width) == "G":
                            flag=0
                        setChar(x, y, "☆", a,width) #探索済み
                        pp[0]=x
                        pp[1]=y                        
                        if (count%4)==0:
                            print("上")
                        elif (count%4)==1:
                            print("右")
                            count=(count-1)%4
                        elif (count%4)==2:
                            print("下")
                            count=(count-2)%4
                        elif (count%4)==3:
                            print("左")
                            count=(count+1)%4

            if a[width*(height-1)-2]=="☆":
                break
            
            """次のstart,goal"""
            #ns=ss+1
            ss=sg
            #sg=ns

            if acount%2==1:
                sg=sg+1
            elif acount%2==0:
                ssx=sg%width
                ssy=(sg-ssx)//width
                if ssy<height/2:
                    sg=width*(height-2)+ssx
                else:
                    sg=width+ssx
            acount=acount+1
            """
            if acount%4==1:
                sg=width*b+(width-b-1)
            elif acount%4==2:
                sg=width*(height-b-1)+(width-b-1)
            elif acount%4==3:
                sg=width*(height-b-1)+(b+1)
            elif acount%4==0:
                b=b+1
                sg=width*b+b
            acount=acount+1
            """
            








if __name__ == "__main__":
    key=input('mode')    
    grax=[]
    gray=[]
    if key=="1":
        app_mode1()
        pyp.scatter(grax,gray)
        pyp.plot(grax,gray)
        pyp.show()
    else:
        drv_mtr(forward)
        ctrl_mtr_wait(pals)
        drv_mtr(stop)
    