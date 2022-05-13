import cv2
import numpy as np
import time
import math


def red_detect(img):
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
    
    return mask1 + mask2

def blue_detect(img):
    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 青色のHSVの値域
    hsv_min = np.array([90,64,0])
    hsv_max = np.array([150,255,255])
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    
    return mask

def analysis_blob(binary_img):
    # 2値画像のラベリング処理
    label = cv2.connectedComponentsWithStats(binary_img)

    # ブロブ情報を項目別に抽出
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)
    max_index=0
    # ブロブ面積最大のインデックス
    if np.argmax(data[:, 4]):
        max_index = np.argmax(data[:, 4])


    # 面積最大ブロブの情報格納用
    maxblob = {}

    # 面積最大ブロブの各種情報を取得
    maxblob["upper_left"] = (data[:, 0][max_index], data[:, 1][max_index]) # 左上座標
    maxblob["width"] = data[:, 2][max_index]  # 幅
    maxblob["height"] = data[:, 3][max_index]  # 高さ
    maxblob["area"] = data[:, 4][max_index]   # 面積
    maxblob["center"] = center[max_index]  # 中心座標
    
    return maxblob

def main():
    data=[]
    data2=[]
    csvfile_path="C:/Users/masah/Documents/ps/lab/data.csv"
    csvfile_path2="C:/Users/masah/Documents/ps/lab/data2.csv"
    # カメラのキャプチャ
    cap = cv2.VideoCapture(0)
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start=time.time()
    while True:
        flag=0
        # フレームを取得
        ret, frame = cap.read()

        # 赤色検出
        mask = red_detect(frame)
        maskblue = blue_detect(frame)

        if mask.any():
            target = analysis_blob(mask)
            center_x = int(target["center"][0])
            center_y = int(target["center"][1])
            cv2.circle(frame, (center_x, center_y), 30, (0, 200, 0),
                    thickness=3, lineType=cv2.LINE_AA)
            data.append([time.time() - start, center_x, center_y])
            flag=flag+1
        
        if maskblue.any():
            target2= analysis_blob(maskblue)
            center_x2 = int(target2["center"][0])
            center_y2 = int(target2["center"][1])
            cv2.circle(frame, (center_x2, center_y2), 30, (200, 0, 0),
                    thickness=3, lineType=cv2.LINE_AA)

            data2.append([time.time() - start, center_x2, center_y2])
            flag=flag+1
        
        if flag==2:
            a=[640-center_x,0]
            b=[center_x2-center_x,center_y2-center_y]
            c=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
            angle=math.acos(c)
            if center_y2>center_y:
                angle=angle*(-1)
            print(math.degrees(angle))        
        # 結果表示
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        # qキーが押されたら途中終了
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    np.savetxt(csvfile_path, np.array(data), delimiter=",")
    np.savetxt(csvfile_path2, np.array(data2), delimiter=",")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()