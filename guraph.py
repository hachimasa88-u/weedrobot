import numpy as np
import matplotlib.pyplot as plt

def main():
    # CSVのロード
    data = np.genfromtxt(
        "C:/Users/masah/Documents/ps/lab/data.csv", delimiter=",", dtype='float')
    data2 = np.genfromtxt(
        "C:/Users/masah/Documents/ps/lab/data2.csv", delimiter=",", dtype='float')

    # 2次元配列を分割（経過時間t, x座標, y座標の1次元配列)
    t = data[:,0]
    x = data[:,1]
    y = data[:,2]
    t2 = data2[:,0]
    x2 = data2[:,1]
    y2 = data2[:,2]
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.family'] = 'Times New Roman'
 
# 目盛を内側にする。
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
 
# グラフの上下左右に目盛線を付ける。
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
 
# スケール設定
    ax1.set_xlim(0, 800)
    ax1.set_ylim(0, 400)
 
# 軸のラベルを設定する。
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
 
# データプロット
    
    ax1.scatter(x, y, label='Tracking result')
    ax1.scatter(x2, y2, label='Tracking result2')
    ax1.plot(x,y)
    ax1.plot(x2,y2)
    plt.legend()
    fig.tight_layout()
    """
    # グラフにプロット
    plt.rcParams["font.family"] = "Times New Roman" # フォントの種類
    plt.plot(t, x, "r-", label="x")
    plt.plot(t, y, "b-", label="y")
    plt.xlabel("Time[sec]", fontsize=16)     # x軸ラベル
    plt.ylabel("Position[px]", fontsize=16)    # y軸ラベル
    plt.grid()         # グリッド表示
    plt.legend(loc=1, fontsize=16)       # 凡例表示
    plt.show()
    """
# グラフを表示する。
    plt.show()
    plt.close()
    


if __name__ == "__main__":
    main()