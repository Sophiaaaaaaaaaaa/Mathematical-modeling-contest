import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
fig, ax = plt.subplots()
xdata, ydata = [], []      #初始化两个数组
ln, = ax.plot([], [], 'r-', animated=False)  #第三个参数表示画曲线的颜色和线型，具体参见：https://blog.csdn.net/tengqingyong/article/details/78829596
f=[1,2,3,4,5,6,7,8,9,0]

global i
i=1
def init():
    ax.set_xlim(-5, 5)  #设置x轴的范围pi代表3.14...圆周率，
    ax.set_ylim(-5, 5)
    return ln,               #返回曲线

def update(n):
    xdata.append(math.cos(n)+(math.cos(6*n))/6)   #将每次传过来的n追加到xdata中
    ydata.append(math.sin(n)-(math.sin(6*n))/6)
    ln.set_data(xdata, ydata)    #重新设置曲线的值
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 10),     #这里的frames在调用update函数是会将frames作为实参传递给“n”
                    init_func=init, blit=True)
plt.show()