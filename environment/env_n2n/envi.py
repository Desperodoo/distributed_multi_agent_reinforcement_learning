# _*_ coding: utf-8 _*_
import imageio
import string
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties
# font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc")

"""
from envi import sim_moving

# 画出来的是仿真动态图，gif格式
sim_moving(step, n, h_x, h_y, h_e, ep, e_ser, c_r)

# step：整个仿真运行的步数
# n：Pursuer的个数
# h_x：记录下来的每一个Pursuer的x坐标值（历史数据），n行，step列
# h_y：记录下来的每一个Pursuer的y坐标值（历史数据），n行，step列
# h_e：记录下来的Evader的xy坐标值（历史数据），step行，2列
# ep：Evader要到达的目标点位置坐标
# e_ser：Evader的感知半径
# c_r：Pursuer的抓捕半径

"""


def sim_moving(step, n, n_e, h_x, h_y, h_x_e, h_y_e, ep, e_ser, c_r):
    def get_p(kp):
        xxp = h_x[:][kp]
        yyp = h_y[:][kp]
        return xxp, yyp

    def get_e(ke):
        xxe = h_x_e[:][ke]
        yye = h_y_e[:][ke]
        return xxe, yye
    
    fig3, ax3 = plt.subplots()
    image_list = list()
    for k in range(step):
        ax3.cla()
        ax3.scatter(h_x_e[0], h_y_e[0], color='green', edgecolors='white', marker='s')
        ax3.scatter(ep[0], ep[1], color='white', edgecolors='red', marker='^')
        for i in range(n_e):
            ax3.plot([h_x_e[0, i], ep[0]], [h_y_e[0, i], ep[1]], color='purple', linestyle='--', alpha=0.3)

        ax3.plot([5, 15], [5, 5], color='red', linestyle='--', alpha=0.3)
        ax3.plot([15, 15], [5, 15], color='red', linestyle='--', alpha=0.3)
        ax3.plot([15, 5], [15, 15], color='red', linestyle='--', alpha=0.3)
        ax3.plot([5, 5], [15, 5], color='red', linestyle='--', alpha=0.3)

        ax3.set_title('Trajectory', size=15)
        ax3.set_xlabel('x/(m)', size=12)
        ax3.set_ylabel('y/(m)', size=12)
        ax3.grid()
        ax3.axis([0, 20, 0, 20])
        ax3.set_aspect('equal')

        xp, yp = get_p(k)
        xe, ye = get_e(k)
        r_theta = np.linspace(0, 2 * np.pi, 360)

        for ci in range(len(xp)):
            cx = xp[ci] + c_r * np.cos(r_theta)
            cy = yp[ci] + c_r * np.sin(r_theta)
            ax3.plot(cx, cy, color='green')

        for ci in range(len(xe)):
            cx = xe[ci] + e_ser * np.cos(r_theta)
            cy = ye[ci] + e_ser * np.sin(r_theta)
            ax3.plot(cx, cy, color='red')
        

        ax3.plot(xp, yp, 'o', color='green')    
        ax3.plot(xe, ye, 'o', color='red')    
        
        fig3.savefig('temp.png')
        image_list.append(imageio.imread('temp.png'))
        # plt.pause(0.1)
        # plt.close()

    imageio.mimsave('sim_moving.gif', image_list, 'GIF', duration=0.2)