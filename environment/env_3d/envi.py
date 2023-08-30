# _*_ coding: utf-8 _*_

import string
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc")

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


# 绘制动态图
def sim_moving(step, n, n_e, h_x, h_y, h_z, h_x_e, h_y_e, h_z_e, ep, e_ser, c_r, p_com, fig_title, pp_adj, p_e_adj):
    fig = plt.figure(3)
    ax3 = p3.Axes3D(fig)

    # word_list_p = list()
    # word_list_e = list()

    # for cm in range(n):
    #     handle, = ax3.plot_surface([], [], [], rstride=1, cstride=1, color='green', alpha=0.5)
    #     word_list_p.append(handle)
    #
    # for dm in range(n_e):
    #     handle, = ax3.plot_surface([], [], [], rstride=1, cstride=1, color='red', alpha=0.5)
    #     word_list_e.append(handle)


    def get_p(kp):
        xxp = h_x[kp]
        yyp = h_y[kp]
        zzp = h_z[kp]
        return xxp, yyp, zzp

    def get_e(ke):
        xxe = h_x_e[ke]
        yye = h_y_e[ke]
        zze = h_z_e[ke]
        return xxe, yye, zze

    def update(k):
        ax3.clear()
        max_range = 20
        min_range = 0

        ax3.set_xlim3d([min_range, max_range])
        ax3.set_ylim3d([min_range, max_range])
        ax3.set_zlim3d([min_range, max_range])

        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')

        ax3.grid()
        ax3.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax3.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax3.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax3.set(xlim=[0, 20], ylim=[0, 20], zlim=[0, 20])

        ax3.scatter(h_x_e[0], h_y_e[0], h_z_e[0], color='green', edgecolors='white', marker='s')
        ax3.scatter(ep[0], ep[1], ep[2], color='white', edgecolors='red', marker='^')

        for i in range(n_e):
            ax3.plot([h_x_e[0, i], ep[0]], [h_y_e[0, i], ep[1]], [h_z_e[0, i], ep[2]], color='purple', linestyle='--',
                     alpha=0.3)

        ax3.plot([5, 15], [5, 5], [15, 15], color='red', linestyle='--', alpha=0.3)
        ax3.plot([15, 15], [5, 15], [15, 15], color='red', linestyle='--', alpha=0.3)
        ax3.plot([15, 5], [15, 15], [15, 15], color='red', linestyle='--', alpha=0.3)
        ax3.plot([5, 5], [15, 5], [15, 15], color='red', linestyle='--', alpha=0.3)

        ax3.plot([5, 15], [5, 5], [5, 5], color='red', linestyle='--', alpha=0.3)
        ax3.plot([15, 15], [5, 15], [5, 5], color='red', linestyle='--', alpha=0.3)
        ax3.plot([15, 5], [15, 15], [5, 5], color='red', linestyle='--', alpha=0.3)
        ax3.plot([5, 5], [15, 5], [5, 5], color='red', linestyle='--', alpha=0.3)

        ax3.plot([5, 5], [5, 5], [5, 15], color='red', linestyle='--', alpha=0.3)
        ax3.plot([15, 15], [5, 5], [5, 15], color='red', linestyle='--', alpha=0.3)
        ax3.plot([15, 15], [15, 15], [5, 15], color='red', linestyle='--', alpha=0.3)
        ax3.plot([5, 5], [15, 15], [5, 15], color='red', linestyle='--', alpha=0.3)

        xp, yp, zp = get_p(k)
        xe, ye, ze = get_e(k)

        pp = pp_adj[k]
        pe = p_e_adj[k]
        for i in range(1, len(xp)):
            # print(pp[0, i])
            if pp[0, i] == 1:
                ax3.plot([xp[0], xp[i]], [yp[0], yp[i]], [zp[0], zp[i]], color='green', linestyle='-', alpha=0.3)

        if pe[0, 0] == 1:
            ax3.plot([xp[0], xe[0]], [yp[0], ye[0]], [zp[0], ze[0]], color='red', linestyle='-', alpha=0.3)

        ax3.plot(xp, yp, zp, 'o', color='green')  # 返回的第一个值是update函数需要改变的
        ax3.plot(xe, ye, ze, 'o', color='red')

        t = np.linspace(0, np.pi * 2, 20)
        s = np.linspace(0, np.pi, 20)

        # mt, ms = np.meshgrid(t, s)
        # x = p_com * np.outer(np.cos(mt), np.sin(ms)) + xp[0]
        # y = p_com * np.outer(np.sin(mt), np.sin(ms)) + yp[0]
        # z = p_com * np.outer(np.ones(np.size(mt)), np.cos(ms)) + zp[0]
        # ax3.plot_surface(x, y, z, color='green', alpha=0.005)

        # for ci in range(len(xp)):
        #     mt, ms = np.meshgrid(t, s)
        #     x = c_r * np.outer(np.cos(mt), np.sin(ms)) + xp[ci]
        #     y = c_r * np.outer(np.sin(mt), np.sin(ms)) + yp[ci]
        #     z = c_r * np.outer(np.ones(np.size(mt)), np.cos(ms)) + zp[ci]
        #     ax3.plot_surface(x, y, z, color='green', alpha=0.05)

        for ci in range(len(xe)):
            mt, ms = np.meshgrid(t, s)
            x = e_ser * np.outer(np.cos(mt), np.sin(ms)) + xe[ci]
            y = e_ser * np.outer(np.sin(mt), np.sin(ms)) + ye[ci]
            z = e_ser * np.outer(np.ones(np.size(mt)), np.cos(ms)) + ze[ci]
            ax3.plot_surface(x, y, z, color='red', alpha=0.01)

        ax3.text2D(0.05, 0.95, u"Time = {:.2f} s".format(k * 0.5), transform=ax3.transAxes)

        return fig

    ani = FuncAnimation(fig, update, frames=step, interval=200)
    ani.save(fig_title + '.gif', writer='pillow')
    plt.show()
