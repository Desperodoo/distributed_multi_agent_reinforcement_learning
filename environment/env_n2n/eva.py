# _*_ coding: utf-8 _*_

import numpy as np
from scipy.optimize._minimize import minimize

"""
使用方法：

import eva

# 初始化Evader位置与偏航角
exy = np.array([2.5, 2.5])
ega = 0.25 * np.pi

# 设置Evader速度与感知半径
e_max = 1
e_ser = 3

# 设置PSO参数
pn_ = 50
dim_ = 1
max_iter_ = 35

# -------以下为循环内使用-------

# 调用Evader算法
# new_xs, new_ys, a, p_max分别为：Pursuers的x坐标序列，y坐标序列，偏航角序列，速度值
e_xy, e_gama = eva.e_f(exy, ega, e_max, e_ser, new_xs, new_ys, a, p_max, pn_, dim_, max_iter_)

# 更新Evader位置与偏航角
exy = e_xy
ega = e_gama

"""

def e_f(xy, e_ga, e_v, e_sr, npx, npy, bet, p_v, p_, d_, m_, tp):  # 更新e的偏航与位置
    le = len(npx)
    e_nex, e_ney, neb = [], [], []
    for cpn in range(le):
        dis_ep = np.sqrt((xy[0] - npx[cpn]) ** 2 + (xy[1] - npy[cpn]) ** 2)
        if dis_ep <= e_sr:
            e_nex.append(npx[cpn])
            e_ney.append(npy[cpn])
            neb.append(bet[cpn])
    # 进入优化程序
    
    result = minimize(
        fun=obj_func, 
        x0=np.array([0]), 
        args=(tp, xy, e_v, e_nex, e_ney, neb, p_v), 
        method="SLSQP",
        bounds=[[-np.pi, np.pi]]
    )
    # e_x = xy[0] + e_v * np.cos(e_g)
    # e_y = xy[1] + e_v * np.sin(e_g)
    # ecc = np.array([e_x, e_y])
    # 转换到[-pi，pi]
    
    e_g = result.x[0]
    
    # 归一化
    e_g /= np.pi
    return e_g


def obj_func(e_beta, tar, xy0, e_v0, e_nex0, e_ney0, neb0, p_v0):
    sdd = 0
    le = len(neb0)
    dis_ep = np.zeros(le)
    en_x0 = xy0[0] + e_v0 * np.cos(e_beta)
    en_y0 = xy0[1] + e_v0 * np.sin(e_beta)
    for ne in range(le):
        ne_x = e_nex0[ne] + p_v0[ne] * np.cos(neb0[ne])
        ne_y = e_ney0[ne] + p_v0[ne] * np.sin(neb0[ne])
        dis_ep[ne] = np.sqrt((en_x0 - ne_x) ** 2 + (en_y0 - ne_y) ** 2)
    dis_et = np.sqrt((en_x0 - tar[0]) ** 2 + (en_y0 - tar[1]) ** 2)
    dis_ep.sort()
    for d in range(le):
        sdd = sdd + 0.5 / dis_ep[d]
    cdd = 0.5 * dis_et + sdd
    return cdd