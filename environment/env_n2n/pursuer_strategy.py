# coding: utf-8
import math
import random
import numpy as np
from environment.env_n2n.EA_compare import FuncPara


def pursuer_strategy(agent_num, xs, ys, beta, p_max, p_ser, p_com, exy, e_xy, ega, e_max, ang_lim, alg, c_l, fe):
    """

    :param agent_num: as the name describes
    :param xs: x_position of the pursuers
    :param ys: y_position of the pursuers
    :param beta: yaw angle of the pursuers
    :param p_max: maximum velocity of the pursuers
    :param p_ser: sensing range of the pursuers
    :param p_com: communication range of the pursuers
    :param exy: current xy_position of the evader
    :param e_xy: next xy_position of the evader
    :param ega: current yaw angle of the evader
    :param e_max: maximum velocity of the evader
    :param ang_lim: maximum yaw angle change
    :param alg: algorithm type
    :param c_l: collision range or kill radius
    :param fe: formation space
    :return:
    """
    def p_lo(px, py, elo, eg, ema, qn):  # 计算p位置相对于e速度的夹角，用于划分区域
        ver_x = px[qn] - elo[0]
        ver_y = py[qn] - elo[1]
        ver_ex = ema * np.cos(eg)
        ver_ey = ema * np.sin(eg)
        cos_ita = (ver_x * ver_ex + ver_y * ver_ey) / (np.sqrt(ver_x ** 2 + ver_y ** 2) * ema)
        i_ta = math.acos(cos_ita)
        return i_ta

    new_xs = new_ys = new_x = new_y = new_beta = np.zeros(agent_num)
    n_beta_list = list()
    for q in range(agent_num):
        ita = p_lo(xs, ys, exy, ega, e_max, q)  # 计算个体与对象速度矢量的夹角
        #  范围感知
        dis_jpe = np.sqrt((xs[q] - exy[0]) ** 2 + (ys[q] - exy[1]) ** 2)
        if dis_jpe <= p_ser:
            j_pe = 1
        else:
            j_pe = 0

        if (ita < ang_lim) and (j_pe == 1) and False:
            exs = exy[0] + e_max * np.cos(ega)
            eys = exy[1] + e_max * np.sin(ega)
            ver_p = [new_xs[q] - xs[q], new_ys[q] - ys[q]]
            ver_e = [exs - xs[q], eys - ys[q]]
            cos_pe = (ver_p[0] * ver_e[0] + ver_p[1] * ver_e[1]) / (np.sqrt(ver_p[0] ** 2 + ver_p[1] ** 2)
                                                                    * np.sqrt(ver_e[0] ** 2 + ver_e[1] ** 2))
            ang_pe = math.acos(cos_pe)
            crz_ep = ver_e[0] * ver_p[1] - ver_e[1] * ver_p[0]  # 利用外积判断

            if ang_pe <= 0.125 * np.pi:
                if crz_ep < 0:
                    new_beta[q] = beta[q] + ang_pe
                else:
                    new_beta[q] = beta[q] - ang_pe

            new_x[q] = xs[q] + p_max * np.cos(new_beta[q])
            new_y[q] = ys[q] + p_max * np.sin(new_beta[q])

        else:
            #  范围通信
            p_rax, p_ray = [], []
            for pk in range(agent_num):
                dis_pp = np.sqrt((xs[q] - xs[pk]) ** 2 + (ys[q] - ys[pk]) ** 2)
                if dis_pp <= p_com and dis_pp != 0:
                    p_rax.append(new_xs[pk])
                    p_ray.append(new_ys[pk])

            # 进入优化程序
            my_alg = FuncPara(alg, c_l, p_rax, p_ray,
                              xs, ys, p_max, beta, fe, q, exy, e_xy, ega, e_max, ita, p_ser, j_pe)
            n_beta, time_cost = my_alg.algorithm()
            n_beta_list.append(n_beta)
            # 新的位置
            new_beta[q] = beta[q] + n_beta
            # 转换到[-pi，pi]
            if new_beta[q] > np.pi:
                new_beta[q] -= 2 * np.pi
            elif new_beta[q] < -np.pi:
                new_beta[q] += 2 * np.pi
            # 归一化
            new_beta[q] /= np.pi
    return new_beta, n_beta_list
