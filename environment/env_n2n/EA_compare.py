# coding: utf-8
import math
import time
import numpy as np
from sko.SA import SA
from sko.PSO import PSO


class FuncPara(object):
    def __init__(self, alg_name, col, x, y, xs_0, ys_0, v_0, b_0,
                 fle, q_0, xy0, e_xy0, e_ga0, e_v0, ita_0, p_se0, jpe0):
        self.alg_name = alg_name
        self.col = col
        self.x = x
        self.y = y
        self.xs_0 = xs_0
        self.ys_0 = ys_0
        self.v_0 = v_0
        self.b_0 = b_0
        self.fle = fle
        self.q_0 = q_0
        self.xy0 = xy0
        self.e_xy0 = e_xy0
        self.e_ga0 = e_ga0
        self.e_v0 = e_v0
        self.ita_0 = ita_0
        self.p_se0 = p_se0
        self.jpe0 = jpe0

    def algorithm(self):
        choose = self.alg_name
        best_x, best_y = 0, 0
        et, st = 0, -1

        # def obj_func(p):
        #     x1 = p[0]
        #     y1 = x1 + 1
        #     re_sum = y1 ** 2 + 3
        #     return re_sum

        if choose == 'sa':
            st = time.time()
            sa = SA(func=self.obj_func, x0=[0], T_max=100, T_min=1e-7, L=35, max_stay_counter=100)
            best_x, best_y = sa.run()
            et = time.time()
        elif choose == 'pso':
            st = time.time()
            pso = PSO(func=self.obj_func, dim=1, pop=50, max_iter=35, lb=[-0.25 * np.pi], ub=[0.25 * np.pi],
                      w=0.73, c1=1.5, c2=1.5)
            pso.run()
            et = time.time()
            best_x, best_y = pso.gbest_x, pso.gbest_y
        tc = et - st
        # print('\n', 'algorithm:', choose, '\n', 'best_x:', best_x, '\n', 'best_y:', best_y, '\n', 'time_cost:', tc)
        return best_x[0], tc

    # ---------------------p目标函数-----------------------------
    def obj_func(self, p):
        theta = p
        jud = len(self.x)
        sy_bac = 0  # 对面潜在合作判定
        cou_rl = 0
        rc_s, lc_s = 0, 0
        dis_ra, dis_fl = 0, 0  # 存放通信范围内与邻居的距离和，避碰/队形
        nei_x, nei_y = [], []
        nei_xs, nei_ys = [], []

        ne_xy0 = np.zeros(2)
        ne_xy0[0] = self.e_xy0[0]
        ne_xy0[1] = self.e_xy0[1]

        x_0 = self.xs_0[self.q_0] + self.v_0 * np.cos(self.b_0[self.q_0] + theta)
        y_0 = self.ys_0[self.q_0] + self.v_0 * np.sin(self.b_0[self.q_0] + theta)
        dis_sel0 = np.sqrt((x_0 - self.e_xy0[0]) ** 2 + (y_0 - self.e_xy0[1]) ** 2)

        if jud >= 2:
            for r in range(jud):
                ln = np.sqrt((self.x[r] - x_0) ** 2 + (self.y[r] - y_0) ** 2)

                dis_ra = dis_ra + self.col ** 3 / ln ** 3

                if ln < self.fle:
                    dis_fl = dis_fl + self.fle ** 2 / ln - self.fle
                else:
                    if jud <= 6:
                        dis_fl = dis_fl + np.exp(ln - self.fle) - 1
                    else:
                        dis_fl = dis_fl + np.log(ln - self.fle + 1)

                if self.jpe0 == 1:
                    nei_xs.append(self.xs_0[self.q_0])
                    nei_ys.append(self.ys_0[self.q_0])

                    dr = np.sqrt((self.x[r] - self.xy0[0]) ** 2 + (self.y[r] - self.xy0[1]) ** 2)
                    if dr <= self.p_se0:  # 寻找同样能感知到e的邻居集合
                        nei_x.append(self.x[r])
                        nei_y.append(self.y[r])

                        nei_xs.append(self.x[r])
                        nei_ys.append(self.y[r])

                    ver_per = [self.x[r] - self.xy0[0], self.y[r] - self.xy0[1]]
                    ver_pe0 = [self.xs_0[self.q_0] - self.xy0[0], self.ys_0[self.q_0] - self.xy0[1]]
                    cos_per0 = (ver_per[0] * ver_pe0[0] + ver_per[1] * ver_pe0[1]) / \
                               (np.sqrt(ver_per[0] ** 2 + ver_per[1] ** 2) *
                                np.sqrt(ver_pe0[0] ** 2 + ver_pe0[1] ** 2))
                    ang_per0 = math.acos(cos_per0)
                    if ang_per0 >= 3 * np.pi / 4:
                        sy_bac = 1  # 说明对面有潜在可合作者

            avo_col = dis_ra  # 避碰评价部分
            avo_fle = dis_fl  # 队形评价部分

            if self.jpe0 == 1 and (self.ita_0 > 0.5 * np.pi or sy_bac == 1):  # 朝向我0，没有合作者0
                # 开始计算目标函数相关部分
                jun = len(nei_x)
                if jun >= 2:
                    for i in range(jun):
                        for j in range(i + 1, jun):
                            a = np.sqrt((nei_x[i] - x_0) ** 2 + (nei_y[i] - y_0) ** 2)
                            b = np.sqrt((nei_x[j] - nei_x[i]) ** 2 + (nei_y[j] - nei_y[i]) ** 2)
                            c = np.sqrt((nei_x[j] - x_0) ** 2 + (nei_y[j] - y_0) ** 2)
                            hc = 0.5 * (a + b + c)
                            s = np.sqrt(hc * (hc - a) * (hc - b) * (hc - c))
                            rc = a * b * c / (4 * s)  # 计算外接圆半径
                            rc_s = rc + rc_s
                            cou_rl = cou_rl + 1
                    if self.ita_0 > 3 * np.pi / 4:
                        rl_sum = 1 * (0.1411 * rc_s / cou_rl + 0 * lc_s / cou_rl + 0 * dis_sel0 / self.col +
                                      0.2627 * self.ita_0 + 0.455 * avo_col)
                    else:  # 收缩比牵引部分小一点，防止目标还没进包围就收缩
                        rl_sum = 1 * (0.1411 * rc_s / cou_rl + 0 * lc_s / cou_rl + 0.1411 * dis_sel0 / self.col +
                                      0.2627 * (2 * self.ita_0) + 0.455 * avo_col)
                    return rl_sum  # 有两个以上邻居，能感知到e
                else:
                    rl_sum = 1 * (0.1643 * (np.exp(dis_sel0 / self.col) - 1) + 0.3059 * self.ita_0 + 0.5298 * avo_col)
                    return rl_sum
            else:
                rl_sum = avo_fle
                return rl_sum  # 有两个以上邻居，不能感知到e

        elif jud == 1:
            dis_pp0 = np.sqrt((x_0 - self.x[0]) ** 2 + (y_0 - self.y[0]) ** 2)
            avo_col1 = self.fle ** 2 / dis_pp0 - self.fle
            if self.jpe0 == 1:
                rl_sum = dis_sel0 + self.ita_0 + avo_col1
                return rl_sum  # 只有一个邻居，能感知到e
            else:
                bap = [10, 10]  # 设定了回归据点--------！
                dis_pb = np.sqrt((x_0 - bap[0]) ** 2 + (y_0 - bap[1]) ** 2)
                rl_sum = dis_pp0 + dis_pb + avo_col1
                return rl_sum  # 只有一个邻居，不能感知到e

        else:
            if self.jpe0 == 1:
                rl_sum = dis_sel0 + self.ita_0
                return rl_sum  # 没有邻居，能感知到e
            else:
                bap = [10, 10]  # 设定了回归据点--------！
                dis_pb = np.sqrt((x_0 - bap[0]) ** 2 + (y_0 - bap[1]) ** 2)
                rl_sum = dis_pb
                return rl_sum  # 没有邻居，不能感知到e
