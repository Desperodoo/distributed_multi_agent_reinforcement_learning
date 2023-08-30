# coding: utf-8
import math
import numpy as np
from sko.PSO import PSO
from scipy.optimize import root


class Point:
    def __init__(self, idx, x, y, z, phi, gamma, v, v_max, sen_range, comm_range, ang_lmt, v_lmt):
        self.idx = idx
        self.x = x
        self.y = y
        self.z = z
        self.phi = phi
        self.gamma = gamma
        self.v = v
        self.v_max = v_max
        self.sensor_range = sen_range
        self.comm_range = comm_range
        self.ang_lmt = ang_lmt
        self.v_lmt = v_lmt
        self.active = True

    def step(self, step_size, a):
        phi, gamma, v = a[0], a[1], a[2]
        phi = phi * np.pi
        gamma = gamma * np.pi / 2
        v = (v + 1) / 2 * self.v_max
        delta_gamma = np.clip(gamma - self.gamma, -self.ang_lmt, self.ang_lmt)
        delta_v = np.clip(v - self.v, -self.v_lmt, self.v_lmt)
        self.gamma += delta_gamma
        self.v += delta_v
        # self.v = np.clip(self.v, 0, self.v_max)

        sign_phi_next_phi = np.sign(phi * self.phi)
        if sign_phi_next_phi >= 0:
            delta_phi = np.clip(phi - self.phi, -self.ang_lmt, self.ang_lmt)
        else:
            if abs(phi - self.phi) < 2 * np.pi - abs(phi - self.phi):  # clockwise
                delta_phi = np.clip(phi - self.phi, -self.ang_lmt, self.ang_lmt)
            else:
                delta_phi = 2 * np.pi - abs(phi - self.phi)  # anti-clockwise
                sign = -np.sign(phi - self.phi)
                delta_phi = np.clip(delta_phi, 0, self.ang_lmt) * sign
        self.phi += delta_phi
        if self.phi > np.pi:
            self.phi -= 2 * np.pi
        elif self.phi < -np.pi:
            self.phi += 2 * np.pi

        if self.active:
            self.x += v * np.cos(self.gamma) * np.cos(phi) * step_size
            self.y += v * np.cos(self.gamma) * np.sin(phi) * step_size
            self.z += v * np.sin(self.gamma) * step_size


class FuncPara(object):
    def __init__(self, col, x, y, z, xs_0, ys_0, zs_0, phi_0, gamma_0, v_0, v_max_0,
                 fle, q_0, ang_lmt, v_lmt, exyz_0, e_xyz_0, e_phi_0, e_ga_0, e_v_0, e_v_max_0,
                 ita_0, p_se0, jpe0, step_size):
        self.col = col
        self.x = x  # Prediction of next step agent position
        self.y = y
        self.z = z
        self.xs_0 = xs_0  # Current agent position
        self.ys_0 = ys_0
        self.zs_0 = zs_0
        self.v_0 = v_0
        self.v_max_0 = v_max_0
        self.phi_0 = phi_0
        self.gamma_0 = gamma_0
        self.fle = fle
        self.q_0 = q_0  # agent identifier
        self.exyz_0 = exyz_0
        self.e_xyz0 = e_xyz_0
        self.e_phi0 = e_phi_0
        self.e_ga0 = e_ga_0
        self.e_v0 = e_v_0
        self.e_v_max0 = e_v_max_0
        self.ita_0 = ita_0
        self.p_se0 = p_se0
        self.jpe0 = jpe0
        self.step_size = step_size

        self.point = Point(
            idx=0,
            x=xs_0[q_0],
            y=ys_0[q_0],
            z=zs_0[q_0],
            phi=phi_0[q_0],
            gamma=gamma_0[q_0],
            v=v_0[q_0],
            v_max=v_max_0,
            sen_range=None,
            comm_range=None,
            ang_lmt=ang_lmt,
            v_lmt=v_lmt
        )

    def algorithm(self):
        pso = PSO(func=self.obj_func, dim=2, pop=50, max_iter=35, lb=[-1, -1], ub=[1, 1],
                  w=0.7, c1=2, c2=2)
        pso.run()
        best_x = pso.gbest_x.tolist()
        # print('\n', 'algorithm:', choose, '\n', 'best_x:', best_x, '\n', 'best_y:', best_y, '\n', 'time_cost:', tc)
        # print(best_x)
        # history = pso.gbest_y_hist
        # plt.plot(history)
        # plt.show()
        return best_x

    # ---------------------p目标函数-----------------------------
    def obj_func(self, phi, gamma):
        p = [phi, gamma, 1]
        jud = len(self.x)  # 通信范围内的队友数（不包含自己）
        sy_bac = 0  # 对面潜在合作判定
        cou_rl = 0
        rc_s, lc_s = 0, 0
        dis_ra, dis_fl = 0, 0  # 存放通信范围内与邻居的距离和，避碰/队形
        nei_x, nei_y, nei_z = [], [], []

        ne_xy0 = np.zeros(3)
        ne_xy0[0] = self.e_xyz0[0]
        ne_xy0[1] = self.e_xyz0[1]
        ne_xy0[2] = self.e_xyz0[2]

        self.point.step(step_size=0.5, a=p)
        x_0 = self.point.x
        y_0 = self.point.y
        z_0 = self.point.z
        dis_sel0 = np.sqrt((x_0 - self.e_xyz0[0]) ** 2 + (y_0 - self.e_xyz0[1]) ** 2 + (z_0 - self.e_xyz0[2]) ** 2)

        if jud >= 3:  # 队友数大于等于2
            for r in range(jud):  # 针对每一个队友
                ln = np.sqrt((self.x[r] - x_0) ** 2 + (self.y[r] - y_0) ** 2 + (self.z[r] - z_0) ** 2)
                # TODO: Adjust the power of the power function
                dis_ra = dis_ra + self.col ** 3 / ln ** 3
                if ln < self.fle:
                    dis_fl = dis_fl + self.fle ** 2 / ln - self.fle
                else:
                    # TODO: Adjust the critical d_fle
                    if jud <= 6:
                        dis_fl = dis_fl + np.exp(ln - self.fle) - 1
                    else:
                        dis_fl = dis_fl + np.log(ln - self.fle + 1)

                if self.jpe0 == 1:  # 如果能感知到Evader，就可以判断在通讯范围内的队友与Evader之间的拓扑关系（队友是否能感知到Evader）
                    dr = np.sqrt((self.x[r] - self.exyz_0[0]) ** 2 + (self.y[r] - self.exyz_0[1]) ** 2 + (self.z[r] - self.exyz_0[2]) ** 2)

                    if dr <= self.p_se0:  # 寻找同样能感知到e的邻居集合
                        nei_x.append(self.x[r])
                        nei_y.append(self.y[r])
                        nei_z.append(self.z[r])

                    ver_per = [self.x[r] - self.exyz_0[0], self.y[r] - self.exyz_0[1], self.z[r] - self.exyz_0[2]]
                    ver_pe0 = [self.xs_0[self.q_0] - self.exyz_0[0], self.ys_0[self.q_0] - self.exyz_0[1], self.zs_0[self.q_0] - self.exyz_0[2]]
                    cos_per0 = (ver_per[0] * ver_pe0[0] + ver_per[1] * ver_pe0[1] + ver_per[2] * ver_pe0[2]) / \
                               (np.sqrt(ver_per[0] ** 2 + ver_per[1] ** 2 + ver_per[2] ** 2) *
                                np.sqrt(ver_pe0[0] ** 2 + ver_pe0[1] ** 2 + ver_pe0[2] ** 2))

                    ang_per0 = math.acos(cos_per0)
                    # TODO: Adjust the angle
                    if ang_per0 >= 3 * np.pi / 4:
                        sy_bac = 1  # 说明对面有潜在可合作者

            avo_col = dis_ra  # 避碰评价部分
            avo_fle = dis_fl  # 队形评价部分

            if self.jpe0 == 1 and (self.ita_0 > 0.5 * np.pi or sy_bac == 1):  # 朝向我0，没有合作者0
                # 开始计算目标函数相关部分
                jun = len(nei_x)  # 寻找同样能感知到e的邻居数量
                if jun >= 3:
                    for i in range(jun):
                        for j in range(i + 1, jun):
                            for k in range(j + 1, jun):
                                class Radius:
                                    def __init__(self, x0, xn):
                                        self.x0 = x0
                                        self.x1 = xn[0]
                                        self.x2 = xn[1]
                                        self.x3 = xn[2]

                                    def __call__(self, args):
                                        x, y, z, radius = args[0], args[1], args[2], args[3]
                                        return np.array(
                                            [
                                                radius - np.linalg.norm([x - self.x0[0], y - self.x0[1], z - self.x0[2]]),
                                                radius - np.linalg.norm([x - self.x1[0], y - self.x1[1], z - self.x1[2]]),
                                                radius - np.linalg.norm([x - self.x2[0], y - self.x2[1], z - self.x2[2]]),
                                                radius - np.linalg.norm([x - self.x3[0], y - self.x3[1], z - self.x3[2]]),
                                            ]
                                        )
                                nei_pos = list(map(list, zip(*[nei_x, nei_y, nei_z])))

                                sphere = Radius(
                                    x0=[self.xs_0[self.q_0], self.ys_0[self.q_0], self.zs_0[self.q_0]],
                                    xn=nei_pos
                                )

                                roots = root(sphere, x0=np.array([1, 1, 1, 1]))
                                rc = roots.x[-1]
                                rc_s = rc + rc_s
                                cou_rl = cou_rl + 1

                    if self.ita_0 > 3 * np.pi / 4:
                        rl_sum = 1 * (0.1411 * rc_s / cou_rl + 0 * lc_s / cou_rl + 0 * dis_sel0 / self.col +
                                      0.2627 * self.ita_0 + 0.455 * avo_col)
                    else:  # 收缩比牵引部分小一点，防止目标还没进包围就收缩
                        rl_sum = 1 * (0.1411 * rc_s / cou_rl + 0 * lc_s / cou_rl + 0.5411 * dis_sel0 / self.col +
                                      0.2627 * (2 * self.ita_0) + 0.455 * avo_col)
                    return rl_sum  # 有两个以上邻居，能感知到e
                else:
                    rl_sum = 1 * (0.5643 * (np.exp(dis_sel0 / self.col) - 1) + 0.3059 * self.ita_0 + 0.5298 * avo_col)
                    return rl_sum
            else:
                rl_sum = avo_fle
                return rl_sum  # 有两个以上邻居，不能感知到e
        elif jud == 2:
            for r in range(jud):  # 针对每一个队友
                ln = np.sqrt((self.x[r] - x_0) ** 2 + (self.y[r] - y_0) ** 2 + (self.z[r] - z_0) ** 2)
                # TODO: Adjust the power of the power function
                dis_ra = dis_ra + self.col ** 3 / ln ** 3
                if ln < self.fle:
                    dis_fl = dis_fl + self.fle ** 2 / ln - self.fle
                else:
                    # TODO: Adjust the critical d_fle
                    dis_fl = dis_fl + np.log(ln - self.fle + 1)
            avo_col = dis_ra  # 避碰评价部分
            rl_sum = 1 * (0.5643 * (np.exp(dis_sel0 / self.col) - 1) + 0.3059 * self.ita_0 + 0.5298 * avo_col)
            return rl_sum

        elif jud == 1:  # 队友数等于1
            dis_pp0 = np.sqrt((x_0 - self.x[0]) ** 2 + (y_0 - self.y[0]) ** 2 + (z_0 - self.z[0]) ** 2)  # 和队友的距离
            avo_col1 = self.fle ** 2 / dis_pp0 - self.fle
            if self.jpe0 == 1:
                rl_sum = dis_sel0 + self.ita_0 + avo_col1
                return rl_sum  # 只有一个邻居，能感知到e
            else:
                bap = [10, 10, 10]  # 设定了回归据点--------！
                dis_pb = np.sqrt((x_0 - bap[0]) ** 2 + (y_0 - bap[1]) ** 2 + (z_0 - bap[2]) ** 2)
                rl_sum = dis_pp0 + dis_pb + avo_col1
                return rl_sum  # 只有一个邻居，不能感知到e

        else:  # 没有队友
            if self.jpe0 == 1:
                rl_sum = dis_sel0 + self.ita_0  # 与逃逸者的距离+视线与逃逸者速度的夹角
                return rl_sum  # 没有邻居，能感知到e
            else:
                bap = [10, 10, 10]  # 设定了回归据点--------！
                dis_pb = np.sqrt((x_0 - bap[0]) ** 2 + (y_0 - bap[1]) ** 2 + (z_0 - bap[2]) ** 2)
                rl_sum = dis_pb
                return rl_sum  # 没有邻居，不能感知到e
