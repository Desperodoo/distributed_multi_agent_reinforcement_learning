# coding: utf-8
import math
import time
import numpy as np
from sko.PSO import PSO
from scipy.optimize import root
from scipy.optimize import minimize
# from environment.env_3d.EA_compare import FuncPara


def pursuer_strategy(agent_num, xs, ys, zs, phi, gamma, v, p_max, p_ser, p_com, exyz, e_xyz, ephi, ega, ev, e_max, ang_lmt, v_lmt, c_l, fe, time_step):
    """
    :param agent_num: as the name describes
    :param xs: x_position of the pursuers
    :param ys: y_position of the pursuers
    :param zs: z_position of the pursuers
    :param phi: yaw angle of the pursuers
    :param gamma: pitch angle of the pursuers
    :param v: velocity of the pursuers
    :param p_max: maximum velocity of the pursuers
    :param p_ser: sensing range of the pursuers
    :param p_com: communication range of the pursuers
    :param exyz: current xyz_position of the evader
    :param e_xyz: next xyz_position of the evader
    :param ephi: current yaw angle of the evader
    :param ega: current pitch angle of the evader
    :param e_v: velocity of the evader
    :param e_max: maximum velocity of the evader
    :param ang_lmt: maximum angle change
    :param c_l: collision range or kill radius
    :param fe: formation space
    :return:
    """
    def p_lo(px, py, pz, elo, e_phi, e_ga, ema, qn):  # 计算p位置相对于e速度的夹角，用于划分区域
        ver_x = px[qn] - elo[0]
        ver_y = py[qn] - elo[1]
        ver_z = pz[qn] - elo[2]
        ver_ex = ema * np.cos(e_phi) * np.cos(e_ga)
        ver_ey = ema * np.sin(e_phi) * np.cos(e_ga)
        ver_ez = ema * np.sin(e_ga)
        if ema == 0:
            i_ta = np.pi
        else:
            cos_ita = (ver_x * ver_ex + ver_y * ver_ey + ver_z * ver_ez) / \
                      (np.sqrt(ver_x ** 2 + ver_y ** 2 + ver_z ** 2) * ema)
            i_ta = math.acos(cos_ita)
        return i_ta

    new_xs = np.zeros(agent_num)
    new_ys = np.zeros(agent_num)
    new_zs = np.zeros(agent_num)

    for m in range(agent_num):
        new_xs[m] = xs[m] + v[m] * np.cos(phi[m]) * np.cos(gamma[m]) * time_step
        new_ys[m] = ys[m] + v[m] * np.sin(phi[m]) * np.cos(gamma[m]) * time_step
        new_zs[m] = zs[m] + v[m] * np.sin(gamma[m]) * time_step

    action_list = list()
    for q in range(agent_num):
        ita = p_lo(xs, ys, zs, exyz, ephi, ega, ev, q)  # 计算个体与对象速度矢量的夹角
        #  范围感知
        dis_jpe = np.sqrt((xs[q] - exyz[0]) ** 2 + (ys[q] - exyz[1]) ** 2 + (zs[q] - exyz[2]) ** 2)
        if dis_jpe <= p_ser:
            j_pe = 1
        else:
            j_pe = 0

        #  范围通信
        p_rax, p_ray, p_raz = [], [], []
        for pk in range(agent_num):
            dis_pp = np.sqrt((xs[q] - xs[pk]) ** 2 + (ys[q] - ys[pk]) ** 2 + (zs[q] - zs[pk]) ** 2)
            if dis_pp <= p_com and dis_pp != 0:
                p_rax.append(new_xs[pk])
                p_ray.append(new_ys[pk])
                p_raz.append(new_zs[pk])

        # 进入优化程序
        my_alg = FuncPara(col=c_l, x=p_rax, y=p_ray, z=p_raz,
                          xs_0=xs[q], ys_0=ys[q], zs_0=zs[q], phi_0=phi[q], gamma_0=gamma[q], v_0=v[q], v_max_0=p_max,
                          fle=fe, ang_lmt=ang_lmt, v_lmt=v_lmt,
                          exyz_0=exyz, e_xyz_0=e_xyz, e_phi_0=ephi, e_ga_0=ega, e_v_0=ev, e_v_max_0=e_max,
                          ita_0=ita, p_se0=p_ser, jpe0=j_pe, step_size=time_step)

        action = my_alg.opt()
        action_list.append(action)

    return action_list


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
    def __init__(self, col, x, y, z,
                 xs_0, ys_0, zs_0, phi_0, gamma_0, v_0, v_max_0, fle, ang_lmt, v_lmt,
                 exyz_0, e_xyz_0, e_phi_0, e_ga_0, e_v_0, e_v_max_0,
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
        self.ang_lmt = ang_lmt
        self.v_lmt = v_lmt
        self.neigh_num = len(self.x)

    def pso(self):
        # print('neighbor: ', self.neigh_num)
        # front_time = time.time()
        pso = PSO(func=self.obj_func, dim=3, pop=50, max_iter=35, lb=[-1, -1, -1], ub=[1, 1, 1],
                  w=0.7, c1=2, c2=2)
        pso.run()
        g_best_x = pso.gbest_x.tolist()  # gbest_x : array_like, shape is (1,dim)
        g_best_y = pso.gbest_y.tolist()
        # print('pso_result: ', g_best_x, ' cost: ', g_best_y, ' time: ', time.time() - front_time)

        action = g_best_x
        return action

    def opt(self):
        bounds_phi = [
            np.clip((self.phi_0 - self.ang_lmt) / np.pi, -1, 1),
            np.clip((self.phi_0 + self.ang_lmt) / np.pi, -1, 1)
        ]
        # ga belong to [-pi/2, pi/2]
        bounds_ga = [
            np.clip((self.gamma_0 - self.ang_lmt) / (np.pi / 2), -1, 1),
            np.clip((self.gamma_0 + self.ang_lmt) / (np.pi / 2), -1, 1)
        ]
        # v belong to [0, v_max]
        bounds_v = [
            np.clip((self.v_0 - self.v_lmt) * 2 - 1, -1, 1),
            np.clip((self.v_0 + self.v_lmt) * 2 - 1, -1, 1)
        ]

        init_phi = self.phi_0 / np.pi
        init_ga = self.gamma_0 / (np.pi / 2)
        init_v = self.v_0 * 2 - 1

        def obj_fun0(
                action, e_xyz0, exyz_0,
                x, y, z, xs_0, ys_0, zs_0, phi_0, gamma_0, v_0, v_max_0, ang_lmt, v_lmt,
                col, fle, ita_0, jpe_0, p_se0
        ):
            point = Point(
                idx=0,
                x=xs_0,
                y=ys_0,
                z=zs_0,
                phi=phi_0,
                gamma=gamma_0,
                v=v_0,
                v_max=v_max_0,
                sen_range=None,
                comm_range=None,
                ang_lmt=ang_lmt,
                v_lmt=v_lmt
            )
            point.step(step_size=0.5, a=action)
            x_0 = point.x
            y_0 = point.y
            z_0 = point.z
            dis_sel0 = np.sqrt((x_0 - e_xyz0[0]) ** 2 + (y_0 - e_xyz0[1]) ** 2 + (z_0 - e_xyz0[2]) ** 2)
            if jpe_0 == 1:
                rl_sum = dis_sel0 + ita_0  # 与逃逸者的距离+视线与逃逸者速度的夹角
                return rl_sum  # 没有邻居，能感知到e
            else:
                bap = [10, 10, 10]  # 设定了回归据点--------！
                dis_pb = np.sqrt((x_0 - bap[0]) ** 2 + (y_0 - bap[1]) ** 2 + (z_0 - bap[2]) ** 2)
                rl_sum = dis_pb
                return rl_sum  # 没有邻居，不能感知到e

        def obj_fun1(
                action, e_xyz0, exyz_0,
                x, y, z, xs_0, ys_0, zs_0, phi_0, gamma_0, v_0, v_max_0, ang_lmt, v_lmt,
                col, fle, ita_0, jpe_0, p_se0
        ):
            point = Point(
                idx=0,
                x=xs_0,
                y=ys_0,
                z=zs_0,
                phi=phi_0,
                gamma=gamma_0,
                v=v_0,
                v_max=v_max_0,
                sen_range=None,
                comm_range=None,
                ang_lmt=ang_lmt,
                v_lmt=v_lmt
            )
            point.step(step_size=0.5, a=action)
            x_0 = point.x
            y_0 = point.y
            z_0 = point.z
            dis_sel0 = np.sqrt((x_0 - e_xyz0[0]) ** 2 + (y_0 - e_xyz0[1]) ** 2 + (z_0 - e_xyz0[2]) ** 2)

            dis_pp0 = np.sqrt((x_0 - x[0]) ** 2 + (y_0 - y[0]) ** 2 + (z_0 - z[0]) ** 2)  # 和队友的距离
            avo_col1 = fle ** 2 / dis_pp0 - fle
            if jpe_0 == 1:
                rl_sum = dis_sel0 + self.ita_0 + avo_col1
                return rl_sum  # 只有一个邻居，能感知到e
            else:
                bap = [10, 10, 10]  # 设定了回归据点--------！
                dis_pb = np.sqrt((x_0 - bap[0]) ** 2 + (y_0 - bap[1]) ** 2 + (z_0 - bap[2]) ** 2)
                rl_sum = dis_pp0 + dis_pb + avo_col1
                return rl_sum  # 只有一个邻居，不能感知到e

        def obj_fun2(
                action, e_xyz0, exyz_0,
                x, y, z, xs_0, ys_0, zs_0, phi_0, gamma_0, v_0, v_max_0, ang_lmt, v_lmt,
                col, fle, ita_0, jpe_0, p_se0
        ):
            point = Point(
                idx=0,
                x=xs_0,
                y=ys_0,
                z=zs_0,
                phi=phi_0,
                gamma=gamma_0,
                v=v_0,
                v_max=v_max_0,
                sen_range=None,
                comm_range=None,
                ang_lmt=ang_lmt,
                v_lmt=v_lmt
            )
            point.step(step_size=0.5, a=action)
            x_0 = point.x
            y_0 = point.y
            z_0 = point.z
            dis_sel0 = np.sqrt((x_0 - e_xyz0[0]) ** 2 + (y_0 - e_xyz0[1]) ** 2 + (z_0 - e_xyz0[2]) ** 2)
            dis_ra, dis_fl = 0, 0  # 存放通信范围内与邻居的距离和，避碰/队形
            for r in range(len(x)):  # 针对每一个队友
                ln = np.sqrt((x[r] - x_0) ** 2 + (y[r] - y_0) ** 2 + (z[r] - z_0) ** 2)
                # TODO: Adjust the power of the power function
                dis_ra = dis_ra + col ** 3 / ln ** 3
                if ln < fle:
                    dis_fl = dis_fl + fle ** 2 / ln - fle
                else:
                    # TODO: Adjust the critical d_fle
                    dis_fl = dis_fl + np.log(ln - fle + 1)
            avo_col = dis_ra  # 避碰评价部分
            rl_sum = 1 * (0.8 * (np.exp(dis_sel0 / col) - 1) + 0.3059 * ita_0 + 0.5298 * avo_col)
            return rl_sum

        def obj_fun3(
                action, e_xyz0, exyz_0,
                x, y, z, xs_0, ys_0, zs_0, phi_0, gamma_0, v_0, v_max_0, ang_lmt, v_lmt,
                col, fle, ita_0, jpe_0, p_se0
        ):
            jud = len(x)  # 通信范围内的队友数（不包含自己）
            sy_bac = 0  # 对面潜在合作判定
            cou_rl = 0
            rc_s, lc_s = 0, 0
            dis_ra, dis_fl = 0, 0  # 存放通信范围内与邻居的距离和，避碰/队形
            nei_x, nei_y, nei_z = [], [], []

            point = Point(
                idx=0,
                x=xs_0,
                y=ys_0,
                z=zs_0,
                phi=phi_0,
                gamma=gamma_0,
                v=v_0,
                v_max=v_max_0,
                sen_range=None,
                comm_range=None,
                ang_lmt=ang_lmt,
                v_lmt=v_lmt
            )

            point.step(step_size=0.5, a=action)
            x_0 = point.x
            y_0 = point.y
            z_0 = point.z
            dis_sel0 = np.sqrt((x_0 - e_xyz0[0]) ** 2 + (y_0 - e_xyz0[1]) ** 2 + (z_0 - e_xyz0[2]) ** 2)

            for r in range(len(x)):  # 针对每一个队友
                ln = np.sqrt((x[r] - x_0) ** 2 + (y[r] - y_0) ** 2 + (z[r] - z_0) ** 2)
                # TODO: Adjust the power of the power function
                dis_ra = dis_ra + col ** 3 / ln ** 3
                if ln < fle:
                    dis_fl = dis_fl + fle ** 2 / ln - fle
                else:
                    # TODO: Adjust the critical d_fle
                    if jud <= 6:
                        dis_fl = dis_fl + np.exp(ln - fle) - 1
                    else:
                        dis_fl = dis_fl + np.log(ln - fle + 1)

                if jpe_0 == 1:  # 如果能感知到Evader，就可以判断在通讯范围内的队友与Evader之间的拓扑关系（队友是否能感知到Evader）
                    dr = np.sqrt((x[r] - exyz_0[0]) ** 2 + (y[r] - exyz_0[1]) ** 2 + (z[r] - exyz_0[2]) ** 2)
                    if dr <= p_se0:  # 寻找同样能感知到e的邻居集合
                        nei_x.append(x[r])
                        nei_y.append(y[r])
                        nei_z.append(z[r])

                    ver_per = [x[r] - exyz_0[0], y[r] - exyz_0[1], z[r] - exyz_0[2]]
                    ver_pe0 = [xs_0 - exyz_0[0], ys_0 - exyz_0[1], zs_0 - exyz_0[2]]
                    cos_per0 = (ver_per[0] * ver_pe0[0] + ver_per[1] * ver_pe0[1] + ver_per[2] * ver_pe0[2]) / \
                               (np.sqrt(ver_per[0] ** 2 + ver_per[1] ** 2 + ver_per[2] ** 2) *
                                np.sqrt(ver_pe0[0] ** 2 + ver_pe0[1] ** 2 + ver_pe0[2] ** 2))

                    ang_per0 = math.acos(cos_per0)
                    # TODO: Adjust the angle
                    if ang_per0 >= 3 * np.pi / 4:
                        sy_bac = 1  # 说明对面有潜在可合作者

            avo_col = dis_ra  # 避碰评价部分
            avo_fle = dis_fl  # 队形评价部分

            if jpe_0 == 1 and (ita_0 > 0.5 * np.pi or sy_bac == 1):  # 朝向我0，没有合作者0
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
                                sphere = Radius(x0=[x_0, y_0, z_0], xn=nei_pos)
                                roots = root(sphere, x0=np.array([1, 1, 1, 1]))
                                rc = roots.x[-1]
                                rc_s = rc + rc_s
                                cou_rl = cou_rl + 1

                    if ita_0 > 3 * np.pi / 4:
                        rl_sum = 1 * (0.1411 * rc_s / cou_rl + 0 * lc_s / cou_rl + 0 * dis_sel0 / col +
                                      0.2627 * ita_0 + 0.455 * avo_col)
                    else:  # 收缩比牵引部分小一点，防止目标还没进包围就收缩
                        rl_sum = 1 * (0.1411 * rc_s / cou_rl + 0 * lc_s / cou_rl + 0.8 * dis_sel0 / col +
                                      0.2627 * (2 * ita_0) + 0.455 * avo_col)
                    return rl_sum  # 有两个以上邻居，能感知到e
                else:
                    rl_sum = 1 * (0.8 * (np.exp(dis_sel0 / col) - 1) + 0.3059 * ita_0 + 0.5298 * avo_col)
                    return rl_sum
            else:
                rl_sum = avo_fle
                return rl_sum  # 有两个以上邻居，不能感知到e

        if self.neigh_num == 0:
            obj_fun_opt = obj_fun0
        elif self.neigh_num == 1:
            obj_fun_opt = obj_fun1
        elif self.neigh_num == 2:
            obj_fun_opt = obj_fun2
        else:
            obj_fun_opt = obj_fun3

        result = minimize(
            fun=obj_fun_opt,
            # x0=np.array([init_phi, init_ga, init_v]),
            x0=np.array([0, 0, 0]),
            args=(
                self.e_xyz0, self.exyz_0,
                self.x, self.y, self.z, self.xs_0, self.ys_0, self.zs_0, self.phi_0, self.gamma_0, self.v_0,
                self.v_max_0, self.ang_lmt, self.v_lmt, self.col, self.fle, self.ita_0, self.jpe0, self.p_se0
            ),
            method='SLSQP',
            bounds=[
                bounds_phi,
                bounds_ga,
                bounds_v
            ]
        )
        # print(
        #     "opt_result: ", result.x,
        #     " cost: ", obj_fun_opt(
        #         result.x, self.e_xyz0, self.exyz_0,
        #         self.x, self.y, self.z, self.xs_0, self.ys_0, self.zs_0, self.phi_0, self.gamma_0, self.v_0,
        #         self.v_max_0, self.ang_lmt, self.v_lmt, self.col, self.fle, self.ita_0, self.jpe0, self.p_se0
        #     ),
        #     "pso_cost: ", obj_fun_opt(
        #         g_best_x, self.e_xyz0, self.exyz_0,
        #         self.x, self.y, self.z, self.xs_0, self.ys_0, self.zs_0, self.phi_0, self.gamma_0, self.v_0,
        #         self.v_max_0, self.ang_lmt, self.v_lmt, self.col, self.fle, self.ita_0, self.jpe0, self.p_se0
        #     ),
        #     " time: ", time.time() - front_time
        # )
        action = result.x.tolist()
        return action

    # ---------------------p目标函数-----------------------------
    def obj_func(self, phi, gamma, v):
        p = [phi, gamma, v]
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

        point = Point(
            idx=0,
            x=self.xs_0,
            y=self.ys_0,
            z=self.zs_0,
            phi=self.phi_0,
            gamma=self.gamma_0,
            v=self.v_0,
            v_max=self.v_max_0,
            sen_range=None,
            comm_range=None,
            ang_lmt=self.ang_lmt,
            v_lmt=self.v_lmt
        )
        point.step(step_size=0.5, a=p)
        x_0 = point.x
        y_0 = point.y
        z_0 = point.z
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
                    ver_pe0 = [self.xs_0 - self.exyz_0[0], self.ys_0 - self.exyz_0[1], self.zs_0 - self.exyz_0[2]]
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
                                    x0=[x_0, y_0, z_0],
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
