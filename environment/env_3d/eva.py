# _*_ coding: utf-8 _*_
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.optimize import minimize
from sko.PSO import PSO
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
            self.x += self.v * np.cos(self.gamma) * np.cos(phi) * step_size
            self.y += self.v * np.cos(self.gamma) * np.sin(phi) * step_size
            self.z += self.v * np.sin(self.gamma) * step_size


def e_f(xyz, e_phi, e_ga, e_v, e_v_max, e_sr, npx, npy, npz, nphi, ngamma, nv, tp, time_step, ang_lmt, v_lmt, kill_radius):  # 更新e的偏航与位置
    le = len(npx)  # the number of pursuers
    e_nx, e_ny, e_nz, e_nphi, e_ngamma, e_nv = [], [], [], [], [], []
    for cpn in range(le):
        dis_ep = np.sqrt((xyz[0] - npx[cpn]) ** 2 + (xyz[1] - npy[cpn]) ** 2 + (xyz[2] - npz[cpn]) ** 2)
        if dis_ep <= e_sr:
            e_nx.append(npx[cpn])
            e_ny.append(npy[cpn])
            e_nz.append(npz[cpn])
            e_nphi.append(nphi[cpn])
            e_ngamma.append(ngamma[cpn])
            e_nv.append(nv[cpn])

    # PSO
    # obj_fun = ObcFunc(
    #     tar=tp,
    #     xyz=xyz,
    #     phi=e_phi,
    #     gamma=e_ga,
    #     v=1,
    #     v_max=e_v_max,
    #     nx=e_nx,
    #     ny=e_ny,
    #     nz=e_nz,
    #     nphi=e_nphi,
    #     ngamma=e_ngamma,
    #     nv=e_nv,
    #     time_step=time_step
    # )
    # front_time = time.time()
    # e_pso = PSO(func=obj_fun.func, dim=d_, pop=p_, max_iter=m_, lb=[-1, -1, -1], ub=[1, 1, 1], w=0.7, c1=2, c2=2)
    # e_pso.run()
    # g_best_x = e_pso.gbest_x.tolist()  # gbest_x : array_like, shape is (1,dim)
    # g_best_y = e_pso.gbest_y.tolist()
    # print('pso_result: ', g_best_x, ' cost: ', g_best_y, ' time: ', time.time() - front_time)

    # Opt
    # front_time = time.time()
    # 输入是真实值，输出为归一化值
    # phi belong to [-pi, pi]
    bounds_phi = [np.clip((e_phi - ang_lmt) / np.pi, -1, 1), np.clip((e_phi + ang_lmt) / np.pi, -1, 1)]
    # ga belong to [-pi/2, pi/2]
    bounds_ga = [np.clip((e_ga - ang_lmt) / (np.pi / 2), -1, 1), np.clip((e_ga + ang_lmt) / (np.pi / 2), -1, 1)]
    # v belong to [0, v_max]
    bounds_v = [np.clip((e_v - v_lmt) * 2 - 1, -1, 1), np.clip((e_v + v_lmt) * 2 - 1, -1, 1)]

    init_phi = e_phi / np.pi
    init_ga = e_ga / (np.pi / 2)
    init_v = e_v * 2 - 1

    result = minimize(
        fun=obj_func,
        x0=np.array([init_phi, init_ga, init_v]),
        # x0=np.array([0, 0, 0]),
        args=(tp, xyz, e_phi, e_ga, e_v, e_v_max, e_nx, e_ny, e_nz, e_nphi, e_ngamma, e_nv, time_step, kill_radius),
        method='SLSQP',
        bounds=[
            bounds_phi,
            bounds_ga,
            bounds_v
        ]
    )
    # print(
    #     "opt_result: ", result.x,
    #     " cost: ", obj_func(result.x, tp, xyz, e_phi, e_ga, e_v, e_v_max, e_nx, e_ny, e_nz, e_nphi, e_ngamma, e_nv, time_step, kill_radius),
    #     " time: ", time.time() - front_time
    # )
    action = result.x.tolist()
    return action


class ObcFunc:
    def __init__(self, tar, xyz, phi, gamma, v, v_max, nx, ny, nz, nphi, ngamma, nv, time_step):
        self.target = tar
        self.xyz = xyz
        self.phi = phi
        self.gamma = gamma
        self.v = v
        self.v_max = v_max
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nphi = nphi
        self.ngamma = ngamma
        self.nv = nv
        self.n = len(nx)
        self.time_step = time_step

    def func(self, phi, gamma, v):
        evader = Point(
            idx=0,
            x=self.xyz[0],
            y=self.xyz[1],
            z=self.xyz[2],
            phi=self.phi,
            gamma=self.gamma,
            v=self.v,
            v_max=self.v_max,
            sen_range=None,
            comm_range=None,
            ang_lmt=np.pi / 4,
            v_lmt=0.4
        )
        evader.step(step_size=0.5, a=[phi, gamma, v])
        new_x, new_y, new_z = evader.x, evader.y, evader.z
        sdd = 0
        dis_ep = np.zeros(self.n)
        for i in range(self.n):
            new_nx = self.nx[i] + self.nv[i] * np.cos(self.nphi[i]) * np.cos(self.ngamma[i]) * self.time_step
            new_ny = self.ny[i] + self.nv[i] * np.sin(self.nphi[i]) * np.cos(self.ngamma[i]) * self.time_step
            new_nz = self.nz[i] + self.nv[i] * np.sin(self.ngamma[i]) * self.time_step
            dis_ep[i] = np.sqrt((new_x - new_nx) ** 2 + (new_y - new_ny) ** 2 + (new_z - new_nz) ** 2)
        dis_et = np.sqrt((new_x - self.target[0]) ** 2 + (new_y - self.target[1]) ** 2 + (new_z - self.target[2]) ** 2)
        dis_ep.sort()
        for d in range(self.n):
            sdd = sdd + 0.1 / dis_ep[d] ** 3
        cdd = 5 * dis_et + sdd
        # print(cdd)
        return cdd


def obj_func(action, target, xyz, phi, gamma, v, v_max, nx, ny, nz, nphi, ngamma, nv, time_step, kill_radius):
    evader = Point(
        idx=0,
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        phi=phi,
        gamma=gamma,
        v=v,
        v_max=v_max,
        sen_range=None,
        comm_range=None,
        ang_lmt=np.pi / 4,
        v_lmt=0.4
    )
    evader.step(step_size=0.5, a=action)
    new_x, new_y, new_z = evader.x, evader.y, evader.z
    n = len(nx)
    sdd = 0
    dis_ep = np.zeros(n)
    for i in range(n):
        new_nx = nx[i] + nv[i] * np.cos(nphi[i]) * np.cos(ngamma[i]) * time_step
        new_ny = ny[i] + nv[i] * np.sin(nphi[i]) * np.cos(ngamma[i]) * time_step
        new_nz = nz[i] + nv[i] * np.sin(ngamma[i]) * time_step
        dis_ep[i] = np.sqrt((new_x - new_nx) ** 2 + (new_y - new_ny) ** 2 + (new_z - new_nz) ** 2)
    dis_et = np.sqrt((new_x - target[0]) ** 2 + (new_y - target[1]) ** 2 + (new_z - target[2]) ** 2)
    dis_ep.sort()
    for d in range(n):
        sdd = sdd + 1 / (dis_ep[d] / kill_radius) ** 5
    cdd = 1 * dis_et + sdd
    # print(cdd)
    return cdd

