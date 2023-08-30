import numpy as np
from environment.env_3d import eva
from environment.env_3d.pursuer_strategy import pursuer_strategy
# import random
# from abc import abstractmethod


# Three Dimensional Version
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
        if self.active:
            phi, gamma, v = a[0], a[1], a[2]  # belong to [-1, 1]
            phi = phi * np.pi  # belong to [-pi, pi]
            gamma = gamma * np.pi / 2  # belong to [-pi / 2, pi /2]
            v = (v + 1) / 2 * self.v_max  # belong to [0, v_max]
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

            self.x += self.v * np.cos(self.gamma) * np.cos(phi) * step_size
            self.y += self.v * np.cos(self.gamma) * np.sin(phi) * step_size
            self.z += self.v * np.sin(self.gamma) * step_size
        else:
            pass


class Pursuer(Point):
    def __init__(self, idx, x, y, z, phi, gamma, v, v_max, sen_range, comm_range, ang_lmt, v_lmt):
        super().__init__(idx, x, y, z, phi, gamma, v, v_max, sen_range, comm_range, ang_lmt, v_lmt)
        self.is_pursuer = True


class Evader(Point):
    def __init__(self, idx, x, y, z, phi, gamma, v, v_max, sen_range, comm_range, ang_lmt, v_lmt):
        super().__init__(idx, x, y, z, phi, gamma, v, v_max, sen_range, comm_range, ang_lmt, v_lmt)
        self.is_pursuer = False


def square_root(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


class ParticleEnv:
    def __init__(self):
        cfg = {
            'p_vmax': 0.7,
            'e_vmax': 1,
            'x_bound': [5, 15],
            'y_bound': [5, 15],
            'z_bound': [5, 15],
            'p_sen_range': 3,
            'p_comm_range': 6,
            'e_sen_range': 3,
            'e_comm_range': 6,
            'target': [17.5, 17.5, 17.5],
            'kill_radius': 0.5,
            'ang_lmt': np.pi / 4,
            'v_lmt': 0.4
        }
        self.p_obs_dim = 6  # x, y, z, phi, gamma, v
        self.e_obs_dim = 6
        self.env_name = 'ParticleEnvBoundGra'
        self.p_vmax = cfg['p_vmax']
        self.e_vmax = cfg['e_vmax']
        self.x_bound = cfg['x_bound']
        self.y_bound = cfg['y_bound']
        self.p_sen_range = cfg['p_sen_range']
        self.p_comm_range = cfg['p_comm_range']
        self.e_sen_range = cfg['e_sen_range']
        self.e_comm_range = cfg['e_comm_range']

        self.target = cfg['target']
        self.kill_radius = cfg['kill_radius']
        self.ang_lmt = cfg['ang_lmt']
        self.v_lmt = cfg['v_lmt']
        self.random = np.random
        # self.random.seed(10086)

        self.state_dim = self.p_obs_dim + self.e_obs_dim
        # action space
        self.action_dim = 3

        self.n_episode = 0
        self.max_step = 200
        self.shadow_epi = 1000
        self.target_return = 1000

        self.step_size = 0.5
        self.time_step = 0

        self.curriculum = False
        self.p_num = None
        self.e_num = None
        self.p_list = None
        self.p_idx = None
        self.e_list = None
        self.e_idx = None
        self.state = None

    def initialize(self, p_num):
        self.p_num = p_num
        self.e_num = 1

    def reset(self):
        self.target = [
            np.random.rand() * 20,
            np.random.rand() * 20,
            np.random.rand() * 20
        ]
        self.p_list = dict()
        self.p_idx = list()
        self.e_list = dict()
        self.e_idx = list()

        self.time_step = 0
        self.n_episode += 1
        if self.n_episode >= self.shadow_epi / 2:
            self.curriculum = False

        def gen_init_p_pos(p_num):
            n_points = p_num
            min_dist = 4
            sample = []
            while len(sample) < n_points:
                newp = np.random.normal(loc=10, scale=2, size=(3,)).clip(5, 15)
                collision = 0
                for p in sample:
                    if np.linalg.norm(newp - p) < min_dist:
                        collision += 1
                        break
                if collision == 0:
                    sample.append(newp)
            return sample

        p_pos = gen_init_p_pos(self.p_num)
        for i in range(self.p_num):
            self.p_list[f'{i}'] = Pursuer(
                idx=i,
                x=p_pos[i][0],
                y=p_pos[i][1],
                z=p_pos[i][2],
                phi=(2 * np.random.rand() - 1) * np.pi,  # phi belong to [-pi, pi]
                gamma=(2 * np.random.rand() - 1) * np.pi / 2,  # gamma belong to [-pi/2, pi/2]
                v=0,  # v belong to [0, vmax]
                v_max=self.p_vmax,
                sen_range=self.p_sen_range,
                comm_range=self.p_comm_range,
                ang_lmt=self.ang_lmt,
                v_lmt=self.v_lmt
            )
            self.p_idx.append(i)

        e_pos = [20 - self.target[0], 20 - self.target[1], 20 - self.target[2]]
        for i in range(self.e_num):
            self.e_list[f'{i}'] = Evader(
                idx=i,
                x=e_pos[0],
                y=e_pos[1],
                z=e_pos[2],
                phi=(2 * np.random.rand() - 1) * np.pi,  # phi belong to [-pi, pi]
                gamma=(2 * np.random.rand() - 1) * np.pi / 2,  # gamma belong to [-pi/2, pi/2]
                v=0,  # v belong to [0, vmax]
                v_max=self.e_vmax,
                sen_range=self.e_sen_range,
                comm_range=self.e_comm_range,
                ang_lmt=self.ang_lmt,
                v_lmt=self.v_lmt
            )
            self.e_idx.append(i)

    def step(self, action):
        self.time_step += 1
        idx = 0
        for pursuer_idx in self.p_idx:
            pur = self.p_list[f'{pursuer_idx}']
            pur.step(self.step_size, action[idx])
            idx += 1

        reward = self.reward(True)
        self.update_agent_active()
        active = self.get_active()
        done = True if self.get_done() or self.time_step >= self.max_step else False

        return reward, done, active

    def get_done(self):
        for idx in self.e_idx:
            agent = self.e_list[f'{idx}']
            if np.linalg.norm([agent.x - self.target[0], agent.y - self.target[1], agent.z - self.target[2]]) <= self.kill_radius:
                return True

        p_alive, e_alive = 0, 0
        for idx in self.p_idx:
            if self.p_list[f'{idx}'].active:
                p_alive += 1
        if p_alive == 0:
            return True

        for idx in self.e_idx:
            if self.e_list[f'{idx}'].active:
                e_alive += 1
        if e_alive == 0:
            return True

        return False

    def get_active(self):
        active = []
        for idx in self.p_idx:
            active.append(1 if self.p_list[f'{idx}'].active else 0)
        return active

    def get_agent_state(self, is_pursuer, idx):
        agent = self.p_list[f'{idx}'] if is_pursuer else self.e_list[f'{idx}']
        return [agent.x, agent.y, agent.z, agent.phi, agent.gamma, agent.v]

    def get_team_state(self, is_pursuer, rules=True):
        if rules:
            team_state = list()
            idx_list = self.p_idx if is_pursuer else self.e_idx
            agent_list = self.p_list if is_pursuer else self.e_list
            for idx in idx_list:
                if agent_list[f'{idx}'].active:
                    team_state.append(self.get_agent_state(is_pursuer, idx))
        else:
            idx_list = self.p_idx if is_pursuer else self.e_idx
            team_state = [self.get_agent_state(is_pursuer, idx) for idx in idx_list]
        return team_state

    def agent_reward(self, agent_idx, is_pursuer=True):
        reward = 0
        is_collision = self.collision_detection(agent_idx, is_pursuer, is_inner=False)
        reward += sum(is_collision) * 1

        inner_collision = self.collision_detection(agent_idx, is_pursuer, is_inner=True)
        reward -= (sum(inner_collision) - 1) * 1
        return reward

    def reward(self, is_pursuer):
        reward = []
        for idx in self.p_idx:
            if self.p_list[f'{idx}'].active:
                reward.append(self.agent_reward(idx, is_pursuer))
            else:
                reward.append(0)
        return reward

    def update_agent_active(self):
        p_idx_list = list()
        p_alive_list = list()
        for idx in self.p_idx:
            if self.p_list[f'{idx}'].active:
                p_idx_list.append(idx)
                p_collision = self.collision_detection(idx, True, True)  # whether collision with the teammate
                p_e_collision = self.collision_detection(idx, True, False)
                p_alive_list.append(bool(sum(p_collision) + sum(p_e_collision) - 1))
        # if bool(sum(p_e_collision)):
        #     print(1)
        e_idx_list = list()
        e_alive_list = list()
        for idx in self.e_idx:
            if self.e_list[f'{idx}'].active:
                e_idx_list.append(idx)
                e_collision = self.collision_detection(idx, False, True)  # whether collision with the teammate
                e_p_collision = self.collision_detection(idx, False, False)
                e_alive_list.append(bool(sum(e_collision) + sum(e_p_collision) - 1))

        for i, idx in enumerate(p_idx_list):
            agent = self.p_list[f'{idx}']
            if p_alive_list[i]:
                agent.active = False
                agent.x = 1000
                agent.y = 1000
                agent.z = 1000
                agent.phi = 0
                agent.gamma = 0
                agent.v = 0

        for i, idx in enumerate(e_idx_list):
            agent = self.e_list[f'{idx}']
            if e_alive_list[i]:
                agent.active = False
                agent.x = 1000
                agent.y = 1000
                agent.z = 1000
                agent.phi = 0
                agent.gamma = 0
                agent.v = 0

    def get_adj_mat(self, obs, be_obs, rag, is_pursuer=True):
        obs_num = len(obs)
        be_obs_num = len(be_obs)
        adj_mat = np.zeros(shape=(obs_num, be_obs_num))

        for i, item_i in enumerate(obs):
            agent_list = self.p_list if is_pursuer else self.e_list
            if agent_list[f'{i}'].active:
                for j, item_j in enumerate(be_obs):
                    if np.linalg.norm([item_i[0] - item_j[0], item_i[1] - item_j[1], item_i[2] - item_j[2]]) <= rag:
                        adj_mat[i, j] = 1
        return adj_mat

    def collision_detection(self, agent_idx, is_pursuer, is_inner):
        agent_state = np.array(self.get_agent_state(is_pursuer, agent_idx))
        adv_state = np.array(self.get_team_state(is_pursuer if is_inner else (not is_pursuer)))
        adv_num = len(adv_state)
        is_collision = list()
        for i in range(adv_num):
            if np.linalg.norm(agent_state[:3] - adv_state[i][:3]) <= self.kill_radius:
                is_collision.append(1)
            else:
                is_collision.append(0)
        return is_collision

    def evader_step(self, p_state):
        p_state = list(map(list, zip(*p_state)))
        # if np.random.rand() > min(self.n_episode, self.shadow_epi) / self.shadow_epi:
        for evader_idx in self.e_idx:
            evader = self.e_list[f'{evader_idx}']
            e_g = eva.e_f(
                xyz=np.array([evader.x, evader.y, evader.z]),
                e_phi=evader.phi,
                e_ga=evader.gamma,
                e_v=evader.v,
                e_v_max=evader.v_max,
                e_sr=evader.sensor_range,
                npx=p_state[0],
                npy=p_state[1],
                npz=p_state[2],
                nphi=p_state[3],
                ngamma=p_state[4],
                nv=p_state[5],
                tp=self.target,
                time_step=self.step_size,
                ang_lmt=self.ang_lmt,
                v_lmt=self.v_lmt,
                kill_radius=self.kill_radius
            )

            evader.step(self.step_size, e_g)

    def maven(self, p_state, e_state, e_state_):
        alive_p_num = len(p_state)
        p_state = list(map(list, zip(*p_state)))
        e_state = list(map(list, zip(*e_state)))
        e_state_ = list(map(list, zip(*e_state_)))
        action = pursuer_strategy(
            agent_num=alive_p_num,
            xs=p_state[0],
            ys=p_state[1],
            zs=p_state[2],
            phi=p_state[3],
            gamma=p_state[4],
            v=p_state[5],
            p_max=self.p_vmax,
            p_ser=self.p_sen_range,
            p_com=self.p_comm_range,
            exyz=e_state[0] + e_state[1] + e_state[2],
            e_xyz=e_state_[0] + e_state_[1] + e_state_[2],
            ephi=e_state[3][0],
            ega=e_state[4][0],
            ev=e_state[5][0],
            e_max=self.e_vmax,
            ang_lmt=self.ang_lmt,
            v_lmt=self.v_lmt,
            c_l=self.kill_radius,
            fe=4,
            time_step=self.step_size
        )  # action belong to [-pi / 4, pi / 4]

        return action
