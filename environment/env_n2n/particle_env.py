import numpy as np
from operator import itemgetter
from environment.env_n2n import eva 
from environment.env_n2n.pursuer_strategy import pursuer_strategy
import random
from abc import abstractmethod


# Discrete Version
class Point:
    def __init__(self, idx, x, y, phi, v, sen_range, comm_range, ang_lmt, v_lmt):
        self.idx = idx
        self.x = x
        self.y = y
        self.phi = phi
        self.v = v
        self.sensor_range = sen_range
        self.comm_range = comm_range
        self.delta_phi_max = ang_lmt
        self.active = True
        self.v_lmt = v_lmt

    @abstractmethod
    def step(self, step_size, a):
        
        raise NotImplementedError


class Pursuer(Point):
    def __init__(self, idx, x, y, phi, v, sen_range, comm_range, ang_lmt, v_lmt):
        super().__init__(idx, x, y, phi, v, sen_range, comm_range, ang_lmt, v_lmt)
        self.is_pursuer = True

    def step(self, step_size, a):
        # a belong to [0, 1, 2, 3, 4, 5, 6, 7, 8]
        if a == 0:
            v = 0
        else:
            v = self.v_lmt
            a = a * np.pi / 4
            if a > np.pi:
                a -= 2 * np.pi
            sign_a_phi = np.sign(a * self.phi)
            
            if sign_a_phi >= 0:
                delta_phi = abs(a - self.phi)
                sign = np.sign(a - self.phi)
            else:
                if abs(a - self.phi) < 2 * np.pi - abs(a - self.phi):
                    delta_phi = abs(a - self.phi)
                    sign = np.sign(a - self.phi)
                else:
                    delta_phi = 2 * np.pi - abs(a - self.phi)
                    sign = -np.sign(a - self.phi)

            delta_phi = np.clip(delta_phi, 0, self.delta_phi_max)
            self.phi = self.phi + sign * delta_phi

            if self.phi > np.pi:
                self.phi -= 2 * np.pi
            elif self.phi < -np.pi:
                self.phi += 2 * np.pi
        
        if self.active:
            self.x += v * np.cos(self.phi) * step_size
            self.y += v * np.sin(self.phi) * step_size
            self.v = v

class Evader(Point):
    def __init__(self, idx, x, y, phi, v, sen_range, comm_range, ang_lmt, v_lmt):
        super().__init__(idx, x, y, phi, v, sen_range, comm_range, ang_lmt, v_lmt)
        self.is_pursuer = False

    def step(self, step_size, a):
        # a belong to [-1, 1]
        a *= np.pi  # now a belong to [-pi, pi]
        sign_a_phi = np.sign(a * self.phi)  # 判断同号异号
        if sign_a_phi >= 0:
            delta_phi = abs(a - self.phi)
            sign = np.sign(a - self.phi)
        else:
            if abs(a - self.phi) < 2 * np.pi - abs(a - self.phi):
                delta_phi = abs(a - self.phi)
                sign = np.sign(a - self.phi)
            else:
                delta_phi = 2 * np.pi - abs(a - self.phi)
                sign = -np.sign(a - self.phi)

        delta_phi = np.clip(delta_phi, 0, self.delta_phi_max)
        
        if self.active:
            self.x += self.v * np.cos(self.phi) * step_size
            self.y += self.v * np.sin(self.phi) * step_size
            self.phi = self.phi + sign * delta_phi
            if self.phi > np.pi:
                self.phi -= 2 * np.pi
            elif self.phi < -np.pi:
                self.phi += 2 * np.pi


def square_root(x1, x2, y1, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class ParticleEnv:
    # def __init__(self, cfg: dict):
    def __init__(self):
        cfg = {
            'p_vmax': 0.3,
            'e_vmax': 1,
            'x_bound': [5, 15],
            'y_bound': [5, 15],
            'p_sen_range': 3,
            'p_comm_range': 6,
            'e_sen_range': 3,
            'e_comm_range': 6,
            'target': [17.5, 17.5],
            'kill_radius': 0.5,
            'ang_lmt': np.pi / 4
        }
        self.p_obs_dim = 3
        self.e_obs_dim = 3
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
        self.random = np.random
        # self.random.seed(10086)

        # self.state_dim = 3
        # action space
        self.action_dim = 1

        self.n_episode = 0
        self.episode_limit = 100
        self.shadow_epi = 1000
        self.target_return = 1000

        self.step_size = 0.5
        self.time_step = 0

        self.curriculum = False
        self.p_num = None
        self.e_num = None
        self.p_list = {}
        self.p_idx = []
        self.e_list = {}
        self.e_idx = []
        self.state = None

    def initialize(self, p_num, e_num):
        self.p_num = p_num
        self.e_num = e_num

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
        done = True if self.get_done() or self.time_step >= self.episode_limit else False

        return reward, done, active

    def evader_step(self, p_state):
        p_state = list(map(list, zip(*p_state)))
        # if np.random.rand() > min(self.n_episode, self.shadow_epi) / self.shadow_epi:
        for evader_idx in self.e_idx:
            evader = self.e_list[f'{evader_idx}']
            if evader.active:
                e_g = eva.e_f(
                    xy=np.array([evader.x, evader.y]),
                    e_ga=evader.phi,
                    e_v=evader.v,
                    e_sr=evader.sensor_range,
                    npx=p_state[0],
                    npy=p_state[1],
                    bet=p_state[2],
                    p_v=[self.p_list[f'{i}'].v for i in self.p_idx],
                    p_=max(10, int(50 * min(self.n_episode, self.shadow_epi) / self.shadow_epi)),
                    d_=1,
                    m_=max(5, int(35 * min(self.n_episode, self.shadow_epi) / self.shadow_epi)),
                    tp=self.target
                )

                evader.step(self.step_size, e_g)

    def reset(self):
        self.target = [
            np.random.rand() * 20,
            np.random.rand() * 20
        ]
        self.p_list = {}
        self.p_idx = []
        self.e_list = {}
        self.e_idx = []

        self.time_step = 0
        self.n_episode += 1
        if self.n_episode >= self.shadow_epi / 2:
            self.curriculum = False

        p_pos = self.gen_init_p_pos()
        # p_pos = [np.array([ 1.17513706, -1.70002972]), np.array([1.42664064, 3.0541168 ]), np.array([ 4.64538204, -0.38121069]), np.array([-1.54771406, -3.81924607]), np.array([4.0746203, 3.548883 ]), np.array([-0.91712585,  0.57027576]), np.array([2.22983261, 1.13171621]), np.array([ 1.71596396, -3.73624217]), np.array([-4.38077446, -3.65947774]), np.array([-3.31172795, -0.03736466]), np.array([-1.24765694,  2.90916915]), np.array([-1.57397666, -1.47348422]), np.array([ 5.        , -3.08973354]), np.array([-2.46218519,  4.55911173]), np.array([-4.45710432,  2.71817435])]
        for i in range(self.p_num):
            self.p_list[f'{i}'] = Pursuer(
                idx=i,
                x=p_pos[i][0] + 10,
                y=p_pos[i][1] + 10,
                # phi=(2 * np.random.rand() - 1) * np.pi,
                phi=np.pi / 4,
                v=0,
                sen_range=self.p_sen_range,
                comm_range=self.p_comm_range,
                ang_lmt=self.ang_lmt,
                v_lmt=self.p_vmax
                )
            self.p_idx.append(i)

        center = [20 - self.target[0], 20 - self.target[1]]
        e_pos = self.gen_init_e_pos(center)
        for i in range(self.e_num):
            self.e_list[f'{i}'] = Evader(
                idx=i,
                x=e_pos[i][0],
                y=e_pos[i][1],
                # x=0 + 2.5,
                # y=0 + 2.5,
                phi=np.pi / 4,
                v=self.e_vmax,
                sen_range=self.e_sen_range,
                comm_range=self.e_comm_range,
                ang_lmt=self.ang_lmt,
                v_lmt=self.e_vmax
                )
            self.e_idx.append(i)

    def gen_init_p_pos(self):
        n_points = self.p_num
        min_dist = 2

        sample = []

        while len(sample) < n_points:
            newp = np.random.normal(loc=0, scale=2, size=(2,)).clip(-8, 8)
            collision = 0
            for p in sample:
                if np.linalg.norm(newp - p) < min_dist:
                    collision += 1
                    break
            if collision == 0:
                sample.append(newp)

        return sample
    
    def gen_init_e_pos(self, center):
        center = np.array(center)
        n_points = self.e_num
        min_dist = 2

        sample = []

        while len(sample) < n_points:
            newp = np.random.normal(loc=center, scale=2, size=(2,)).clip(0, 20)
            collision = 0
            for p in sample:
                if np.linalg.norm(newp - p) < min_dist:
                    collision += 1
                    break
            if collision == 0:
                sample.append(newp)

        return sample

    def get_done(self):
        for idx in self.e_idx:
            agent = self.e_list[f'{idx}']
            if np.linalg.norm([agent.x - self.target[0], agent.y - self.target[1]]) <= self.kill_radius:
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

    def agent_reward(self, agent_idx, is_pursuer=True):
        reward = 0
        is_collision = self.collision_detection(agent_idx, is_pursuer, is_inner=False)
        reward += sum(is_collision) * 1
        
        # for i in self.p_idx:
        #     is_collision = self.collision_detection(i, is_pursuer=is_pursuer, is_inner=False)
        #     reward += sum(is_collision) * 0.5

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
                e_p_collision = self.collision_detection(idx, False, False)
                e_alive_list.append(bool(sum(e_p_collision)))
        
        for i, idx in enumerate(p_idx_list):
            agent = self.p_list[f'{idx}']
            if p_alive_list[i]:
                agent.active = False
                agent.x = 1000
                agent.y = 1000
                agent.phi = 0

        for i, idx in enumerate(e_idx_list):
            agent = self.e_list[f'{idx}']
            if e_alive_list[i]:
                agent.active = False
                agent.x = 1000
                agent.y = 1000
                agent.phi = 0
    
    def get_agent_state(self, is_pursuer, idx):
        agent = self.p_list[f'{idx}'] if is_pursuer else self.e_list[f'{idx}']
        return [agent.x, agent.y, agent.phi]

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

    def get_adj_mat(self, obs, be_obs, rag, is_pursuer=True):
        obs_num = len(obs)
        be_obs_num = len(be_obs)
        adj_mat = np.zeros(shape=(obs_num, be_obs_num))

        for i, item_i in enumerate(obs):
            agent_list = self.p_list if is_pursuer else self.e_list
            if agent_list[f'{i}'].active:
                for j, item_j in enumerate(be_obs):
                    if np.linalg.norm([item_i[0] - item_j[0], item_i[1] - item_j[1]]) <= rag:
                        adj_mat[i, j] = 1
        return adj_mat

    def collision_detection(self, agent_idx, is_pursuer, is_inner):
        agent_state = np.array(self.get_agent_state(is_pursuer, agent_idx))
        adv_state = np.array(self.get_team_state(is_pursuer if is_inner else (not is_pursuer), rules=True))
        adv_num = len(adv_state)
        is_collision = list()
        for i in range(adv_num):
            if np.linalg.norm(agent_state[:2] - adv_state[i][:2]) <= self.kill_radius:
                is_collision.append(1)
            else:
                is_collision.append(0)
        return is_collision

    def maven(self, p_state, e_state, e_state_):
        alive_p_num = len(p_state)
        p_state = list(map(list, zip(*p_state)))
        e_state = list(map(list, zip(*e_state)))
        e_state_ = list(map(list, zip(*e_state_)))
        action = pursuer_strategy(
            alg='pso',
            agent_num=alive_p_num,
            xs=p_state[0],
            ys=p_state[1],
            beta=p_state[2],
            p_max=self.p_vmax,
            p_ser=self.p_sen_range,
            p_com=self.p_comm_range,
            exy=e_state[0] + e_state[1],
            e_xy=e_state_[0] + e_state_[1],
            ega=e_state[2],
            e_max=self.e_vmax,
            ang_lim=self.ang_lmt,
            c_l=self.kill_radius,
            fe=0.4
        )  # action belong to [-pi / 4, pi / 4]

        return action

    def choose_evader(self, networks='actor'):
        results = np.zeros(shape=(self.p_num, self.e_num))
        for agent_idx in self.p_idx:
            if self.p_list[f'{agent_idx}'].active:
                agent_state = self.get_agent_state(is_pursuer=True, idx=agent_idx)
                p_x = agent_state[0]
                p_y = agent_state[1]
                dis_list = list()
                for evader_idx in self.e_idx:
                    if self.e_list[f'{evader_idx}'].active:
                        evader_state = self.get_agent_state(is_pursuer=False, idx=evader_idx)
                        e_x = evader_state[0]
                        e_y = evader_state[1]
                        dist = np.linalg.norm([p_x - e_x, p_y - e_y])
                        if networks == 'actor':
                            if dist < self.p_sen_range:
                                dis_list.append([evader_idx, dist])
                        else:
                            dis_list.append([evader_idx, dist])

                if len(dis_list) > 0:
                    dis_list.sort(key=itemgetter(1))
                    results[agent_idx, dis_list[0][0]] = 1
        return results