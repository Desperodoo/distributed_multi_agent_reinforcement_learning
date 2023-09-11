import time, math
import os, sys, hydra
import warnings
import random
import argparse
import numpy as np
from numba import jit
from numba import prange
from copy import deepcopy
from numba import jit, prange, float64, int32, boolean
from environment.pursuit_evasion_game.base_env import BaseEnv
from environment.pursuit_evasion_game.gif_plotting import sim_moving
from environment.pursuit_evasion_game.Occupied_Grid_Map import OccupiedGridMap
from skimage.segmentation import find_boundaries
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning


def get_boundary_map(occupied_grid_map: OccupiedGridMap) -> OccupiedGridMap:
    boundary_map = deepcopy(occupied_grid_map)
    boundary_map.grid_map = find_boundaries(occupied_grid_map.grid_map, mode='inner')
    boundary_map.obstacles = np.argwhere(boundary_map.grid_map == 1).tolist()
    obstacle_agent = deepcopy(boundary_map.obstacles)
    for obstacle in obstacle_agent:
        obstacle.append(0)
        obstacle.append(0)
    boundary_map.obstacle_agent = obstacle_agent
    return boundary_map
    
@jit(parallel=True)    
def get_raser_map(boundary_map: OccupiedGridMap, num_beams: int32, radius: int32):
    hash_map = np.zeros((*boundary_map.boundaries, len(boundary_map.obstacles)))
    (width, height) = boundary_map.boundaries
    for idx in prange(width * height):
        x = idx // height
        y = idx % height
        for beam in range(num_beams):
            beam_angle = beam * 2 * np.pi / num_beams
            beam_dir_x = np.cos(beam_angle)
            beam_dir_y = np.sin(beam_angle)
            for beam_range in range(radius):
                beam_current_x = x + beam_range * beam_dir_x
                beam_current_y = y + beam_range * beam_dir_y
                if (beam_current_x < 0 or beam_current_x >= width or beam_current_y < 0 or beam_current_y >= height):
                    break
                
                beam_current_pos = [int(beam_current_x), int(beam_current_y)]
                if not boundary_map.is_unoccupied(beam_current_pos):
                    idx = boundary_map.obstacles.index(beam_current_pos)
                    hash_map[x, y, idx] = 1
                    break
    
    hash_map = hash_map.tolist()
    return hash_map


class Pursuit_Env(BaseEnv):
    def __init__(self, cfg):
        super().__init__(cfg.map, cfg.env, cfg.defender, cfg.attacker, cfg.sensor)

    def reset(self):
        self.time_step = 0
        self.n_episode += 1
        self.collision = False
        
        inflated_map = self.init_map()
        self.boundary_map = get_boundary_map(self.occupied_map)
        self.raser_map = get_raser_map(boundary_map=self.boundary_map, num_beams=self.sensor_config.num_beams, radius=self.sensor_config.radius)
        self.inflated_map = deepcopy(inflated_map)
        # No need for navigation and coverage
        self.init_target(inflated_map=inflated_map)
        inflated_map = self.init_defender(min_dist=4, inflated_map=inflated_map)
        # TODO: the target should be assigned to the attacker manually
        self.init_attacker(inflated_map=inflated_map, is_percepted=True, target_list=self.target)
        
    def attacker_step(self):
        # Based On A Star Algorithm
        state = self.get_state(agent_type='defender')
        dynamic_map = deepcopy(self.occupied_map)
        dynamic_map.extended_moving_obstacles(state)
        path_list = list()
        for attacker in self.attacker_list:
            if self.time_step % self.env_config.difficulty == 0:
                attacker.replan(moving_obs=state, occupied_map=self.occupied_map)
            if len(attacker.path) >= 2:
                last_way_point = attacker.path[-1]
                if np.linalg.norm((attacker.x - last_way_point[0], attacker.y - last_way_point[1])) < self.map_config.resolution:
                    attacker.path.pop()
                way_point = attacker.path[-1]  
                    
            else:
                way_point = attacker.path[-1]
            phi = attacker.waypoint2phi(way_point)
            action = [np.cos(phi) * self.attacker_config.vmax, np.sin(phi) * self.attacker_config.vmax]
            [x, y, vx, vy, theta] = attacker.step(action=action)
            
            if dynamic_map.in_bound((x, y)) and dynamic_map.is_unoccupied((x, y)):
                attacker.apply_update([x, y, vx, vy, theta])
            else:
                print(False)
            if np.linalg.norm((self.target[0][0] - x, self.target[0][1] - y)) <= self.attacker_config.collision_radius:
                self.init_target(inflated_map=self.inflated_map)
                attacker.target = self.target[0]
            path_list.append(deepcopy(attacker.path))
        return path_list
        
    def step(self, action):
        next_state = list()
        rewards = list()
        can_applys = list()
        self.time_step += 1
        for idx, defender in enumerate(self.defender_list):
            next_state.append(defender.step(action[idx]))
            
        for state in next_state:
            reward, can_apply = self.defender_reward(state, next_state)
            rewards.append(reward)
            can_applys.append(can_apply)
        
        for idx, defender in enumerate(self.defender_list):
            if can_applys[idx]:
                defender.apply_update(next_state[idx])
                
        done = True if self.time_step >= self.max_steps else False
        info = None
        return rewards, done, info

    def get_done(self):
        pass
    
    def defender_reward(self, state, next_state):
        reward = 0
        can_apply = True
        
        inner_collision = self.collision_detection(state, obstacle_type='defender', next_state=next_state)
        reward -= (sum(inner_collision) - 1) * 1

        obstacle_collision = self.collision_detection(state, obstacle_type='obstacle')
        reward -= obstacle_collision
        
        if reward < 0:
            can_apply = False
            self.collision = True
            return reward, can_apply
        
        boundaries = self.map_config.map_size
        state[0] = np.clip(state[0], 0, boundaries[0] - 1)
        state[1] = np.clip(state[1], 0, boundaries[1] - 1)
        is_collision = self.collision_detection(state, obstacle_type='attacker')
        reward += sum(is_collision) * 1

        return reward, can_apply
    
    def collision_detection(self, state, obstacle_type: str = 'obstacle', next_state: list = None):
        if obstacle_type == 'obstacle':
            collision = False
            for i in range(-1, 2, 1):
                for j in range(-1, 2, 1):
                    inflated_pos = (state[0] + i * self.defender_config.collision_radius, state[1] + j * self.defender_config.collision_radius)
                    if self.occupied_map.in_bound(inflated_pos):
                        collision = not self.occupied_map.is_unoccupied(inflated_pos)
                    if collision == True:
                        break
                if collision == True:
                    break
            return collision
        else:
            if obstacle_type == 'defender':
                obstacles = next_state
            else:
                obstacles = self.get_state(agent_type=obstacle_type)

            collision = list()
            for obstacle in obstacles:
                if np.linalg.norm([obstacle[0] - state[0], obstacle[1] - state[1]]) <= self.defender_config.collision_radius:
                    collision.append(1)
                else:
                    collision.append(0)

            return collision
    
    def get_agent_state(self, agent):
        return [agent.x, agent.y, agent.vx, agent.vy]

    def communicate(self):
        """
        the obstacles have no impact on the communication between agents
        :return: adj_mat: the adjacent matrix of the agents
        """
        states = self.get_state(agent_type='defender')
        adj_mat = np.zeros(shape=(self.num_defender, self.num_defender))
        for i, item_i in enumerate(states):
            for j, item_j in enumerate(states):
                if (i <= j) and (np.linalg.norm([item_i[0] - item_j[0], item_i[1] - item_j[1]]) <= self.defender_config.comm_range):
                    adj_mat[i, j] = 1
                    adj_mat[j, 1] = 1
        adj_mat = adj_mat.tolist()
        return adj_mat
    
    def sensor(self):
        obstacle_adj_list = list()
        attacker_adj_list = list()
        for defender in self.defender_list:
            obstacle_adj = self.raser_map[int(defender.x)][int(defender.y)]
            attacker_adj = defender.find_attacker(
                occupied_grid_map=self.occupied_map, 
                pos=(round(defender.x), round(defender.y)),
                attacker_pos=(round(self.attacker_list[0].x), round(self.attacker_list[0].y)),
            )
            obstacle_adj_list.append(obstacle_adj)
            attacker_adj_list.append(attacker_adj)
        return obstacle_adj_list, attacker_adj_list

    def demon(self):
        theta_list = [i * np.pi / 4 for i in range(0, 8)]
        actions_mat = [[np.cos(t), np.sin(t)] for t in theta_list]
        actions_mat.append([0., 0.])
        action_list = list()
        e_x = self.attacker_list[0].x
        e_y = self.attacker_list[0].y
        for defender in self.defender_list:
            x = defender.x
            y = defender.y
            radius = np.linalg.norm([x - e_x, y - e_y])
            if math.isclose(radius, 0.0, abs_tol=0.01):
                action = [0., 0.]
            else:
                phi = np.sign(e_y - y) * np.arccos((e_x - x) / (radius + 1e-3))
                action = [np.cos(phi), np.sin(phi)]
            middle_a = [np.linalg.norm((a[0] - action[0], a[1] - action[1])) for a in actions_mat]
            action_list.append(middle_a.index(min(middle_a)))
        return action_list


@hydra.main(config_path='./', config_name='config.yaml', version_base=None)
def main(cfg):
    warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
    warnings.simplefilter('ignore', category=NumbaWarning)        
    # Initialize the Env
    env = Pursuit_Env(cfg)
    env.reset()
    done = False
    acc_reward = 0
    # Evaluate Matrix
    epi_obs_p = list()
    epi_obs_e = list()
    epi_target = list()
    epi_r = list()
    epi_path = list()
    epi_p_o_adj = list()
    epi_p_e_adj = list()
    epi_p_p_adj = list()
    epi_extended_obstacles = list()
    win_tag = False
    start_time = time.time()
    idx = 0
    while not done:
        print(idx)
        idx += 1
        state = env.get_state(agent_type='attacker')
        p_p_adj = env.communicate()
        p_o_adj, p_e_adj = env.sensor()
        # action = [random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8]) for _ in range(env_config.num_defender)]
        action = env.demon()
        path = env.attacker_step()
        rewards, done, info = env.step(action)
        acc_reward += sum(rewards)
        # Store Evaluate Matrix
        epi_obs_p.append(env.get_state(agent_type='defender'))
        epi_obs_e.append(env.get_state(agent_type='attacker'))
        epi_target.append(env.target[0])
        epi_r.append(sum(rewards))
        epi_path.append(path)
        epi_p_p_adj.append(p_p_adj)
        epi_p_e_adj.append(p_e_adj)
        epi_p_o_adj.append(p_o_adj)
        # epi_extended_obstacles.append(pred_map.ex_moving_obstacles + pred_map.ex_obstacles)
        if done:
            # Print Game Result
            print('DONE!')
            print('time cost: ', time.time() - start_time)
            print(f'reward: {acc_reward}')

            epi_obs_p = np.array(epi_obs_p)
            epi_obs_e = np.array(epi_obs_e)
            # Plotting
            sim_moving(
                step=env.time_step,
                height=cfg.map.map_size[0],
                width=cfg.map.map_size[1],
                obstacles=env.occupied_map.obstacles,
                boundary_obstacles=env.boundary_map.obstacles,
                # extended_obstacles=epi_extended_obstacles,
                box_width=cfg.map.resolution,
                n_p=cfg.env.num_defender,
                n_e=1,
                p_x=epi_obs_p[:, :, 0],
                p_y=epi_obs_p[:, :, 1],
                e_x=epi_obs_e[:, :, 0],
                e_y=epi_obs_e[:, :, 1],
                path=epi_path,
                target=epi_target,
                e_ser=cfg.attacker.sen_range,
                c_r=cfg.defender.collision_radius,
                p_p_adj=epi_p_p_adj,
                p_e_adj=epi_p_e_adj,
                p_o_adj=epi_p_o_adj,
                dir='sim_moving' + str(time.time())
            )
            break


if __name__ == '__main__':
    main()