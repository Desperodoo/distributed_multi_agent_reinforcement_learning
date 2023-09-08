import argparse


# Env Config
parser = argparse.ArgumentParser("Configuration Setting of the MRS Environment")
parser.add_argument("--env_name", type=str, default='Pursuit-Evasion Game')
parser.add_argument("--max_steps", type=int, default=250, help='The maximum time steps in a episode')
parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')
parser.add_argument("--num_target", type=int, default=1, help='The number of target point, 1 in pursuit, n in coverage, 0 in navigation')
parser.add_argument("--num_defender", type=int, default=10, help='The number of defender (pursuer/server/navigator)')
parser.add_argument("--num_attacker", type=int, default=1, help='The number of attacker (evader/client/target)')
parser.add_argument("--defender_class", type=str, default='Pursuer', help='The class of the defender')
parser.add_argument("--attacker_class", type=str, default='Evader', help='The class of the attacker')
env_config = parser.parse_args()

# Map Config
parser = argparse.ArgumentParser("Configuration Setting of the Map")
parser.add_argument("--resolution", type=int, default=1, help='The resolution of the map')
parser.add_argument("--num_obstacle_block", type=int, default=5, help='The number of the obstacles')
parser.add_argument("--center", type=int, default=(30, 30), help='The center of the obstacles')
parser.add_argument("--variance", type=int, default=10, help='The varience of normal distribution that generate the position of obstacle block')
parser.add_argument("--map_size", type=tuple, default=(60, 60), help='The size of the map')
parser.add_argument("--is3D", type=bool, default=False, help='The dimension of freedom, 2 or 3')
parser.add_argument("--max_num_obstacle", type=int, default=110, help='The max number of boundary obstacle, equivalent to num_obs_block * 22 (boundary of a 6x7 rectangle)')
map_config = parser.parse_args()

# Pursuer Config
parser = argparse.ArgumentParser("Configuration Setting of the Defender")
parser.add_argument("--sen_range", type=int, default=8, help='The sensor range of the agents')
parser.add_argument("--comm_range", type=int, default=16, help='The communication range of the agents')
parser.add_argument("--collision_radius", type=float, default=0.5, help='The smallest distance at which a collision can occur between two agents')
parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')
parser.add_argument("--vmax", type=float, default=2, help='The limitation of the velocity of the defender')
parser.add_argument("--tau", type=float, default=0.2, help='The time constant of first-order dynamic system')
parser.add_argument("--DOF", type=int, default=2, help='The dimension of freedom, 2 or 3')
defender_config = parser.parse_args()

# Evader Config
parser = argparse.ArgumentParser("Configuration Setting of the Attacker")
parser.add_argument("--sen_range", type=int, default=8, help='The sensor range of the agents')
parser.add_argument("--comm_range", type=int, default=16, help='The communication range of the agents')
parser.add_argument("--collision_radius", type=float, default=0.5, help='The smallest distance at which a collision can occur between two agents')
parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')
parser.add_argument("--vmax", type=float, default=4, help='The limitation of the velocity of the defender')
parser.add_argument("--tau", type=float, default=0.2, help='The time constant of first-order dynamic system')
parser.add_argument("--DOF", type=int, default=2, help='The dimension of freedom, 2 or 3')
parser.add_argument("--x_dim", type=int, default=map_config.map_size[0], help='The x-dimension of map')
parser.add_argument("--y_dim", type=int, default=map_config.map_size[1], help='The y-dimension of map')
parser.add_argument("--extend_dis", type=int, default=3, help='The extend distance for astar')
attacker_config = parser.parse_args()

# Sensor Config
parser = argparse.ArgumentParser("Configuration Setting of the Sensor")
parser.add_argument("--num_beams", type=int, default=36, help='The number of beams in LiDAR')
parser.add_argument("--radius", type=int, default=defender_config.sen_range, help='The radius of beams in LiDAR')
sensor_config = parser.parse_args()

# Algorithm Config
parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")
## Main parameters
parser.add_argument("--num_defender", type=int, default=env_config.num_defender, help='The number of defender (pursuer/server/navigator)')
parser.add_argument("--num_attacker", type=int, default=env_config.num_attacker, help='The number of attacker (evader/client/target)')
# parser.add_argument("--env_class", default=ParticleEnv)
parser.add_argument("--state_dim", type=int, default=4, help="The state dimension = pursuer state + evader state")
parser.add_argument("--action_dim", type=int, default=9, help="The dimension of action space")
# parser.add_argument("--agent_class", default=MAPPO)
parser.add_argument("--pretrain_model_cwd", type=str, default="./pretrain/experiment/pretrain_model_15")
parser.add_argument("--learner_device", type=str, default="cuda")
parser.add_argument("--worker_device", type=str, default="cpu")
parser.add_argument("--evaluator_device", type=str, default="cpu")
## Learner Hyperparameters
parser.add_argument("--max_train_steps", type=int, default=int(2e8), help=" Maximum number of training steps")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.05, help="GAE parameter")
parser.add_argument("--K_epochs", type=int, default=1, help="GAE parameter")
parser.add_argument("--entropy_coef", type=float, default=0.05, help="Trick 5: policy entropy")
parser.add_argument("--save_cwd", type=str, default=f"./model")
# parser.add_argument("--mini_batch_size", type=int, default=500, help="Minibatch size (the number of episodes)")
# parser.add_argument("--batch_size", type=int, default=500, help="Batch size (the number of episodes)")
## Worker Hyperparameters
# parser.add_argument("--num_workers", type=int, default=25, help="Number of workers")  
parser.add_argument("--sample_epi_num", type=int, default=1, help="Number of episodes each worker sampled each round")
# Evaluator Hyperparameters
parser.add_argument("--eval_per_step", type=float, default=5000, help="Evaluate the policy every 'eval_per_step'")
parser.add_argument("--evaluate_times", type=float, default=2, help="Evaluate times")
parser.add_argument("--save_gap", type=int, default=int(1e5), help="Save frequency")
# Network Hyperparameters
parser.add_argument("--num_layers", type=int, default=2, help="The number of the hidden layers of RNN")
parser.add_argument("--rnn_hidden_dim", type=int, default=128, help="The dimension of the hidden layer of RNN")
parser.add_argument("--embedding_dim", type=int, default=128, help="The dimension of the middle layer of GNN")
parser.add_argument("--gnn_output_dim", type=int, default=128, help="The dimension of the output layer of GNN")
parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="The dimension of the hidden layer of MLP")
# Tricks
parser.add_argument("--use_spectral_norm", type=bool, default=True, help="Trick 10:spectral normalization")
parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--use_agent_specific", type=float, default=True, help="Whether to use agent specific global state.")
parser.add_argument("--use_value_clip", type=float, default=True, help="Whether to use value clip.")
# Attention Hyperparameters
parser.add_argument("--att_n_heads", type=int, default=8, help="The number of heads of multi-head attention")
parser.add_argument("--att_dim_k", type=int, default=128, help="The dimension of keys/querys network")
parser.add_argument("--att_dim_v", type=int, default=128, help="The dimension of values network")
parser.add_argument("--att_output_dim", type=int, default=128, help="The output dimension of attention")
# DHGN Hyperparameters
parser.add_argument("--vertex_level_aggregator", type=str, default='mean', help='.')
parser.add_argument("--semantic_level_aggregator", type=str, default='mean', help='.')
parser.add_argument("--fcra_aggregator", type=str, default='mean', help='.')
parser.add_argument("--num_relation", type=int, default=3, help='.')
parser.add_argument('--depth', type=int, default=3, help='.')

algo_config = parser.parse_args()