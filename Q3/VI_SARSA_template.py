import sys, time, argparse
import gym
import numpy as np
from tqdm import tqdm
from lib.common_utils import TabularUtils
from lib.regEnvs import *



class Tabular_DP:
    def __init__(self, args):
        self.env = args.env
        self.gamma = 0.99
        self.theta = 1e-5
        self.max_iterations = 1000
        self.nA = self.env.action_space.n
        self.nS = self.env.observation_space.n


    def compute_q_value_cur_state(self, s, value_func):
        q_s = np.zeros(self.nA)
        # all each possible action a, get the action-value function
        #TODO
        return q_s


    def action_to_onehot(self, a):
        """ convert single action to onehot vector"""
        #TODO

        return a_onehot


    def value_iteration(self):
        # initialize the value function
        value_func = np.zeros(self.nS)
        for n_iter in range(1, self.max_iterations+1):
          #TODO
          # we have to compute q[s] in each iteration from scratch
          # and compare it with the q value in previous iteration

          # choose the optimal action and optimal value function in current state
          # output the deterministic policy with optimal value function


        return V_optimal, policy_optimal



class Tabular_TD:
    def __init__(self, args):
        self.env = args.env
        self.num_episodes=10000
        self.gamma = 0.99
        self.alpha = 0.05
        self.env_nA = self.env.action_space.n
        self.env_nS = self.env.observation_space.n
        self.tabularUtils = TabularUtils(self.env)
    

    def sarsa(self):
        """sarsa: on-policy TD control"""
        Q = np.zeros((self.env_nS, self.env_nA))
        for epi in range(self.num_episodes):
            #TODO

        return Q, greedy_policy



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', dest='env_name', type=str,
                        # default="FrozenLake-Deterministic-v1", 
                        default="FrozenLake-Deterministic-8x8-v1",
                        choices=[""])
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    args.env = gym.make(args.env_name)
    tabularUtils = TabularUtils(args.env)

    # test value iteration
    dp = Tabular_DP(args)
    print("================Running value iteration=====================")
    V_optimal, policy_optimal = dp.value_iteration()
    print("Optimal value function: ")
    print(V_optimal)
    print("Optimal policy: ")
    print(policy_optimal)
    
    # test SARSA
    td = Tabular_TD(args)
    Q_sarsa, policy_sarsa = td.sarsa()
    print("Policy from sarsa")
    print(tabularUtils.onehot_policy_to_deterministic_policy(policy_sarsa))

    # render
    tabularUtils.render(policy_optimal)
    tabularUtils.render(policy_sarsa)