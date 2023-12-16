import sys, time, argparse
import gym
import numpy as np
from tqdm import tqdm
from lib.common_utils import TabularUtils
from lib.regEnvs import *



class Tabular_DP:
    def __init__(self, args):
        self.env = args.env
        self.env.reset()
        self.gamma = 0.99
        self.theta = 1e-5
        self.max_iterations = 1000
        self.nA = self.env.action_space.n
        self.nS = self.env.observation_space.n


    def compute_q_value_cur_state(self, s, value_func):
        """Compute the q value for current state s"""
        q_s = np.zeros(self.nA)
        # all each possible action a, get the action-value function
        for a in range(self.nA):
            # get the next state and reward
            prob_of_reaching_sprime = self.env.P[s][a][0]
            next_state = prob_of_reaching_sprime[1]
            reward = prob_of_reaching_sprime[2]

            next_state, reward, done, _ = self.env.step(a)
            # compute the q value
            q_s[a] = reward + self.gamma * value_func[next_state] * (1 - done)
            # reset the environment
            self.env.s = s
    
        return q_s


    def action_to_onehot(self, a):
        """ convert single action to onehot vector"""
        print(a)
        a_onehot = np.zeros(self.nA)
        a_onehot[a] = 1
        return a_onehot


    def value_iteration(self):
        # initialize the value function
        value_func = np.zeros(self.nS)
        delta = 0
        # get the initial reward
        reward = self.env.reset()
        print("Initial reward: ", reward)

        # iterate until convergence or max iterations
        for n_iter in range(1, self.max_iterations+1):
            print("Iteration: ", n_iter)

            # iterate over all states
            for s in range(self.nS):
                V_prev = value_func[s]
                Q = np.sum([self.compute_q_value_cur_state(s, value_func) for s in range(self.nS)], axis=0)
                # compute the q value for current state
                q_s = self.compute_q_value_cur_state(s, value_func)
                # update the value function
                value_func[s] = np.max(q_s)
            print("Value function: ", value_func)

            # we have to compute q[s] in each iteration from scratch
            # and compare it with the q value in previous iteration
            q_s = np.sum([self.compute_q_value_cur_state(s, value_func_prev) for s in range(self.nS)], axis=0)
            value_func = np.max(q_s, axis=1)
            # update delta
            delta = np.max(np.abs(value_func - value_func_prev))
            if delta < self.theta:
                break
        # compute the optimal policy
        policy_optimal = np.zeros((self.nS, self.nA))
        best_action = np.argmax(q_s, axis=1)
        for s in range(self.nS):
            policy_optimal[s] = self.action_to_onehot(best_action[s])

        # choose the optimal action and optimal value function in current state
        # output the deterministic policy with optimal value function
        V_optimal = value_func

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
            pass

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
    print(args)
    args.env = gym.make(args.env_name)
    tabularUtils = TabularUtils(args.env)
    print(tabularUtils)

    # test value iteration
    dp = Tabular_DP(args)
    print("================Running value iteration=====================")
    V_optimal, policy_optimal = dp.value_iteration()
    print("Optimal value function: ")
    print(V_optimal)
    print("Optimal policy: ")
    print(policy_optimal)
    
    # test SARSA
    # td = Tabular_TD(args)
    # Q_sarsa, policy_sarsa = td.sarsa()
    # print("Policy from sarsa")
    # print(tabularUtils.onehot_policy_to_deterministic_policy(policy_sarsa))

    # render
    tabularUtils.render(policy_optimal)
    # tabularUtils.render(policy_sarsa)