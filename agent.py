import torch
import torch.nn.functional as F
import torch.nn as nn

import torch.optim as optim

import numpy as np

from collections import deque

import utils
import model
import time

class Agent():
    """ Agent to interact with environment"""
    def __init__(self, agent_dict={}, model_dict={}):
        """ Initialize Agent object

        Params
        ======
            agent_dict(dict): dictionary containing parameters for agent
            model_dict(dict): dictionary containing parameters for agents model
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_episodes = agent_dict.get("num_episodes", 10000)
        self.save_after = agent_dict.get("save_after", -1)
        self.name = agent_dict.get("name", "banana_collector")

        self.gamma = agent_dict.get("gamma", 0.9)


        self.epsilon = agent_dict.get("epsilon_start", 1.0)
        self.epsilon_decay = agent_dict.get("epsilon_decay", 0.9)
        self.epsilon_min = agent_dict.get("epsilon_min", 0.1)

        self.tau = agent_dict.get("tau", 0.1)

        self.num_replays = agent_dict.get("num_replays", 1)

        self.criterion = nn.MSELoss()

        memory_size = agent_dict.get("memory_size", 2**14)
        batchsize = agent_dict.get("batchsize", 2**10)
        replay_reg = agent_dict.get("replay_reg", 0.0)

        self.replay_buffer = utils.PrioritizedReplayBuffer(memory_size, batchsize, epsilon=replay_reg)

        self.decision_model = model.Model(model_dict).to(self.device)
        self.policy_model = model.Model(model_dict).to(self.device)

        self.optimizer = optim.Adam(self.decision_model.parameters(), lr=1E-3)

        utils.copy_model(self.decision_model, self.policy_model, tau=1.0)
        seed = agent_dict.get("seed", 0)

        torch.manual_seed(seed)
        np.random.seed(seed)

    def targetPolicy(self, q_values):
        """ Returns chosen action and probabilites. Currently set to greedy policy

        Params
        ======
            q_values(array): action values
        """
        return utils.greedy(q_values)


    def decisionQValues(self, state):
        """ Returns action values according to behavior policy as well as state as torch.Tensor

        Params
        ======
            state(array): current state
        """
        state_t = torch.from_numpy(state).float().to(self.device)
        q_values = self.decision_model(state_t)
        return q_values, state_t

    def policyQValues(self, state):
        """ Returns action values according to target policy as well as state as torch.Tensor

        Params
        ======
            state(array): current state
        """
        state_t = torch.from_numpy(state).float().to(self.device)
        q_values = self.policy_model(state_t)
        return q_values, state_t

    def act(self, state):
        """ Choose action according to behavior policy. Returns action values, state as torch.Tensor, chosen action and probabilites

        Params
        ======
            state(array): current state
        """
        q_values, state_t = self.decisionQValues(state)
        action, probs = utils.epsilon_greedy(q_values, self.epsilon)
        return q_values, state_t, action, probs

    def qValuesTarget(self, q_values, reward, action, q_values_next, done):
        """ Returns target action values and TD target

        Params
        ======
            q_values(torch.Tensor): action values
            reward(float): observed reward
            action(int): chosen action
            q_vals_next(torch.Tensor): action values of next state
            done(bool): flag indicating terminal state
        """
        q_values_target = q_values.clone()
        td_target = utils.target(reward, self.gamma, q_values_next, self.targetPolicy, done).float()
        q_values_target[action] = td_target
        return q_values_target, td_target

    def trainDecisionModel(self, q_values, reward, action, q_values_next, done):
        """ Train behavior model. Returns TD error and loss

        Params
        ======
            q_values(torch.Tensor): action values
            reward(float): observed reward
            action(int): chosen action
            q_vals_next(torch.Tensor): action values of next state
            done(bool): flag indicating terminal state
        """
        q_values_target, td_target = self.qValuesTarget(q_values, reward, action, q_values_next, done)

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, q_values_target.detach())
        loss.backward()
        self.optimizer.step()
        utils.copy_model(self.decision_model, self.policy_model, tau=self.tau)

        td_error = (td_target - q_values[action]).item()

        return td_error, loss


    def learn(self):
        """ Train behavior model """
        for t in range(self.num_replays):
            states_t, actions, rewards, next_states_t, dones = self.replay_buffer.sample()
            states_t = torch.stack(states_t).to(self.device)
            next_states_t = torch.stack(next_states_t).to(self.device)
                

            q_values = self.decision_model(states_t)
            q_values_next = self.policy_model(next_states_t)
            q_values_target = q_values.clone()

            for i in range(len(actions)):
                action = actions[i]
                q_values_target[i, action] = utils.target(rewards[i], self.gamma, q_values_next[i], self.targetPolicy, dones[i])
                
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, q_values_target.detach())
            loss.backward()
            self.optimizer.step()

        utils.copy_model(self.decision_model, self.policy_model, tau=self.tau)


    def save_state(self):
        """ Save current state of agent """
        torch.save(self.decision_model, self.name + "_decision.model")
        torch.save(self.policy_model, self.name + "_policy.model")
        f_state = open(self.name + "_parameters.dat", "w")
        f_state.write("gamma = " + str(self.gamma) + "\n")
        f_state.write("epsilon = " + str(self.epsilon) + "\n")
        f_state.write("epsilon_decay = " + str(self.epsilon_decay) + "\n")
        f_state.write("epsilon_min = " + str(self.epsilon_min) + "\n")
        f_state.write("tau = " + str(self.tau) + "\n")
        f_state.write("num_replays = " + str(self.num_replays))
        f_state.close()

    def load_state(self):
        """ Load current state of agent """
        for line in open(self.name + "_parameters.dat", "r"):
            param, val = line.split(" = ")
            if "gamma" in param:
                self.gamma = float(val)
            elif "epsilon_decay" in param:
                self.epsilon_decay = float(val)
            elif "epsilon_min" in param:
                self.epsilon_min = float(val)
            elif "epsilon" in param:
                self.epsilon = float(val)
            elif "tau" in param:
                self.tau = float(val)
            elif "num_replays" in param:
                self.num_replays = int(val)
        
        self.decision_model = torch.load(self.name + "_decision.model")
        self.policy_model = torch.load(self.name + "_policy.model")


    def run(self, env):
        """ Train agent in environment env

        Params
        ======
            env(Env): environment to train agent in
        """
        recent_scores = deque(maxlen=100)
        recent_losses = deque(maxlen=100)

        f = open("performance.log", "w")
        f.write("#Score\tAvg.Score\tLoss\tAvg.Loss\n")

        for e in range(self.num_episodes):
            score = 0.0
            state = env.reset()
            done = False

            losses = []

            while not done:
                q_values, state_t, action, probs = self.act(state)
                next_state, reward, done, _ = env.step(action)
                score += reward
                q_values_next, next_state_t = self.policyQValues(next_state)
                td_error, loss = self.trainDecisionModel(q_values, reward, action, q_values_next, done)

                self.replay_buffer.append(state_t, action, reward, next_state_t, done, td_error)
                state = next_state
                losses.append(loss.item())

            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

            recent_scores.append(score)
            print("Iteration %i: score: %f\taverage_score: %f" % (e, score, np.mean(recent_scores)))

            avg_loss = sum(losses)/len(losses)
            recent_losses.append(avg_loss)

            f.write(str(score)+ "\t" + str(np.mean(recent_scores)) + "\t" + str(avg_loss) + "\t" + str(np.mean(recent_losses)) + "\n")

            self.learn()

            f.flush()
        
            if e == self.save_after:
                self.save_state()

        f.close()


    def evaluate(self, env, num_episodes=100, delay=0.0):
        """ Evaluate agent performance in environment env

        Params
        ======
            env(Env): environment to train agent in
            num_episodes(int): number of episodes to run
            delay(float): time delay to make visualization slower
        """
        recent_scores = deque(maxlen=num_episodes)

        for e in range(num_episodes):
            score = 0.0
            state = env.reset()
            done = False

            while not done:
                q_values, state_t, action, probs = self.act(state)
                next_state, reward, done, _ = env.step(action)
                
                time.sleep(delay)

                score += reward
                state = next_state

            recent_scores.append(score)
            print("Iteration %i: score: %f\taverage_score: %f" % (e, score, np.mean(recent_scores)))


            time.sleep(10*delay)

        print("#"*20)
        print("Overall average: %f\tlast 100 episodes: %f" % (np.mean(recent_scores), np.mean(np.array(recent_scores)[-100:])))


