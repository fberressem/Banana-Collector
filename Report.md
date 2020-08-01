### Description of the Implemented Approach

The approach implemented in this repository is based on *reinforcement learning*, i.e. it is a machine learning approach in which an agent tries to improve his performance by interacting with the environment. In every step *t* of an episode, the agent chooses one of the actions mentioned above *A<sub>t</sub>* depending on the state *S<sub>t</sub>* he is in and observes the next state as well as a response in the form of a reward *R<sub>t</sub>* which is a real number in general. In this implementation, the algorithm is *value-based*, which means that the agent chooses the action by consulting an *action-value function*    
<p align="center"> <img src="https://latex.codecogs.com/svg.latex?&space;q_\pi(s,a)" /></p>

which is given as the expected return *G* when taking action *a* in state *s* and subsequently following policy <img src="https://latex.codecogs.com/svg.latex?\pi" />:

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?q_\pi(s,a)=\left<G_t|S_t=s,A_t=a\right>_\pi=\left<\left.\sum_{k=0}^\infty\gamma^kR_{t+k+1}\right|S_t=s,A_t=a\right>_\pi" /></p>

In this equation <img src="https://latex.codecogs.com/svg.latex?&space;0\leq\gamma<1" /> is a discounting factor that describes how valuable future rewards are compared to present ones and ensures that the expected return *G* is finite as long as the reward sequence *{ R<sub>k</sub> }* is bounded.

To learn this action-value function the agent makes an (typically very poor) initial guess for the action-value function and updates it according to an *update rule*. The update rule chosen here is a *1-step Temporal-Difference (TD) Learning* update rule, which for a sequence of *(state, reward, action, next state, next action)* *(S<sub>t</sub>, R<sub>t</sub>, A<sub>t</sub>, S<sub>t+1</sub>, A<sub>t+1</sub>)* reads

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?\begin{align*}q_\pi(S_t,A_t)&=q_\pi(S_t,A_t)+\alpha\left[R_t+\gamma\left<q_\pi(S_{t+1},\cdot)\right>_{\mathcal{P}_{\bar{\pi}}(\cdot|A_{t+1})}-q_{\pi}(S_{t},A_{t})\right]\\&=q_\pi(S_{t},A_{t})+\alpha\left[R_{t}+\gamma\left(\sum_{\tilde{a}}q_\pi(S_{t+1},\tilde{a})\mathcal{P}_{\bar{\pi}}(\tilde{a}|A_{t+1})\right)-q_{\pi}(S_{t},A_{t})\right]\end{align*}\" /></p>

Now in these equations there is some explaining to do as I introduced some non-standard notation. First of all, <img src="https://latex.codecogs.com/svg.latex?\alpha" /> is the so called *learning-rate*, which is typically chosen quite small to improve the convergence of the updates by decreasing the fluctuations. The (newly-introduced) notation of <img src="https://latex.codecogs.com/svg.latex?\left<q_\pi(S_{t+1},\cdot)\right>_{\mathcal{P}_{\bar{\pi}}(\cdot|A_{t+1})}\" /> can be understood as follows: Given a *target policy* <img src="https://latex.codecogs.com/svg.latex?\bar{\pi}\" /> that does not have to be equal to the *behavior policy* <img src="https://latex.codecogs.com/svg.latex?\pi\" /> as well as the next action taken by the agent *A_<sub>t+1</sub>* the value of <img src="https://latex.codecogs.com/svg.latex?\mathcal{P}_{\bar{\pi}}(\tilde{a}|A_{t+1})\" /> corresponds to the probabilty  of or weight factor for using the action-value of <img src="https://latex.codecogs.com/svg.latex?\tilde{a}\" /> in the estimation of the future expected returns. Basically this allows for both on- and off-policy updates (due to the flexibility in <img src="https://latex.codecogs.com/svg.latex?\bar{\pi}\" />) as well as the usage of the *Sarsa*-update rule. The most common 1-step TD update rules can be recovered as follows:
1. Sarsa by setting
<p align="center"> <img src="https://latex.codecogs.com/svg.latex?\mathcal{P}_{\bar{\pi}}(\tilde{a}|A_{t+1})=\delta_{\tilde{a},A_{t+1}}\" /></p>
2. Q-Learning by setting
<p align="center"> <img src="https://latex.codecogs.com/svg.latex?\mathcal{P}_{\bar{\pi}}(\tilde{a}|A_{t+1})=\delta_{\tilde{a},argmax_{a}(q(S_{t+1},a))" /></p>
3. Expected Sarsa by setting
<p align="center"> <img src="https://latex.codecogs.com/svg.latex?\mathcal{P}_{\bar{\pi}}(\tilde{a}|A_{t+1})=\pi(\tilde{a}|S_{t+1})" /></p>

In principle, this leads to more flexibility in the choice of the update rule, as one can basically encode any on- and off-policy 1-step TD update rule that depends either on the action-values of the next state, *S<sub>t+1</sub>*, or on the the action that was actually taken next, *A<sub>t+1</sub>*.

In the final implementation, a plain Q-Learning update rule was used, even though also other update rules would be possible. The performance was enhanced by using a *dueling network architecture* was used. In this architecture, the calculation of action-values is split into the calculation of state-values and advantages of the different actions, such that

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?q_\pi(s,a)=\tilde{v}_\pi(s)+Adv_\pi(s,a)\" /></p>

where <img src="https://latex.codecogs.com/svg.latex?\tilde{v}_\pi(\cdot)\" /> is a (modified) state-value function, i.e. it describes the value of a given state. As this splitting cannot be done unambiguously (as for example you can always subtract a value from <img src="https://latex.codecogs.com/svg.latex?\tilde{v}_\pi(s)\" /> and add it to <img src="https://latex.codecogs.com/svg.latex?Adv_\pi(s,a)\" />) the convention

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?\tilde{v}_\pi(s)=v_\pi(s)+\left<Adv_\pi(s,\cdot)\right>\" /></p>

was chosen, where <img src="https://latex.codecogs.com/svg.latex?v_\pi(\cdot)\" /> now corresponds to the real state-value function.

Another improvement to the algorithm was the usage of *prioritized replay buffers*: Replay buffers are storages for sequences observed by the agent while interacting with the environment. The memories in the replay buffer can be used to train the agent while not actually interacting with the environment by reusing previous observations. This leads to a more efficient usage of experiences, in turn making learning more efficient. Besides that, it typically leads to better generalization, as the agent is trained on potentially old memories, so that it does not forget about previous experiences and so that it is subject to a larger variety of different situations. Replay buffers can be considered a very simple "model of the environment" in that they assume that memories from the past are representative for the underlying dynamics of the environment. Reusing old memories does not have to be done uniformly but can be prioritized, for example by taking the previous TD error into account when choosing the experiences to relive, hence the name *prioritized* replay buffers.

To make training more stable, *fixed Q-targets* were used. In this technique, the agent uses two neural networks of the same architecture, where one is network not trained via gradient descent but whose weigths <img src="https://latex.codecogs.com/svg.latex?\omega\" /> are updated using soft updates:

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?\omega=\tau\omega^{\prime}+(1-\tau)\omega\" /></p>

here, <img src="https://latex.codecogs.com/svg.latex?\omega^{\prime}" /> are the weights of the neural network that is trained using some form of gradient descent.

### Network Architecture and Hyperparameters

The neural networks used here were simple *dense networks*, i.e. they consist of fully connected layers only. The architecture was as follows:

- Input layer of size `37` (corresponding to the 37 dimensions of the state)
- Hidden layer with `64` neurons and `ReLU`-activation
- Hidden layer with `32` neurons and `ReLU`-activation
- Hidden layer with `16` neurons and `ReLU`-activation
- Output layer with `4` neurons (corresponding to the 4 possible actions) without activation function, i.e. linear

The hyperparameters used are given in the table below:

| Hyperparameter   |      Value      |
|----------|:-------------:|
| Q-target parameter <img src="https://latex.codecogs.com/svg.latex?\tau" /> |  0.1  |
| Discount factor <img src="https://latex.codecogs.com/svg.latex?\gamma" /> |    0.95   |
| Start value of <img src="https://latex.codecogs.com/svg.latex?\epsilon" /> | 1.0 |
| Decay rate of <img src="https://latex.codecogs.com/svg.latex?\epsilon" /> | 0.99 |
| Minimum value of <img src="https://latex.codecogs.com/svg.latex?\epsilon" /> | 0.01 |
| Batchsize | 2^11 |
| Size of replay buffer | 2^20 |
| Number of replays per learning phase | 2 |
| Learning rate <img src="https://latex.codecogs.com/svg.latex?\alpha" /> | 0.001 |


### Results

The reinforcement agent as configured in `main.py` reaches the required average score (averaged over the last 100 episodes) of **`+13`** after about 400 episodes, but its average score drops immediately after. The agent reaches its peak performance after about 2300 episodes with an stonishing average score of about **`16`**, however it is not able to keep its performance so high throughout the rest of the training. 

![results](https://github.com/fberressem/Banana-Collector/blob/master/Results.png)

In general, the agent performs quite well and solves the task very quickly, however its performance could be more stable. In the following, there are some suggestions on how to improve on that.

### Future Improvements

There are some improvements that may be implemented in the future:

1. To make training more stable, one might change the fixed Q-targets part to *double Q-Learning* such that the choice of actions while interacting with the environment is done using either one of the neural networks with the update rules being applied accordingly. In this case, there would be no dedicated *target network* anymore, while training should still be improved, as the networks would still be regularizing each other.

2. One might modify the rewards the agent sees to steer its behavior, e.g. by punishing collecting blue bananas with **`-2`** points instead of **`-1`** to discourage collecting blue bananas.

3. The prioritization in the replay buffer could take into account how successful (or unsuccessful) the episodes they stem from were. This might improve stability of learning as learning about very good or very bad actions would be emphasized.

4. The final experiences from all the episodes could be disregarded when learning, as they are not representative for the value of the state. That is, the agent does not know about whether the next state is a terminal one or not when calculating its expected return, so that when training on this memory, it skews the value of the state, making training less stable. Hence, the performance might be improved by disregarding all next-to-terminal states in the training phase.

5. Better architectures might be found using grid-searching or more sophisticated methods, like evolutionary algorithms. This, however, can be a very time-consuming task and does not yield as many interesting insights into the internal processes of the reinforcement learning agent, so that one should probably rather look into the other proposed improvements. 
