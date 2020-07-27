### Description of the Implemented Approach

The approach implemented in this repositroy is based on *reinforcement learning*, i.e. it is a machine learning approach in which an agent tries to improve his performance by interacting with the environment. In every step *t* of an episode, the agent chooses one of the actions mentioned above *A<sub>t</sub>* depending on the state *S<sub>t</sub>* he is in and observes the next state as well as a response in the form of a reward *R<sub>t</sub>* which is a real number in general. In this implementation, the algorithm is *value-based*, which means that the agent chooses the action by consulting an *action-value function*    
<p align="center"> <img src="https://latex.codecogs.com/svg.latex?&space;q_\pi(s,a)" /></p>

which is given as the expected return *G* when taking action *a* in state *s* and subsequently following policy <img src="https://latex.codecogs.com/svg.latex?&space;\pi" />:

<p align="center"> <img src="https://latex.codecogs.com/svg.latex?&space;q_\pi(s,a)=\left<G_t|S_t=s,A_t=a\right>_\pi=\left<\left.\sum_{k=0}^\infty\gamma^kR_{t+k+1}\right|S_t=s,A_t=a\right>_\pi" /></p>

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
