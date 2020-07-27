# Banana Collector
Solution to the first project of Udacitys Reinforcement Learning Nanodegree

![video](https://user-images.githubusercontent.com/63595824/88564330-5f5c1080-d033-11ea-88d4-e0f4b3fdd98c.gif)

### Quick Installation

To set up the python environment and install all requirements for using this repository, follow the instructions given below:
1. Create and activate a new environment with Python 3:
    ```bash
    python3 -m venv /path/to/virtual/environment
    source /path/to/virtual/environment/bin/activate
    ```
2. Clone the Udacity repository and navigate to the `python/` folder to install the `unityagents`-package:
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    cd -
    ```
3. Download the Unity-environment from Udacity and unzip it:
    ```bash
    wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
    unzip Banana_Linux.zip
    rm Banana_Linux.zip
    ```

### Quick Start

After installing all requirements and activating the virtual environment training the agent can be started by executing

```bash
python main.py
```

Configurations like enabling the visualization or adjustments to the architecture is possible through the dictionaries defined in `main.py`.
During training a file called `performance.log` is created, which holds information about the current score, the average score of the last 100 episodes, the current loss of the neural network and the average loss of the last 100 episodes. Furthermore, if the agents variable `save_after` is set to a value larger than 0, after the given number of epochs the agents current parameters will be saved in a file with the agents name plus `_parameters.dat`, while the current weights of the agents models will be saved in a file with the agents name plus `_decision.model` or `_policy.model` respectively.

Running

```bash
python evaluate.py
```

allows to evaluate the performance of the saved agent of the given name. 


### Background Information
In this environment the player (or agent) has to navigate a world populated with yellow and blue bananas. The aim is to collect as many points as possible, where yellow bananas yield **`+1`** points and the blue ones **`-1`**. At every timestep, the player may choose to either go forward (**`0`**), go backward (**`1`**), turn left (**`2`**) or turn right (**`3`**). After a given time the game is over, hence making this task an episodic one. The environment is considered solved when the player achieves an average score of **`+13`** over 100 consecutive episodes.

For more information on the approach that was used to solve this environment, see `Report.md`.
