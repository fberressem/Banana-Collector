# Banana Collector
Solution to the first project of Udacitys Reinforcement Learning Nanodegree

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



### Background Information
In this environment the player (or agent) has to navigate a world populated with yellow and blue bananas. The aim is to collect as many points as possible, where yellow bananas yield **`+1`** points and the blue ones **`-1`**. At every timestep, the player may choose to either go forward (**`0`**), go backward (**`1`**), turn left (**`2`**) or turn right (**`3`**). After a given time the game is over, hence making this task an episodic one. The environment is considered solved when the player achieves an average score of **`+13`** over 100 consecutive episodes.

For more information on the approach that was used to solve this environment, see `Report.md`.
