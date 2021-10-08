# Navigation
This is the P1 project for the Deep Reinforcement Learning Nanodegree on https://www.udacity.com/.

The goal of this project is to create an agent that can be trained to navigate a large world, collecting as many yellow bananas as possible while ignoring the blue ones.

Banana world is an episodic and continuous environment where yellow and blue bananas are randomly dropped into the environment. The player receives a reward of **-1 for walking over a blue banana** and **1 for walking over a yellow one**. An example of the trained agent can be seen in the gif below.

![trained agent](https://i.imgur.com/0JG7ud8.gif)

The state space is comprised of 37 dimensions. Included are the agent's velocity as well as ray-based perception of objects in the agents view.

The action space is comprised of 4 discrete actions. Specifically:
 - `0` = move forward
 - `1` = move backward
 - `2` = turn left
 - `3` = turn right

The problem is solved once the agent has achieved an average score of 13 over the last 100 episodes.

## Getting started
This was built and **tested on 64 bit, Windows 10 only**.

 1. Clone this project. `git clone https://github.com/bigbap/Udacity_DRL_Navigation`
 2. Download banana environment from https://drive.google.com/file/d/16JEchv_Mo58pk3Qy-d2VRCsyvm_wDbMv/view?usp=sharing
 3. extract the contents of the downloaded environment into the cloned project directory.

You will need to install Anaconda from https://www.anaconda.com/. Once Anaconda is installed, open up **PowerShell** and type the following commands to create a new environment:

 1. `conda create --name navigation python=3.6`. This will create a new environment called **navigation** where we will install the dependencies.
 2. `conda activate navigation`. This will activate the environment we just created.

Now we can install the dependencies:

 1. `pip3 install unityagents`. Should be version 0.4.0
 2. `pip3 install torch`. Should be version 1.9.1

## Training the model
Open `Navigation.ipynb` with **Jupyter Notebook** and follow the instructions to train a new agent.

If you want to watch the trained agent playing, open `NavigationPlay.ipynb` with **Jupyter Notebook** and follow the instructions.

*Note: if you get any errors, please ensure that you have followed all the instructions in the **Getting started** section. Pay special attention to the dependencies and their versions. If you are not sure what packages you have installed, use `conda list` to list all the installed packages and their versions for the activated environment*

## Running the tests

Open up **PowerShell** and type `python -m unittest -v` to run the unit tests.