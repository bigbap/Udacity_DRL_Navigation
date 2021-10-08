from unityagents import UnityEnvironment

class Environment:
    def __init__(self, no_graphics=True, seed=0):
        self.env = UnityEnvironment(file_name="./environment/Banana.exe", no_graphics=no_graphics, seed=seed)
        
        # get the default brain
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        # number of actions
        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.brain.vector_observation_space_size
    
    def reset(self, train_mode=True):
        env_info = self.env.reset(train_mode=train_mode)[self.brain_name]
        state = env_info.vector_observations[0]
        done = False
        
        return state, done

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        state_prime = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        
        return reward, state_prime, done
    
    def close(self):
        self.env.close()

def episode(env, agent, train_mode=True, ep=None):
    score = 0
    state, done = env.reset(train_mode=train_mode)
    while not done:
        action = agent.act(state, ep=ep).astype(int)
      
        reward, state_prime, done = env.step(action)
        agent.step(state, action, reward, state_prime, done)

        score += reward
        state = state_prime
    
    return score