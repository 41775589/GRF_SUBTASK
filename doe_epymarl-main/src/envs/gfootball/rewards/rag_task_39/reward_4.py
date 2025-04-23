import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effective and safe clearance of the ball under pressure."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_reward = 0.2
        self.pressure_threshold = 0.3  # Threshold for what is considered 'under pressure'

    def reset(self):
        """Resets the environment and counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the wrapper along with the environment state."""
        wrapper_state = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        to_pickle.update(wrapper_state)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the wrapper and the environment from the saved state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on the ball clearance under pressure."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            if o['game_mode'] in (2, 3, 4, 5):  # Consider pressure in relevant game modes only
                ball_owner = o['ball_owned_team']
                if ball_owner == 0:  # Ball is with the left team
                    opponent_distances = [np.linalg.norm(np.array([o['left_team'][o['active']]] 
                                            - pos)) for pos in o['right_team']]
                    min_distance = min(opponent_distances)
                    if min_distance < self.pressure_threshold:
                        reward[i] += self.clearance_reward
                        components["clearance_reward"][i] = self.clearance_reward
        return reward, components

    def step(self, action):
        """Steps environment, computes rewards, and add debugging info to returned information."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for idx, act in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{idx}"] = act
        return observation, reward, done, info
