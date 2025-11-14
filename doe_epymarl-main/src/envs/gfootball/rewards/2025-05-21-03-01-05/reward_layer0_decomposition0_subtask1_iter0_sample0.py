import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that modifies reward to enhance defensive robustness and 
    counter-attack efficiency from the flanks. """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Reset the environment and the sticky actions counter. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Serialize the critical environment data. """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Deserialize the environment data. """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """ Modify the reward based on specific defensive actions and positions. """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components[f"player_{rew_index}_position"] = []
            reward_enhancement = 0
            
            # Reward players being well-positioned in the flanks for counter-attacks
            if o['right_team'][o['active']][0] > 0:  # x position on the right of center line
                reward_enhancement += 0.1  # Reward for positioning on the attack-friendly side
                components[f"player_{rew_index}_position"].append(0.1)

            # Special focus on effective use of dribble and quick transitions
            if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # dribbling
                reward_enhancement += 0.05
                components[f"player_{rew_index}_position"].append(0.05)

            reward[rew_index] += reward_enhancement
        
        return reward, components

    def step(self, action):
        """ Process the action, modify and return the observation, reward, done, and info. """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)  # final cumulative reward
        for key, value in components.items():
            info[f"component_{key}"] = sum(value) if isinstance(value, list) else value
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        
        return observation, reward, done, info
