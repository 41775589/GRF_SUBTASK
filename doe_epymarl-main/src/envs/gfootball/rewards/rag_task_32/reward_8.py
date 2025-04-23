import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that applies crossing and sprinting task specific rewards for wingers,
    focusing on the accuracy and timing of crosses from the wings, including high-speed dribbling.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "crossing_reward": [0.0] * len(reward),
                      "sprinting_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            #
            # Reward for successful crosses from the wings
            #
            # Check for a potential successful cross (ball crosses towards the center from wing with intent)
            if o['ball_owned_team'] == 1 and o['ball'][1] > 0.2 and abs(o['ball'][0]) > 0.7:
                # Check if the cross is directed towards the center from the sides
                components["crossing_reward"][rew_index] = 1.0
                reward[rew_index] += components["crossing_reward"][rew_index]

            #
            # Reward agents for sprinting down the wings
            #
            if 'sticky_actions' in o:
                if o['sticky_actions'][8] == 1  and abs(o['left_team'][o['active']][0]) > 0.5:
                    # Reward sprinting down the wing (x-coordinate > 0.5 implies being near sideline)
                    components["sprinting_reward"][rew_index] = 0.5
                    reward[rew_index] += components["sprinting_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
