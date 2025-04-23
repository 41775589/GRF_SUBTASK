import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper for emphasizing crossing and sprinting abilities, specifically designed for wingers.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and the sticky actions counter for a new episode.
        """
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Return the state of the environment with added custom checkpoint data.
        """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the environment with custom checkpoint data.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        """
        Custom reward function that emphasizes effective sprinting towards and accurate crossing from the wings.
        """
        observation = self.env.unwrapped.observation()

        components = {
            "base_score_reward": reward.copy(),
            "sprint_bonus": [0.0] * len(reward), 
            "crossing_bonus": [0.0] * len(reward)
        }

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage sprinting: check if the sprint action is active
            if o['sticky_actions'][8]:  # action_sprint index
                components["sprint_bonus"][rew_index] = 0.1
                reward[rew_index] += components["sprint_bonus"][rew_index]

            # Encourage accurate crossing from the wings
            if o['ball_owned_team'] == 0 and abs(o['ball'][1]) > 0.3: # ball closer to side-lines
                components["crossing_bonus"][rew_index] = 0.3
                reward[rew_index] += components["crossing_bonus"][rew_index]

        return reward, components

    def step(self, action):
        """
        Overriding the step function to incorporate the custom reward modifications.
        """
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
