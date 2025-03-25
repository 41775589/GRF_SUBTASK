import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on quick decision-making and 
    efficient ball handling to initiate counter-attacks immediately after recovering possession.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Placeholder for state setting logic if any needed
        return from_pickle

    def reward(self, reward):
        """
        Enhances the reward function by providing positive feedback for quick ball control and counter-attacking 
        immediately after possession recovery.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]
            if o['game_mode'] == 0:  # Normal gameplay
                if (o['ball_owned_team'] == 0 and self.sticky_actions_counter[8]):  # Sprinting with ball possession
                    bonus_reward = 0.1
                    reward[i] += bonus_reward
                    components[f"possession_sprint_reward_{i}"] = bonus_reward
                if (o['ball_owned_team'] == 0 and o['ball_direction'][0] > 0.02):  # Positive X direction at good speed
                    bonus_reward = 0.05
                    reward[i] += bonus_reward
                    components[f"counter_attack_reward_{i}"] = bonus_reward

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
