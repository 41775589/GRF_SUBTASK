import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that encourages offensive strategies focusing on dribbling, passing, and shooting."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward tuning parameters
        self.pass_reward = 0.05
        self.dribble_reward = 0.1
        self.shooting_accuracy_bonus = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {} # Arbitrarily set, adjust based on actual state needs
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)  # Import state from the pickle
        # Assuming state saving/loading for Sticky Actions not necessary directly here, adjust if needed
        return from_pickle

    def reward(self, reward):
        # Access the observation
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "shot_on_target_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Consider different types of passes and dribbles
            if o['sticky_actions'][9] == 1:  # dribble action is active
                reward[rew_index] += self.dribble_reward
                components["dribble_reward"][rew_index] = self.dribble_reward

            # Reward successful passes
            if (o['game_mode'] in [5, 6]):  # Assuming modes 5 and 6 relate to successful high or long passes
                reward[rew_index] += self.pass_reward
                components["pass_reward"][rew_index] = self.pass_reward

            # Reward for shooting towards the goal
            if (o['ball'][0] > 0.9 and abs(o['ball'][1]) < 0.044 and o['ball_owned_team'] == 0):  # Near opponent's goal area with ball possession
                reward[rew_index] += self.shooting_accuracy_bonus
                components["shot_on_target_reward"][rew_index] = self.shooting_accuracy_bonus

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
            for i, action_item in enumerate(agent_obs['sticky_actions']):
                if action_item:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
