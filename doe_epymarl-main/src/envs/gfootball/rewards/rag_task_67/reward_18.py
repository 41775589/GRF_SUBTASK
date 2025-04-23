import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes ball control and passing from defense to attack."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_bonus = 0.05
        self.dribble_bonus = 0.03
        self.control_bonus = 0.02

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state_data = self.env.get_state(to_pickle)
        state_data['sticky_actions_counter'] = self.sticky_actions_counter
        return state_data

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_bonus": [0.0] * len(reward),
                      "dribble_bonus": [0.0] * len(reward),
                      "control_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if team controls the ball
            if o['ball_owned_team'] == 0:
                # Calculate bonus for maintaining control in pressured situations
                components["control_bonus"][rew_index] = self.control_bonus
                reward[rew_index] += components["control_bonus"][rew_index]

            # Examine the player's actions for passing
            if o['sticky_actions'][7] == 1:  # action_bottom_left corresponds to short pass logic
                components["passing_bonus"][rew_index] = self.passing_bonus
                reward[rew_index] += components["passing_bonus"][rew_index]

            # Dribbling while moving up the field under pressure
            if o['sticky_actions'][9] == 1:  # action_dribble
                components["dribble_bonus"][rew_index] = self.dribble_bonus
                reward[rew_index] += components["dribble_bonus"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Adding reward component sums to the info dictionary
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                if action_state == 1:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = action_state
        return observation, reward, done, info
