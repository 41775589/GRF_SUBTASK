import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward to enhance offensive maneuvers and precision in finishing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            components["positioning_reward"][rew_index] = 0
            # Encourage advancing towards the enemy goal with the ball
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                x_position = o['ball'][0]
                if x_position > 0.5:  # Ball is in the opponent's half
                    components["positioning_reward"][rew_index] = (x_position - 0.5) * 2  # Scale reward to be more significant as it approaches the goal

            # Additional reward for precise ball control in critical areas
            if o['game_mode'] == 0 and o['ball_owned_team'] == 1:  # Normal game mode and ball owned by right team
                y_position = abs(o['ball'][1])
                if y_position < 0.1 and x_position > 0.8:  # Ball is central and very close to the goal
                    components["positioning_reward"][rew_index] += 1.0  # Big reward for precision in finishing

            # Combine components with game rewards
            reward[rew_index] += components["positioning_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Aggregate rewards for information
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky action counters for each action type
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                if action:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
