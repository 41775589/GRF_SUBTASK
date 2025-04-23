import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on maintaining strategic positioning 
    and switching effectively between offensive and defensive strategies.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        self.sticky_actions_counter = state['sticky_actions_counter']
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "positional_reward": np.zeros(len(reward))
        }

        for i, o in enumerate(observation):
            # Strategic positioning: moderate rewards for maintaining positions between defense and attack range.
            player_position = o['left_team'] if o['active'] < len(o['left_team']) else o['right_team']
            x_position = player_position[o['active']][0]

            # Defensive reward when towards own goal
            if x_position < -0.5:
                components['positional_reward'][i] += 0.02

            # Offensive reward when past midpoint towards opponent's goal
            if x_position > 0.5:
                components['positional_reward'][i] += 0.03

            # Adjust reward based on strategic switching
            if 'ball_owned_team' in o:
                if o['ball_owned_team'] == 0 and x_position < 0:  # Defensive when own team has ball
                    components['positional_reward'][i] += 0.01
                elif o['ball_owned_team'] == 1 and x_position > 0:  # Attacking when opponent has ball
                    components['positional_reward'][i] += 0.01

            reward[i] = components['base_score_reward'][i] + components['positional_reward'][i]

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
