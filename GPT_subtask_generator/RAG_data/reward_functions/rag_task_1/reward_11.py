import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized dense reward based on offensive maneuvers
    and dynamic adaptation during different game phases."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initializing count of sticky actions

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
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        # Initialize reward component multipliers
        pass_progress_component = 0.1
        attack_mode_component = 0.2

        # Adding rewards based on dynamic game states
        if observation['game_mode'] == 0:  # Normal play
            if observation['ball_owned_team'] == 0:  # Our agent has the ball
                components['possession_reward'] = [pass_progress_component * 5] * len(reward)
            else:
                components['possession_reward'] = [0.0] * len(reward)

        # Reward for successful aggressive maneuvers in the offensive mode
        if observation['game_mode'] in [2, 3, 4]:  # Set pieces close to opponent's goal
            components['attack_mode_reward'] = [attack_mode_component] * len(reward)

        # Calculate the final rewards
        for i, r in enumerate(reward):
            reward[i] += components['possession_reward'][i]
            if 'attack_mode_reward' in components:
                reward[i] += components['attack_mode_reward'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Append reward components to info for debugging purposes
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update action sticky info
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
