import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on strategic positioning,
       lateral and backward movement, and transitions from defense to counterattack."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Initialize components
            if "defensive_reward" not in components:
                components["defensive_reward"] = [0.0] * len(reward)

            # Encourage maintaining organization when opponent has ball
            if o['ball_owned_team'] == 1 and o['game_mode'] == 0:
                ball_x = o['ball'][0]
                player_x = o['right_team'][rew_index][0]
                # Reward for being behind the ball when the opposition owns it
                if player_x < ball_x:
                    components["defensive_reward"][rew_index] = 0.2

            # Encourage transitions based on position change and sticky actions (e.g., sprint toggled)
            if self.sticky_actions_counter[8] > 0:  # Sprint action
                delta_position = np.linalg.norm(o['right_team'][rew_index] - self.previous_pos[rew_index])
                # Reward for significant positional changes while sprinting
                if delta_position > 0.05:
                    components["defensive_reward"][rew_index] += delta_position * 0.5

            # Update previous positions for next call to reward function
            self.previous_pos[rew_index] = o['right_team'][rew_index]

            # Apply reward components
            reward[rew_index] += components["defensive_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
