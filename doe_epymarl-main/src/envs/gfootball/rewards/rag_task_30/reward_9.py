import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on strategic positioning, lateral and backward movement, and accelerating
    the transition from defense to counterattack, aimed at strengthening the team's defensive resilience."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reset()

    def reset(self):
        """Resets the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Gets the state of the environment and adds this wrapper's state to it."""
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the state of the environment and updates this wrapper's state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Encourages strategic defense and rapid switch to counterattack by modifying the reward based on the game state."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "strategic_positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            num_owned_by_left_team = sum(o['left_team_active'])
            num_owned_by_right_team = sum(o['right_team_active'])

            # Encourage keeping the ball and strategic backward movement when in own half
            if o['ball_owned_team'] == o['active']:
                ball_x_pos = o['ball'][0]
                own_half = True if ball_x_pos < 0 else False
                if own_half:
                    components["strategic_positioning_reward"][rew_index] = 0.2
                    reward[rew_index] += components["strategic_positioning_reward"][rew_index]

            # Encourage lateral movement and positioning across the field
            player_x_pos = o['left_team'][o['active']][0] if o['ball_owned_team'] == 0 else o['right_team'][o['active']][0]
            strategic_positions = [-0.8, -0.4, 0, 0.4, 0.8]
            if any(abs(player_x_pos - pos) <= 0.1 for pos in strategic_positions):
                components["strategic_positioning_reward"][rew_index] += 0.3
                reward[rew_index] += components["strategic_positioning_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Performs an environment step and augments the reward information."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
