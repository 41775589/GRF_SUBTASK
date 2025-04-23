import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward function aimed at developing a well-rounded defensive 
    strategy with quick transitions into counterattacks by improving positional awareness and responsiveness."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize counters for improved defensive positions and quick transitions
        self.defensive_positions_counter = np.zeros(5, dtype=int)  # Number of specific defensive positions
        self.transition_speed_bonus = 0.1  # Reward bonus for quick transitions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "defensive_positions_counter": self.defensive_positions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_positions_counter = from_pickle['CheckpointRewardWrapper']['defensive_positions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Defensive strategy improvement by maintaining specific defensive positions
            if o['game_mode'] == 3:  # Assuming game_mode 3 implies a defensive context like a free kick defense
                if self.defensive_positions_counter[rew_index] < 1:  # Reward only the first time for simplification
                    self.defensive_positions_counter[rew_index] += 1
                    components["defensive_reward"][rew_index] = 0.05  # Reward for being in a good defensive position
                    reward[rew_index] += components["defensive_reward"][rew_index]

            # Rewards for quick transitions from defense to counterattack
            if o['ball_owned_team'] == 0:  # Assuming the agent's team owns the ball
                ball_speed = np.linalg.norm(o['ball_direction'])
                if ball_speed > 0.01:  # Threshold for determining a quick transition (made up value for example)
                    components["transition_reward"][rew_index] = self.transition_speed_bonus
                    reward[rew_index] += components["transition_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
