import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A gym reward wrapper that focuses on enhancing defending strategies by
    rewarding proper tackling, efficient player movement for stopping opponents,
    and performing effective passes under pressure.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the sticky actions counter upon starting a new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state related to the reward wrapper.
        """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state from saved state information.
        """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Calculate the dense reward focusing on defending skills.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackling_reward": [0.0] * len(reward),
            "movement_reward": [0.0] * len(reward),
            "passing_pressure_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for effective tackling
            if o['game_mode'] in [3, 5, 6]:  # Represents defensive game modes like FreeKick, ThrowIn, Penalty
                components["tackling_reward"][rew_index] = 0.1
            
            # Reward for efficient movements in defending positions
            if o['ball_owned_team'] == 0 and o['left_team_active'][o['active']]:
                distance = np.linalg.norm(o['left_team'][o['active']] - o['ball'][:2])
                if distance < 0.1:  # closer to the ball indicates better positioning
                    components["movement_reward"][rew_index] = 0.05

            # Reward for passes under pressure
            if ('ball_owned_player' in o and o['ball_owned_player'] == o['active']):
                if np.any(o['sticky_actions'][5:7]):  # checking for bottom or bottom left/right actions (typically under pressure)
                    components["passing_pressure_reward"][rew_index] = 0.15

            # Aggregate rewards
            total_component_reward = sum([
                components["tackling_reward"][rew_index],
                components["movement_reward"][rew_index],
                components["passing_pressure_reward"][rew_index]
            ])
            reward[rew_index] += total_component_reward

        return reward, components

    def step(self, action):
        """
        Execute environment step and augment with customized rewards.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
