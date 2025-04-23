import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized attacking training rewards."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goal_distance_thresholds = np.linspace(0.2, 1, 10)[::-1]  # Array of distance thresholds from the opponent's goal.
        self.checkpoints_collected = [False] * 10  # Track checkpoints for distance covered towards goal.
        self.attack_intensity_reward = 0.1  # Reward for intense offensive plays

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoints_collected = [False] * 10
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'checkpoints_collected': self.checkpoints_collected
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.checkpoints_collected = from_pickle['CheckpointRewardWrapper']['checkpoints_collected']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward),
                      "attack_intensity": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Reward players for moving closer to the opponent's goal with the ball
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:  # Ensure the active agent is from the right team and has the ball
                ball_pos_x = o['ball'][0]
                for i, threshold in enumerate(self.goal_distance_thresholds):
                    if ball_pos_x > threshold and not self.checkpoints_collected[i]:
                        components["checkpoint_reward"][rew_index] += self.attack_intensity_reward
                        self.checkpoints_collected[i] = True

            # Add incentives for aggressive play in attacking phases
            if o['game_mode'] in [1, 4, 6]:  # KickOff, Corner, Penalty
                components["attack_intensity"][rew_index] += self.attack_intensity_reward

            # Update reward for current player
            reward[rew_index] += components["checkpoint_reward"][rew_index] + components["attack_intensity"][rew_index]

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
