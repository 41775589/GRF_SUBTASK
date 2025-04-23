import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on dribbling and using the sprint action."""

    def __init__(self, env):
        super().__init__(env)
        self._player_was_sprinting = {}
        self._collected_checkpoints = {}
        self._checkpoint_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and clear tracking variables."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_checkpoints = {}
        self._player_was_sprinting = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on player control over the ball, dribbling, and sprinting."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for index, o in enumerate(observation):
            # Check if player is sprinting
            if o['sticky_actions'][8] == 1 and not self._player_was_sprinting.get(index, False):
                reward[index] += 0.1  # Reward sprint initiation
                components["checkpoint_reward"][index] += 0.1
                self._player_was_sprinting[index] = True
            elif o['sticky_actions'][8] == 0:
                self._player_was_sprinting[index] = False
            
            # Reward controlled ball movement
            if o['ball_owned_team'] == o['active']:
                dist_to_goal = abs(o['ball'][0] - 1)  # Distance to opponent's goal (assuming team 0 direction is towards +x)
                checkpoints = int(dist_to_goal * 10)
                previous_checkpoints = self._collected_checkpoints.get(index, 0)
                if checkpoints > previous_checkpoints:
                    reward_upgrade = (checkpoints - previous_checkpoints) * self._checkpoint_reward
                    reward[index] += reward_upgrade
                    components["checkpoint_reward"][index] += reward_upgrade
                    self._collected_checkpoints[index] = checkpoints

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
