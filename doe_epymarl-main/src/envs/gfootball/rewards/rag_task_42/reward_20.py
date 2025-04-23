import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for mastering midfield dynamics including 
    enhanced coordination under pressure and strategic repositioning for 
    offense and defense transitions.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define checkpoints on the field transversally for strategic positioning
        self.midfield_checkpoints = np.linspace(-0.8, 0.8, 5) # 5 strategic regions across the midfield
        self.checkpoint_history = {}  # To remember checkpoint crossings
        self.checkpoint_reward = 0.1  # Reward increment for each checkpoint

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoint_history = {}  
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.checkpoint_history
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.checkpoint_history = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_pos = o['left_team'][o['active']] if o['active'] < len(o['left_team']) else o['right_team'][o['active'] - len(o['left_team'])]

            # Find closest midfield checkpoint to the active player's x-coordinate
            closest_checkpoint = np.argmin(np.abs(self.midfield_checkpoints - active_player_pos[0]))
            checkpoint_key = f"{rew_index}_{closest_checkpoint}"

            # Reward players crossing into different checkpoint regions not yet visited this episode
            if checkpoint_key not in self.checkpoint_history:
                self.checkpoint_history[checkpoint_key] = True
                reward[rew_index] += self.checkpoint_reward
                components["checkpoint_reward"][rew_index] = self.checkpoint_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self._reset_sticky_actions(obs)
        return observation, reward, done, info

    def _reset_sticky_actions(self, obs):
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                if action == 1:
                    self.sticky_actions_counter[i] += 1
