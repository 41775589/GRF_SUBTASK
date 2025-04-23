import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for strategic positioning and movement transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_checkpoints = [
            (-1, 0.42), (-1, -0.42), (1, 0.42), (1, -0.42), (0, 0)  # corners and center of the field
        ]
        # Reward for reaching each checkpoint zone
        self.position_checkpoint_reward = 0.05
        # Additional reward for changing from defensive to offensive mode and vice versa
        self.transition_reward = 0.1
        self.last_mode = None  # Keep track of the last strategic mode

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_mode = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'last_mode': self.last_mode,
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_mode = from_pickle['CheckpointRewardWrapper']['last_mode']
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "checkpoint_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward),
        }
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            player_pos = observation[rew_index]['right_team'][observation[rew_index]['active']]
            
            # Check checkpoints for positional rewards
            for checkpoint in self.position_checkpoints:
                distance = np.linalg.norm(np.array(checkpoint) - player_pos[:2])
                if distance < 0.1:  # If player is within a small radius of the checkpoint
                    components["checkpoint_reward"][rew_index] = self.position_checkpoint_reward
                    reward[rew_index] += components["checkpoint_reward"][rew_index]
            
            # Reward changing strategic positioning (defensive <-> offensive)
            current_mode = 'offensive' if observation[rew_index]['ball_owned_team'] == 1 else 'defensive'
            
            if self.last_mode is not None and self.last_mode != current_mode:
                components["transition_reward"][rew_index] = self.transition_reward
                reward[rew_index] += components["transition_reward"][rew_index]
            
            self.last_mode = current_mode
        
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
