import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides rewards for successful sliding tackles near the defensive third during counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.num_tackles = 0  # to count successful tackles
        self.tackle_reward = 3.0  # reward for successful tackle
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and tackle counter."""
        self.num_tackles = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get current state with any additional checkpoint data."""
        to_pickle['num_tackles'] = self.num_tackles
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set state from pickle and recover additional checkpoint data."""
        from_pickle = self.env.set_state(state)
        self.num_tackles = from_pickle.get('num_tackles', 0)
        return from_pickle

    def reward(self, reward):
        """Customize reward function to include tackle rewards."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] != 0:
                # Only apply rewards during normal game play mode
                continue 

            if o['ball_owned_team'] == 0 and 'left_team' in o:
                # Detect if near defensive third and a sliding action is made
                curr_player_pos = o['left_team'][o['active']]
                if curr_player_pos[0] < -0.5 and o['sticky_actions'][6]:  # action_bottom (sliding)
                    # Award for successful tackle
                    reward[rew_index] += self.tackle_reward
                    self.num_tackles += 1
                    components["tackle_reward"][rew_index] = self.tackle_reward

        return reward, components

    def step(self, action):
        """Execute a step in the environment, recording and augmenting rewards."""
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
