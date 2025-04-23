import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes perfecting standing tackles during gameplay, which focuses on player control and limiting penalties."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.active_tackle_reward = 0.05  # Reward for actively attempting tackles
        self.successful_tackle_reward = 1  # High reward for successful tackle
        self.penalty_for_foul = -1  # Penalty for causing a foul during tackle
        self._actions_taken = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._actions_taken.clear()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._actions_taken
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._actions_taken = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        new_reward = reward.copy()
        if observation is None:
            return new_reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            action_tackle_info = o.get('sticky_actions', np.zeros(10))

            # Reward for simply attempting tackles where applicable
            components.setdefault("tackle_attempt_reward", [0.0] * len(reward))
            if action_tackle_info[6]:  # Assuming index 6 represents a tackle action
                new_reward[rew_index] += self.active_tackle_reward
                components["tackle_attempt_reward"][rew_index] = self.active_tackle_reward

            # Checking tackling success and fouls from game_mode information
            game_mode = o.get('game_mode', 0)
            components.setdefault("tackle_success_reward", [0.0] * len(reward))
            components.setdefault("foul_penalty_reward", [0.0] * len(reward))
            
            if game_mode == 6:  # Assuming game mode 6 indicates a foul 
                new_reward[rew_index] += self.penalty_for_foul
                components["foul_penalty_reward"][rew_index] = self.penalty_for_foul
            
            if self._successful_tackle(o, rew_index):
                new_reward[rew_index] += self.successful_tackle_reward
                components["tackle_success_reward"][rew_index] = self.successful_tackle_reward

        return new_reward, components

    def _successful_tackle(self, observation, index):
        """Method to determine if a tackle was successful based on environment observation."""
        ball_owner = observation.get('ball_owned_player', -1)
        # Assuming successful tackle changes ball ownership to the agent's team (0 for left, 1 for right)
        return ball_owner in [observation.get('active', -1), observation.get('designated', -1)]

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
