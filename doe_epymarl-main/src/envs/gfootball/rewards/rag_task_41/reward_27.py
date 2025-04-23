import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """Wrapper to enhance attacking skills by promoting specialized training in creative offensive plays."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.successful_passes = 0
        self.attempts_on_goal = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.successful_passes = 0
        self.attempts_on_goal = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle.update({'successful_passes': self.successful_passes,
                          'attempts_on_goal': self.attempts_on_goal})
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.successful_passes = from_pickle.get('successful_passes', 0)
        self.attempts_on_goal = from_pickle.get('attempts_on_goal', 0)
        return from_pickle

    def reward(self, reward):
        """Calculates new rewards considering attacking progression metrics including passes and goal attempts."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "attack_reward": np.zeros(len(reward))}
        
        for obs_idx, obs in enumerate(observation):
            base_reward = reward[obs_idx]
            if obs['game_mode'] == 0 and obs['ball_owned_team'] == 0:  # Normal game mode and ball owned by the left team
                player_with_ball = obs['active']
                action_taken = obs['sticky_actions']
                
                # Reward for successful passes
                if 1 in [action_taken[0], action_taken[3], action_taken[4], action_taken[6]]:
                    self.successful_passes += 1
                    components["attack_reward"][obs_idx] += 0.01  # modest reward for pass attempts
                    
                # Additional reward if a shot on goal is taken
                if obs['ball'][0] > 0.5 and action_taken[9] == 1:  # Y-axis progression and dribble closer to opposite goal
                    self.attempts_on_goal += 1
                    components["attack_reward"][obs_idx] += 0.1  # higher reward for shots on goal

                # Update base reward based on added components
                reward[obs_idx] += (self.successful_passes + self.attempts_on_goal) * components["attack_reward"][obs_idx]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
