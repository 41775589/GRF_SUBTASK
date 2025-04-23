import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adjusts the reward function to train a goalkeeper. The reward focuses on
    shot stopping, quick decision making for ball distribution under pressure, and effective
    communication with defenders.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_stop_reward = 0.3  # Reward for stopping a shot
        self.distribution_reward = 0.2  # Reward for effective ball distribution
        self.communication_reward = 0.1  # Reward for communication

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Initialize or reset our checkpoint tracking from saved state
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "shot_stop_reward": [0.0] * len(reward),
                      "distribution_reward": [0.0] * len(reward),
                      "communication_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Assume goalkeeper index is 0
            if o['active'] == 0:  # Check if the active player is the goalkeeper
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == 0:
                    # Goalkeeper owns the ball, hence distribution decision point
                    components["distribution_reward"][rew_index] = self.distribution_reward
                if o['game_mode'] in {6}:  # Close to a penalty or close save situation
                    components["shot_stop_reward"][rew_index] = self.shot_stop_reward
                # Placeholder for communication assessment
                # (e.g., might include position, directives to defenders, etc.)
                # This could be more complex involving more state observation
                components["communication_reward"][rew_index] = self.communication_reward
                
            for key in components:
                reward[rew_index] += components[key][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
