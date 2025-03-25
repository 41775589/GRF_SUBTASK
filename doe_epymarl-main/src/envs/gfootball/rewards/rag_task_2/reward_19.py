import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for tactical defensive play."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._position_checkpoints = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._position_checkpoints = {}
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_position_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for index in range(len(reward)):
            o = observation[index]
            components["defensive_position_reward"][index] = 0
            
            # Define specific defensive actions and reward strategy
            if o['game_mode'] == 0:  # Normal game mode
                if o['ball_owned_team'] == 0 and index not in self._position_checkpoints:
                    # Encourage maintaining ball control within defensive context
                    components['defensive_position_reward'][index] += 0.1
                if o['left_team'][index][0] < -0.5:  # Player closer to own goal
                    # Reward for positioning strategically during defensive scenario
                    components['defensive_position_reward'][index] += 0.2
                
                reward[index] += components['defensive_position_reward'][index]
                self._position_checkpoints[index] = True

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Capture rewards components for detailed insights during debugging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions usage info
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._position_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._position_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle
