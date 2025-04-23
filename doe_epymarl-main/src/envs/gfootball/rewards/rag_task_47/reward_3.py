import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering sliding tackles during defensive counter-attacks in the defensive third."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Sticky actions track whether the corresponding action is active

    def reset(self):
        # Reset sticky actions and other stateful elements if needed
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        # Access the latest observation to manipulate the reward based on current state
        observation = self.env.unwrapped.observation()
        
        components = {
            "base_score_reward": reward.copy(),
            "tackle_position_reward": [0.0, 0.0]
        }

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            components["tackle_position_reward"][rew_index] = 0.0  # Initialize component for this index

            # Check for successful defense actions in the defensive third
            if 'game_mode' in o and o['game_mode'] != 0:
                # When game mode indicates a potential defensive scenario (e.g., throw-ins, corners in the defensive third)
                ball_pos = o['ball'][0]  # x position of the ball
                if ball_pos < -0.5:
                    # Close to own goal (left third of the field)
                    if 6 in o['sticky_actions']:
                        # 'sliding' action is active
                        components["tackle_position_reward"][rew_index] = 0.5  # Reward for well-timed sliding

            # Update reward value for the current player
            reward[rew_index] += components["tackle_position_reward"][rew_index]

        return reward, components

    def step(self, action):
        # Execute action in the environment
        observation, reward, done, info = self.env.step(action)
        
        # Augment reward according to custom components
        reward, components = self.reward(reward)
        
        # Inject modified rewards and their components into the info dict for debug purposes
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        
        # Reset sticky actions
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
