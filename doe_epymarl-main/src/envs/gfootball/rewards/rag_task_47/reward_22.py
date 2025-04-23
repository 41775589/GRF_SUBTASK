import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for executing sliding tackles effectively
    during counter-attacks and high-pressure defensive situations.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sliding_tackle_reward = 0.5
        self.tackle_distance_threshold = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment state and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the current environment state including the reward wrapper settings.
        """
        to_pickle['CheckpointRewardWrapper'] = {'sliding_tackle_reward': self.sliding_tackle_reward,
                                                'tackle_distance_threshold': self.tackle_distance_threshold}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the current environment state based on a loaded state.
        """
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['CheckpointRewardWrapper']
        self.sliding_tackle_reward = state_data['sliding_tackle_reward']
        self.tackle_distance_threshold = state_data['tackle_distance_threshold']
        return from_pickle

    def reward(self, reward):
        """
        Calculate the reward given the game's context, emphasizing effective sliding tackles.
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'sliding_tackle_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            # Base score reward, added directly
            base_reward = reward[rew_index]

            # Check if a sliding tackle action is being made
            if o['sticky_actions'][6]:  # Assuming position 6 in sticky_actions is for slide action
                ball_position = o['ball'][:2]  # Considering only x, y position
                player_position = o['right_team' if o['active'] in o['right_team_active'] else 'left_team'][o['active']]
                
                # Calculate distance to the ball
                distance_to_ball = np.linalg.norm(np.array(ball_position) - np.array(player_position[:2]))

                # If the sliding tackle is within a close range to the ball, reward it
                if distance_to_ball < self.tackle_distance_threshold:
                    components['sliding_tackle_reward'][rew_index] = self.sliding_tackle_reward
                    base_reward += self.sliding_tackle_reward

            reward[rew_index] = base_reward
        
        return reward, components

    def step(self, action):
        """
        Takes an action in the environment, modifies the reward based on defensive performance checks.
        """
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
