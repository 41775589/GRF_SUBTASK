import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for defensive responsiveness and interception.
    The defensive reward is determined based on the ability to block opponent progress
    towards the goal and successful ball interception.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Resets the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Retrieve the state along with the additional custom state information.
        """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state from the provided state data.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Modifies the reward based on the defense performance.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check if opposing team has control
            if o['ball_owned_team'] == 1:  # Supposing '1' is the opposing team
                player_pos = o['left_team'][o['active']]
                ball_pos = o['ball'][:2]

                # Calculate the Euclidian distance from the player to the ball
                dst_to_ball = np.sqrt(np.sum(np.square(np.array(player_pos) - np.array(ball_pos))))

                # Encourage being nearer to the ball (better positions for potential interceptions/tackles)
                if dst_to_ball < 0.1:
                    components["defensive_reward"][rew_index] = 1.0 - dst_to_ball  # Reward inversely related to distance
                else:
                    components["defensive_reward"][rew_index] = 0.0

                # Update the actual reward
                reward[rew_index] += components["defensive_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Takes a step in the environment, then applies reward modification.
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
