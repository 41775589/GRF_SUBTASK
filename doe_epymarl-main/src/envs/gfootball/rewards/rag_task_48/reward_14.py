import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that enhances the reward signal for successfully executing high passes from midfield
    that create direct scoring opportunities.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.midfield_thresholds = np.array([-0.2, 0.2])  # approximated midfield range on the x-axis
        self.high_pass_reward = 0.3  # Reward for a valid high pass
        self.scoring_opportunity_reward = 1.0  # Reward if the pass leads to a scoring opportunity
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the sticky action counts in the state.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the sticky action counts from the saved state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Adjust the original reward based on the successful execution of high passes from midfield positions.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0, 0.0],
                      "scoring_opportunity_reward": [0.0, 0.0]}
                      
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            ball_position = o['ball'][0]  # X coordinate of the ball position
            game_mode = o['game_mode']
            
            # Check if the ball is in midfield and a high pass is executed
            if self.midfield_thresholds[0] <= ball_position <= self.midfield_thresholds[1] and o['sticky_actions'][8] == 1:
                # High pass executed within the midfield area
                components["high_pass_reward"][rew_index] = self.high_pass_reward
                reward[rew_index] += components["high_pass_reward"][rew_index]

                # Check the transition of the game mode to a potential scoring opportunity
                if game_mode in [1, 6]:  # Kickoff or penalty indicates potential direct scoring chance
                    components["scoring_opportunity_reward"][rew_index] = self.scoring_opportunity_reward
                    reward[rew_index] += components["scoring_opportunity_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Execute a step in the environment, adjust the reward and append reward components to the info.
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
