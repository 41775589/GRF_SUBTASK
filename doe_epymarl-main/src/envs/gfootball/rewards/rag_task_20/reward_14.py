import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on enhancing behaviors crucial for offensive strategies, optimizing team
    coordination, breaking opponent's defense, and adapting between immediate scoring and positioning strategies.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._num_zones = 5  # Divide the field into five zones for rewards
        self._passing_reward = 0.05
        self._positioning_reward = 0.03
        self._scoring_opportunity_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and reward mechanisms.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state of the reward wrapper.
        """
        to_pickle['CheckpointRewardWrapper_state'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Load the state of the reward wrapper.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_state']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Custom reward function to promote better offensive play by rewarding passing, positioning and potential scoring opportunities.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward),
                      "scoring_opportunity_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Give passing rewards based on the number of correct passes and ball control
            if o['ball_owned_team'] == 0 and o['sticky_actions'][9]:  # Team 0 has the ball and is dribbling
                components["passing_reward"][rew_index] = self._passing_reward
            
            # Positioning reward for moving towards strategic locations
            my_position = o['right_team'][o['active']]
            if my_position[0] > 0:  # Moving into the opponent's half
                components["positioning_reward"][rew_index] = self._positioning_reward
            
            ## Add reward for both getting the ball close to the goal and attempting shots
            if o['ball'][0] > 0.8:  # Ball is very close to opposite goal
                components["scoring_opportunity_reward"][rew_index] = self._scoring_opportunity_reward

            # Combine all rewards
            reward[rew_index] += (components["passing_reward"][rew_index] +
                                  components["positioning_reward"][rew_index] +
                                  components["scoring_opportunity_reward"][rew_index])

        return reward, components

    def step(self, action):
        """
        Take a step using the action and override reward calculation.
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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
