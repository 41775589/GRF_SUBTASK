import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper focusing on defensive skills such as positioning, interception,
    marking, and tackling to prevent the opponent from scoring. Rewards actions
    like Sliding (action 6), Stop-Dribble (tracked via sticky actions), and Stop-Moving (action 0).
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track stickiness for actions

    def reset(self):
        """
        Reset the wrapper state and environment state.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Serialize the state of the wrapper along with the environment.
        """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Deserialize the state of the wrapper along with the environment.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Enhance the reward function focusing on defense-oriented behaviors.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "defensive_skill_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            components["defensive_skill_reward"][rew_index] = 0.0
            # Reward for Sliding
            if o['sticky_actions'][6]:
                components["defensive_skill_reward"][rew_index] += 0.05

            # Reward for persistent Stop-Dribble or No Movement if opponent is in ball possession
            if o['ball_owned_team'] == 1:  # If opponent has the ball
                if not o['sticky_actions'][9]:  # Not dribbling
                    components["defensive_skill_reward"][rew_index] += 0.1
                if o['sticky_actions'][0]:  # No movement
                    components["defensive_skill_reward"][rew_index] += 0.05

            # Update final reward
            reward[rew_index] += components["defensive_skill_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Execute a step, enhance the reward using the customized reward function, and update info dict.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
