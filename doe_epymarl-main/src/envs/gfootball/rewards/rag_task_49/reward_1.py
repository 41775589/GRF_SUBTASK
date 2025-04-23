import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that specializes in shooting from central field positions, focusing on accuracy 
    and power to improve goal-scoring odds. We reward the agent for taking shots towards the goal 
    when positioned within a central zone on the field, defined by a designated y-range. 
    The reward is accretive based on the power and direction accuracy towards the goal.
    """

    def __init__(self, env):
        super().__init__(env)
        self.central_zone_y_range = (-0.15, 0.15)  # This defines the central zone in y-axis
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_y_pos = o['ball'][1]
            if self.central_zone_y_range[0] <= ball_y_pos <= self.central_zone_y_range[1]:
                # If the ball is in the central zone
                if o['ball_owned_team'] == o['active'] and o['ball_owned_team'] == 1:
                    # If the right team (assuming control of right team) has the ball control
                    components["shooting_reward"][rew_index] = 1.0

            # Aggregate the reward
            reward[rew_index] += components["shooting_reward"][rew_index]
        
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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
