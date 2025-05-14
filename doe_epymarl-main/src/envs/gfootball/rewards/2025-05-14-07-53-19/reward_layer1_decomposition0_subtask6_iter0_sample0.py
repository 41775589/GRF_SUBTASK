import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that specifically focuses on enhancing dribbling skills."""

    def __init__(self, env):
        super().__init__(env)
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
        # Obtain the environment's observation
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "skill_improvement_reward": [0.0]}

        if observation is None:
            return reward, components
          
        o = observation[0]
        
        # Encourage dribbling effectively
        dribbling = o['sticky_actions'][9]  # Index 9 refers to dribbling
        has_ball = (o['ball_owned_player'] == o['active'] and
                    o['ball_owned_team'] == 0)  # Assumes the controlled team is 0 
                    
        # Adding an extra reward component based on dribbling while having the ball
        if dribbling and has_ball:
            reward[0] += 0.2  # Arbitrary small reward
            components["skill_improvement_reward"][0] = 0.2
      
        # Encourage stopping dribbling effectively
        not_dribbling = not dribbling
        if not_dribbling and has_ball:
            reward[0] += 0.1  # Smaller reward than dribbling
            components["skill_improvement_reward"][0] = 0.1

        # The final reward consists of the base environment reward plus additional skill reward
        reward[0] += components["base_score_reward"][0] + components["skill_improvement_reward"][0] 
        return [reward[0]], components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_state
        return observation, reward, done, info
