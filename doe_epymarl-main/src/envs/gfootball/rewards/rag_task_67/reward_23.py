import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on ball control skills during game transitions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality = 0.3
        self.dribble_reward = 0.2

    def reset(self):
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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_quality_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if a pass was successful in terms of short or long pass under pressure
            if o['game_mode'] in [2, 3, 4, 5]:  # Assume these modes involve passing
                if o['score'][1] > 0:  # Right team score is a proxy for successful play action
                    components["pass_quality_reward"][rew_index] = self.pass_quality
                    reward[rew_index] += components["pass_quality_reward"][rew_index]
            
            # Reward for dribbling while possessing the ball in a pressure situation
            if o['sticky_actions'][9] == 1 and o['ball_owned_team'] == o['active']:
                components["dribble_reward"][rew_index] = self.dribble_reward
                reward[rew_index] += components["dribble_reward"][rew_index]

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
