import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that specializes in enhancing dribbling and dynamic positioning tactics."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_rewards = {}
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['dribble_rewards'] = self.dribble_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.dribble_rewards = from_pickle.get('dribble_rewards', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            designated_player = o['designated']
            active_player_actions = o['sticky_actions']
            dribble_started = active_player_actions[9] == 1  # Assuming 'action_dribble' is at position 9

            if designated_player == o['active'] and dribble_started:
                if rew_index not in self.dribble_rewards:
                    self.dribble_rewards[rew_index] = 0
                # Reward the player for each step they continue dribbling the ball
                components["dribbling_reward"][rew_index] += 0.05
                reward[rew_index] += components["dribbling_reward"][rew_index]

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
