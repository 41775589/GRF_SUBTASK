import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards the agent for effectively using sprints to improve positioning defensively."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_rewards = np.zeros(2)  # Assuming a 2-agent setup for simplicity
        self.sprint_reward_amount = 0.05
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_rewards.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sprint_rewards'] = self.sprint_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sprint_rewards = from_pickle['sprint_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "sprint_reward": self.sprint_rewards.copy()
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            player_obs = observation[idx]
            if 'sticky_actions' in player_obs:
                if player_obs['sticky_actions'][8] == 1:  # Assuming index 8 corresponds to sprint action
                    self.sprint_rewards[idx] += self.sprint_reward_amount
                    reward[idx] += self.sprint_rewards[idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
