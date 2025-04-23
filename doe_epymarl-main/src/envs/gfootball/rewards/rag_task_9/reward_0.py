import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward based on offensive skills like
    passing, shooting, dribbling, and sprinting.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_reward = 0.5
        self.pass_reward = 0.2
        self.dribble_reward = 0.1
        self.sprint_reward = 0.05

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
        components = {'base_score_reward': reward.copy(), 
                      'shot_reward': [0.0]*len(reward), 
                      'pass_reward': [0.0]*len(reward),
                      'dribble_reward': [0.0]*len(reward),
                      'sprint_reward': [0.0]*len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            if 'sticky_actions' in o:
                actions = o['sticky_actions']
                # Check and reward Shot (assumed index 9)
                if actions[9]:
                    components['shot_reward'][rew_index] = self.shot_reward

                # Check and reward Pass actions (assumed indices 2 for Long Pass and 1 for Short Pass)
                if actions[2] or actions[1]:
                    components['pass_reward'][rew_index] = self.pass_reward

                # Check and reward Dribble (assumed index 8)
                if actions[8]:
                    components['dribble_reward'][rew_index] = self.dribble_reward

                # Check and reward Sprint (assumed index 7)
                if actions[7]:
                    components['sprint_reward'][rew_index] = self.sprint_reward

                # Calculate the final reward for current step
                reward[rew_index] += sum([
                    components['shot_reward'][rew_index],
                    components['pass_reward'][rew_index],
                    components['dribble_reward'][rew_index],
                    components['sprint_reward'][rew_index]
                ])

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
