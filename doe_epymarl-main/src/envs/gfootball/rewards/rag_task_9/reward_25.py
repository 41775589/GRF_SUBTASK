import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that augments the reward in the Football environment to foster learning of offensive skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Actions associated with offensive play: Short Pass, Long Pass, Shot, Dribble and Sprint are assumed to be:
        # 5 - Short Pass
        # 6 - Long Pass
        # 7 - Shot
        # 8 - Dribble
        # 9 - Sprint
        self.offensive_actions = [5, 6, 7, 8, 9]
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.action_rewards = {
            5: 0.05,  # Short Pass
            6: 0.06,  # Long Pass
            7: 1.0,   # Shot
            8: 0.03,  # Dribble
            9: 0.02   # Sprint
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = copy.deepcopy(self.sticky_actions_counter)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "offensive_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o is None:
                continue
                
            # Check if any specified offensive action is taken
            offensive_bonus = 0
            for action_id in self.offensive_actions:
                if o['sticky_actions'][action_id]:
                    offensive_bonus += self.action_rewards[action_id]

            # Update reward with the offensive bonus
            reward[rew_index] += offensive_bonus
            components["offensive_bonus"][rew_index] = offensive_bonus
        
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
