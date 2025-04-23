import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a custom reward for offensive skills improvement in football."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        reward_modifiers = [0.0] * len(reward)  # Initialize reward modifiers for all agents
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        # Constants for reward tuning
        PASS_REWARD = 0.1
        SHOT_REWARD = 0.5
        DRIBBLE_SPRINT_BONUS = 0.05

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'sticky_actions' in o:
                actions = o['sticky_actions']
                # Actions: pass (short/long), shot, dribble, sprint
                if actions[6] >= 1 or actions[5] >= 1:  # short pass or long pass
                    reward_modifiers[rew_index] += PASS_REWARD
                if actions[4] >= 1:  # shot
                    reward_modifiers[rew_index] += SHOT_REWARD
                if actions[8] >= 1 or actions[9] >= 1:  # dribble or sprint
                    reward_modifiers[rew_index] += DRIBBLE_SPRINT_BONUS
                    
                # Update reward for each agent based on their actions and control of the ball
                if o['ball_owned_team'] == o['active'] and o['ball_owned_player'] == rew_index:
                    reward[rew_index] += reward_modifiers[rew_index]

                self.sticky_actions_counter = actions  # Update actions counter for info
            else:
                # Player doesn't have opportunities or didn't act
                reward_modifiers[rew_index] += 0

        components['offensive_skill_enhancement'] = reward_modifiers

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
                self.sticky_actions_counter[i] = action
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
