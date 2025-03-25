import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for learning offensive skills including passing, shooting, and dribbling."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_reward = 0.3
        self.shot_on_target_reward = 0.5
        self.successful_dribble_reward = 0.2
        self.passing_reward = 0.1
        self.shooting_reward = 0.2
        self.dribbling_reward = 0.15

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
                      "pass_completion_reward": 0.0,
                      "shot_on_target_reward": 0.0,
                      "successful_dribble_reward": 0.0}

        if observation is None:
            return reward, components

        # Components of reward based on the offensive actions
        for rew_index, o in enumerate(observation):
            # Pass completion
            if 'ball_direction' in o:
                if np.linalg.norm(o['ball_direction']) > 0:
                    components["pass_completion_reward"] = self.pass_completion_reward
                    reward[rew_index] += self.passing_reward * self.pass_completion_reward

            # Shot on target (approximated by ball moving significantly towards opponent's goal)
            if 'ball' in o:
                if o['ball'][0] > 0.5:  # Assuming right side is the opponent's side
                    components["shot_on_target_reward"] = self.shot_on_target_reward
                    reward[rew_index] += self.shooting_reward * self.shot_on_target_reward
            
            # Successful dribble (ball moving while in dribbling mode)
            if 'sticky_actions' in o:
                # Assume that action 9 corresponds to dribble
                if o['sticky_actions'][9] == 1 and np.linalg.norm(o['ball_direction']) > 0:
                    components["successful_dribble_reward"] = self.successful_dribble_reward
                    reward[rew_index] += self.dribbling_reward * self.successful_dribble_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                # Update count for sticky actions
                if action == 1:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
