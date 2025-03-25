import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that increases the focus on efficient offensive tactics."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._successful_pass_checks = 0
        self._successful_pass_reward = 0.3
        self._shot_on_goal_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._successful_pass_checks = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['successful_pass_checks'] = self._successful_pass_checks
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._successful_pass_checks = from_pickle['successful_pass_checks']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_completion_reward": [0.0] * len(reward),
                      "shot_on_goal_reward": [0.0] * len(reward)}
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] in (0, 1):  # Normal play or kick-off
                # Reward pass completions that move the ball significantly towards opponent's goal
                if 'ball_owned_team' in o and o['ball_owned_team'] == 0: # if left team owns the ball
                    if 'action' in o and o['action'] == 'short_pass':
                        components["pass_completion_reward"][rew_index] = self._successful_pass_reward
                        reward[rew_index] += 1.5 * components["pass_completion_reward"][rew_index]
                        self._successful_pass_checks += 1
                
                # Reward shots on goal from close range
                if 'ball' in o and o['ball'][0] > 0.5:  # Ball is close to the right-side goal
                    if 'action' in o and o['action'] == 'shot':
                        components["shot_on_goal_reward"][rew_index] = self._shot_on_goal_reward
                        reward[rew_index] += 1.5 * components["shot_on_goal_reward"][rew_index]

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
