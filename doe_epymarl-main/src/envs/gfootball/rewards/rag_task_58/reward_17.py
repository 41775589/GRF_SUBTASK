import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to handle the complex scenario of defense and transition to attack.
    """

    def __init__(self, env):
        super().__init__(env)
        self.last_defensive_action_successful = np.zeros(10, dtype=bool)
        self.ball_recovery_reward = 0.1
        self.secure_pass_reward = 0.05
        self.positional_improvement_reward = 0.02
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._last_ball_position = None

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.last_defensive_action_successful.fill(False)
        self._last_ball_position = None
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward),
                      "positional_improvement_reward": [0.0] * len(reward)}

        for idx, (rew, obs) in enumerate(zip(reward, observation)):
            ball_owned_team = obs['ball_owned_team']
            
            # Reward for recovering the ball from the opponent
            if ball_owned_team == 0 and not self.last_defensive_action_successful[idx]:
                components["defensive_reward"][idx] += self.ball_recovery_reward
                self.last_defensive_action_successful[idx] = True

            # Ensure correct transitioning from defense to attack
            if ball_owned_team == 0 and obs['active'] == obs['designated']:
                if 'right_team_direction' in obs and 'ball_direction' in obs:
                    ball_dir = obs['ball_direction']
                    player_dir = obs['right_team_direction'][obs['active']]

                    # Reward secure pass: aligning player direction with ball movement direction
                    if np.dot(ball_dir[:2], player_dir[:2]) > 0:
                        components["transition_reward"][idx] += self.secure_pass_reward

            # Reward for positional play improvement
            current_ball_pos = obs['ball'][0]
            if self._last_ball_position is not None and current_ball_pos > self._last_ball_position:
                improvement = current_ball_pos - self._last_ball_position
                components["positional_improvement_reward"][idx] += improvement * self.positional_improvement_reward
            
            self._last_ball_position = current_ball_pos

            # Calculate the final reward for this step
            reward[idx] += sum(components[c][idx] for c in components)

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions accounting
        current_obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in current_obs:
            for i, act in enumerate(agent_obs['sticky_actions']):
                if act:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return obs, reward, done, info
