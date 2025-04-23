import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focusing on short passing under pressure and ball retention."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._pass_precision_coef = 0.2
        self._ball_retention_coef = 0.1
        self._proximity_to_ball = {}
        self._last_owned = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._proximity_to_ball = {}
        self._last_owned = {}
        return self.env.reset()

    def reward(self, reward):
        # Monitoring the observations coming from the environment
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "pass_precision": [0.0] * len(reward),
                      "ball_retention": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage active passing and touch under defensive pressure
            if o['ball_owned_team'] == 1:  # Assuming control by right team
                # Reward for holding the ball under pressure (ball retention)
                if o['ball_owned_player'] == o['active'] and self._last_owned.get(rew_index, -1) == o['active']:
                    self._proximity_to_ball[rew_index] = (
                        1 - max(abs(o['ball'][0] - 1), abs(o['ball'][1]))
                    )
                self._last_owned[rew_index] = o['active']

                # Work with sticky actions to judge passing and movement quality
                if 'action_bottom_left' in o['sticky_actions'] or 'action_bottom_right' in o['sticky_actions']:
                    # Assuming successful pass if movement directions are favorable
                    components["pass_precision"][rew_index] = self._pass_precision_coef * self._proximity_to_ball.get(rew_index, 0)

            # Ball retention metric beyond just possession, ensuring it is within a contested area
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                components["ball_retention"][rew_index] += self._ball_retention_coef

            reward[rew_index] += components["pass_precision"][rew_index] + components["ball_retention"][rew_index]

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
