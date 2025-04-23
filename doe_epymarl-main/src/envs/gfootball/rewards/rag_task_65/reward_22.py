import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward to stress the importance of passing and shooting with precision."""

    def __init__(self, env):
        super().__init__(env)
        self.num_zones = 5  # Number of zones to trigger passing and shooting
        self.rewards_pass_shoot = np.zeros(5, dtype=float)
        self.pass_shoot_reward = 0.2
        self.scenario_triggered = np.zeros(5, dtype=bool)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.rewards_pass_shoot.fill(0)
        self.scenario_triggered.fill(False)
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'rewards_pass_shoot': self.rewards_pass_shoot,
            'scenario_triggered': self.scenario_triggered
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.rewards_pass_shoot = from_pickle['CheckpointRewardWrapper']['rewards_pass_shoot']
        self.scenario_triggered = from_pickle['CheckpointRewardWrapper']['scenario_triggered']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "precision_scenario_reward": [0.0] * len(reward)}

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' not in o or o['ball_owned_team'] != 0:
                continue

            # Analyzing game scenario for precision and decision making
            ball_pos = o['ball'][0]  # X position of ball

            # Check for triggering scenarios
            if 0.2 <= ball_pos <= 0.4 and not self.scenario_triggered[0]:
                self.rewards_pass_shoot[0] += self.pass_shoot_reward
                self.scenario_triggered[0] = True
            elif 0.4 < ball_pos <= 0.6 and not self.scenario_triggered[1]:
                self.rewards_pass_shoot[1] += self.pass_shoot_reward
                self.scenario_triggered[1] = True
            elif 0.6 < ball_pos <= 0.8 and not self.scenario_triggered[2]:
                self.rewards_pass_shoot[2] += self.pass_shoot_reward
                self.scenario_triggered[2] = True
            elif 0.8 < ball_pos <= 1.0 and not self.scenario_triggered[3]:
                self.rewards_pass_shoot[3] += self.pass_shoot_reward
                self.scenario_triggered[3] = True
            
            # Apply accumulated reward from scenario triggering
            reward[rew_index] += sum(self.rewards_pass_shoot)
            components["precision_scenario_reward"][rew_index] = sum(self.rewards_pass_shoot)
            
            # Reset scenarios at the end of a play or possession change
            if o.get('game_mode', 0) != 0:
                self.scenario_triggered.fill(False)

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
            if 'sticky_actions' in agent_obs:
                self.sticky_actions_counter += agent_obs['sticky_actions']
        return observation, reward, done, info
