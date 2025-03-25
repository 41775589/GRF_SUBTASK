import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward specifically targeting teamwork and defensive strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = np.zeros(3)
        self.ball_control_rewards = {}
        self._defensive_reward = 0.05
        self._team_coordination_bonus = 0.1
        self._num_defensive_positions = 5

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.previous_ball_position.fill(0)
        self.ball_control_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'previous_ball_position': self.previous_ball_position,
            'ball_control_rewards': self.ball_control_rewards
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        saved_state = from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = saved_state['sticky_actions_counter']
        self.previous_ball_position = saved_state['previous_ball_position']
        self.ball_control_rewards = saved_state['ball_control_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "team_coordination_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for maintaining ball control while in defensive positions
            if ('ball_owned_team' in o and o['ball_owned_team'] in [0, 1] and
                'ball' in o and np.linalg.norm(o['ball'][:2] - self.previous_ball_position[:2]) < 0.05):
                components["defensive_reward"][rew_index] = self._defensive_reward
                reward[rew_index] += components["defensive_reward"][rew_index]

            # Reward for teamwork: distribution of players in strategic defensive positions
            if ('left_team' in o and 'right_team' in o):
                team_positions = o['left_team'] if o['ball_owned_team'] == 0 else o['right_team']
                if len(team_positions) >= self._num_defensive_positions:
                    avg_pos = np.abs(np.mean(team_positions[:, 1]))  # Mean Y positions
                    if avg_pos < 0.2:  # Ensures that team players are strategically spread out
                        components["team_coordination_reward"][rew_index] = self._team_coordination_bonus
                        reward[rew_index] += components["team_coordination_reward"][rew_index]

        self.previous_ball_position = o['ball']
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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
