import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards collaboration between offensive players, specifically midfielders
    in creating play opportunities and strikers in finalizing these opportunities.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.num_checkpoint_zones = 5
        self.completion_rewards = np.zeros(self.num_checkpoint_zones, dtype=float)
        self.midfielder_reward = 0.5
        self.striker_reward = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._reset_rewards()

    def _reset_rewards(self):
        """
        Resets the reward counters.
        """
        self.completion_rewards.fill(0.0)

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self._reset_rewards()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.completion_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.completion_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "positional_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, base_rew in enumerate(reward):
            o = observation[rew_index]

            active_player_pos = o['right_team'][o['active']] if o['ball_owned_team'] == 1 else o['left_team'][o['active']]
            is_midfielder = o['right_team_roles'][o['active']] in [5, 6] if o['ball_owned_team'] == 1 else o['left_team_roles'][o['active']] in [5, 6]
            is_striker = o['right_team_roles'][o['active']] == 9 if o['ball_owned_team'] == 1 else o['left_team_roles'][o['active']] == 9
            
            # Check if the midfielder progresses towards the opponent's half or sets up a play
            if is_midfielder and active_player_pos[0] > 0.5:
                completion_zone = min(int((active_player_pos[0] + 1) * self.num_checkpoint_zones / 2), self.num_checkpoint_zones - 1)
                if self.completion_rewards[completion_zone] == 0:
                    self.completion_rewards[completion_zone] = self.midfielder_reward
                    components["positional_reward"][rew_index] += self.midfielder_reward
                    reward[rew_index] += self.midfielder_reward

            # Check if a striker finishes the set-up with a goal
            if is_striker and (base_rew > 0 or o['score'][1] > o['score'][0]):  # Assume right team is the attacking team
                components["positional_reward"][rew_index] += self.striker_reward
                reward[rew_index] += self.striker_reward

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
