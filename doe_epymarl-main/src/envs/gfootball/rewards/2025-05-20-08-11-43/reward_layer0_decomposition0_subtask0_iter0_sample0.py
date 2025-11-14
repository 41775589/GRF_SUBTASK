import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for coordinated defensive actions and successful counter-attacks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Checking for team possession change and successful tackles or interceptions
            if o['game_mode'] not in (0, 1):  # excluding normal and kickoff modes for simplicity
                # Reward increases with successful defensive strategy implementation
                components.setdefault("defense_reward", [0.0] * len(reward))
                if o['ball_owned_team'] == 0 and o['left_team_roles'][o['active']] in [1, 2, 3] and reward[rew_index] < 0.5:
                    components["defense_reward"][rew_index] += 0.5 - reward[rew_index]  # Incentivize defensive actions
                    reward[rew_index] = 0.5
                elif o['ball_owned_team'] == 1 and o['right_team_roles'][o['active']] in [1, 2, 3] and reward[rew_index] < 0.5:
                    components["defense_reward"][rew_index] += 0.5 - reward[rew_index]
                    reward[rew_index] = 0.5

                # Rewarding counter-attack setups: ball transition from defense to an attacking position
                components.setdefault("counter_attack_setup", [0.0] * len(reward))
                if o['ball_owned_team'] == 0 and any(role in [8, 9] for role in o['left_team_roles']):
                    components["counter_attack_setup"][rew_index] += 1.0
                    reward[rew_index] += 1.0
                elif o['ball_owned_team'] == 1 and any(role in [8, 9] for role in o['right_team_roles']):
                    components["counter_attack_setup"][rew_index] += 1.0
                    reward[rew_index] += 1.0

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
