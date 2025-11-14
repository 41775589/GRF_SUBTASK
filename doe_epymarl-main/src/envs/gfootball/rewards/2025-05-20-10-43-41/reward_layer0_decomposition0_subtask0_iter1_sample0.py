import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances reward for left back and right back players based on their ability to transition effectively
    from defense to offense, focusing on their positioning, intercept skills, and initiation of counter-attacks.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.intercept_reward = 0.3
        self.transition_reward = 0.4
        self.counter_attack_initiation_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper_sticky_actions_counter'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "interception_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward),
            "counter_attack_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            obs = observation[idx]
            player_role = obs['left_team_roles'][obs['active']]

            # Check for roles: left and right back (assuming e_PlayerRole_LB=2, e_PlayerRole_RB=3)
            if player_role not in [2, 3]:
                continue

            # Check for successful intercepts
            if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:
                components["interception_reward"][idx] = self.intercept_reward
                reward[idx] += components["interception_reward"][idx]

            # Check for effective transition to offense
            if 'ball_direction' in obs:
                direction_to_goal = 1 if obs['left_team'][obs['active']][0] > 0 else -1
                if obs['ball_direction'][0] * direction_to_goal > 0:
                    components["transition_reward"][idx] = self.transition_reward
                    reward[idx] += components["transition_reward"][idx]

            # Reward for initiating a counter-attack
            if obs['game_mode'] == 0 and obs['ball_owned_team'] == 0: # Normal game mode
                if np.linalg.norm(obs['ball'] - obs['right_team'][obs['ball_owned_player']]) < 0.2:
                    components["counter_attack_reward"][idx] = self.counter_attack_initiation_reward
                    reward[idx] += components["counter_attack_reward"][idx]

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
