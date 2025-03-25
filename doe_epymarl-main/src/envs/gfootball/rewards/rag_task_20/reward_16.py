import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on offensive strategies, optimizing team play by favoring passes,
       positioning, ball control and scoring."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.05
        self.positioning_reward = 0.02
        self.shot_on_goal_reward = 0.1
        self.control_duration = np.zeros(5)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.control_duration.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter.fill(0)
        self.control_duration.fill(0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward),
            "shot_on_goal_reward": [0.0] * len(reward),
            "control_duration_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Rewards for passes
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and 'sticky_actions' in o:
                # Increase the control duration if the agent's team owns the ball
                self.control_duration[rew_index] += 1
                
                # Check if pass action happened
                if o['sticky_actions'][6] == 1 or o['sticky_actions'][7] == 1:
                    components["pass_reward"][rew_index] = self.pass_reward
                    reward[rew_index] += components["pass_reward"][rew_index]

            # Basic positioning reward
            if 'right_team' in o:
                team_pos = o['right_team']
                ball_pos = o['ball'][:2] if 'ball' in o else [0, 0]
                for teammate in team_pos:
                    distance = np.sqrt((teammate[0] - ball_pos[0])**2 + (teammate[1] - ball_pos[1])**2)
                    if distance < 0.1:  # rewarding close positioning to ball
                        components["positioning_reward"][rew_index] = self.positioning_reward
                        reward[rew_index] += components["positioning_reward"][rew_index]

            # Shot on goal
            if 'game_mode' in o and o['game_mode'] == 6:  # Mode 6 is Penalty
                components["shot_on_goal_reward"][rew_index] = self.shot_on_goal_reward
                reward[rew_index] += components["shot_on_goal_reward"][rew_index]

            # Longer ball control reward
            if self.control_duration[rew_index] > 50:  # arbitrary threshold for "long control"
                components["control_duration_reward"][rew_index] = 0.05 * self.control_duration[rew_index]
                reward[rew_index] += components["control_duration_reward"][rew_index]

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
