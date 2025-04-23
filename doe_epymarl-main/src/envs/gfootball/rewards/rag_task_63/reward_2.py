import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a custom reward tailored for training a goalkeeper. 
    The reward encourages the goalkeeper to intercept shots, make quick 
    decisions under pressure, and effectively communicate with defenders."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.blocked_shots = 0   # Counter for blocked shots
        self.pass_completion = 0 # Counter for completed passes under pressure
        self.communication_efficiency = 0 # Counter for effective communication

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.blocked_shots = 0
        self.pass_completion = 0
        self.communication_efficiency = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['blocked_shots'] = self.blocked_shots
        state['pass_completion'] = self.pass_completion
        state['communication_efficiency'] = self.communication_efficiency
        return state

    def set_state(self, state):
        self.blocked_shots = state['blocked_shots']
        self.pass_completion = state['pass_completion']
        self.communication_efficiency = state['communication_efficiency']
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_block_reward": [0.0] * len(reward),
                      "pass_completion_reward": [0.0] * len(reward),
                      "communication_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            rew = reward[idx]
            o = observation[idx]

            # Reward for blocking shots
            if o['game_mode'] == 6 and o['ball_owned_player'] == o['active']:
                self.blocked_shots += 1
                components["shot_block_reward"][idx] = 1.0

            # Reward for successful passes under pressure
            if o['ball_owned_team'] == 0 and np.any(o['sticky_actions'][6:10]) and o['score'][0] > o['score'][1]:
                self.pass_completion += 1
                components["pass_completion_reward"][idx] = 1.0

            # Reward for effective communication
            if np.all(o['left_team_active']) and not np.any(o['left_team_yellow_card']):
                self.communication_efficiency += 1
                components["communication_reward"][idx] = 1.0

            # Update total reward for this agent
            rew += components["shot_block_reward"][idx] + \
                   components["pass_completion_reward"][idx] + \
                   components["communication_reward"][idx]

            reward[idx] = rew

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
