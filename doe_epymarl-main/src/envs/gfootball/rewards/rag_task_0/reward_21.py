import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for developing offensive strategies in football."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_reward = 0.1
        self.pass_reward_long = 0.2
        self.pass_reward_high = 0.15
        self.shoot_accuracy_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "pass_reward_long": [0.0] * len(reward),
                      "pass_reward_high": [0.0] * len(reward),
                      "shoot_accuracy_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # When player executes a dribble and retains the ball.
            if o['sticky_actions'][9] == 1 and o['ball_owned_team'] == 0:
                components["dribble_reward"][rew_index] = self.dribble_reward
                reward[rew_index] += components["dribble_reward"][rew_index]

            # Rewards for passing strategies.
            game_mode = o['game_mode']
            if game_mode in (2, 3):  # Long pass or High pass modes
                if game_mode == 2:  # Long pass
                    components["pass_reward_long"][rew_index] = self.pass_reward_long
                    reward[rew_index] += components["pass_reward_long"][rew_index]
                elif game_mode == 3:  # High pass
                    components["pass_reward_high"][rew_index] = self.pass_reward_high
                    reward[rew_index] += components["pass_reward_high"][rew_index]

            # Reward player for shooting accurately.
            if game_mode == 6:  # Assuming game mode 6 relates to shooting towards goal.
                components["shoot_accuracy_reward"][rew_index] = self.shoot_accuracy_reward
                reward[rew_index] += components["shoot_accuracy_reward"][rew_index]

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
