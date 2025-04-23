import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward system focused on goalkeeper training
    tasks such as shot stopping, effective communication, and ball distribution.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_checkpoints = 5
        self.communicative_checkpoints = 3
        self.distribution_rewards = 0.2
        self.shot_stopping_rewards = 0.3
        self.communication_rewards = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        return self.env.set_state(from_pickle)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "positional_reward": [0.0] * len(reward),
            "communication_reward": [0.0] * len(reward),
            "distribution_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
             
            # Evaluate shot stopping ability based on positioning and ball proximity
            goalie_position = o['left_team'][o['active']]
            ball_position = o['ball'][:2]
            if np.linalg.norm(goalie_position - ball_position) < 0.3:
                components["positional_reward"][rew_index] = self.shot_stopping_rewards
                reward[rew_index] += components["positional_reward"][rew_index]

            # Rewards for communication, indicated by switching focus or quick response to ball possession change
            if o['game_mode'] in [3, 4, 6]:  # modes that might require quick restarts or communication
                components["communication_reward"][rew_index] = self.communication_rewards
                reward[rew_index] += components["communication_reward"][rew_index]

            # Ball distribution rewards when clearing the ball effectively under pressure
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                if np.abs(ball_position[0]) > 0.7:  # closer to goal boundaries
                    components["distribution_reward"][rew_index] = self.distribution_rewards
                    reward[rew_index] += components["distribution_reward"][rew_index]

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
