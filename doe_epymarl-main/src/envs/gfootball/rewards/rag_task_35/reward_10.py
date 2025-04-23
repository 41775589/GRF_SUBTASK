import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward by focusing on strategic positioning and transitions between defense and attack."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positioning_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positioning_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['PositioningRewards'] = self.positioning_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.positioning_rewards = from_pickle['PositioningRewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, obs in enumerate(observation):
            player_x, player_y = obs['right_team'][obs['active']][:2]
            ball_x, ball_y = obs['ball'][:2]
            ball_distance = np.sqrt((player_x - ball_x) ** 2 + (player_y - ball_y) ** 2)
            
            components["positioning_reward"][rew_index] = 0
            
            # Reward strategic positioning and transition between defense and attack
            if obs['ball_owned_team'] == 1 and obs['ball_owned_player'] == obs['active']:
                # Encouraging moving forward with the ball towards the opponent's goal
                if player_x > 0:
                    components["positioning_reward"][rew_index] += 0.02
                # Encouraging passing or progressing towards the center from sides
                if abs(player_y) > 0.3:
                    components["positioning_reward"][rew_index] += 0.01

            # Defending strategies: positioning in relation to the ball and goal
            if obs['ball_owned_team'] == 0 or obs['ball_owned_team'] == -1:
                if player_x < 0 and ball_x < player_x:
                    # Reward defensive positioning between the ball and own goal
                    components["positioning_reward"][rew_index] += 0.01
            
            # Update the reward with positioning reward
            reward[rew_index] += components["positioning_reward"][rew_index]
        
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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
