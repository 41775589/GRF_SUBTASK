import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that enhances focus on strategic positioning, lateral and backward movement, and quick transition from defense to counterattack, aimed at improving defensive resilience."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Setup waypoint rewards for lateral and backward movements and for transitions from defense to offense
        self.positional_reward_thresholds = np.linspace(-0.9, 0.9, 10) # 10 key positions across the field
        self.position_rewards_collected = {i: False for i in range(len(self.positional_reward_thresholds))}
        self.transition_reward = False

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards_collected = {i: False for i in range(len(self.positional_reward_thresholds))}
        self.transition_reward = False
        return self.env.reset()
        
    def get_state(self, to_pickle):
        to_pickle['position_rewards_collected'] = self.position_rewards_collected
        to_pickle['transition_reward'] = self.transition_reward
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.position_rewards_collected = from_pickle['position_rewards_collected']
        self.transition_reward = from_pickle['transition_reward']
        return from_pickle

    def reward(self, reward):
        updated_rewards = reward.copy()
        components = {"base_score_reward": updated_rewards.copy(),
                      "positional_rewards": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return updated_rewards, components

        for rew_index, o in enumerate(observation):
            # Calculate positional rewards based on lateral and backward positioning on the field
            player_x_position = o['left_team'][o['active']][0] if o['ball_owned_team'] == 0 else o['right_team'][o['active']][0]

            # Award only once per episode as player crosses each positional threshold from left to right
            for i, threshold in enumerate(self.positional_reward_thresholds):
                if not self.position_rewards_collected[i] and player_x_position > threshold:
                    components["positional_rewards"][rew_index] += 0.01
                    updated_rewards[rew_index] += 0.01
                    self.position_rewards_collected[i] = True

            # Transition reward: from having ball in own half to moving it to the opponent's half
            if not self.transition_reward and o['ball'][0] < 0 and o['ball_owned_team'] == 0 and player_x_position > 0:
                components["transition_reward"][rew_index] += 0.3
                updated_rewards[rew_index] += 0.3
                self.transition_reward = True

        return updated_rewards, components

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
