import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoints_collected = [0, 0, 0]
        self.num_checkpoints = 10

    def reset(self):
        """ Reset sticky actions and checkpoints for the new episode. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoints_collected = [0, 0, 0]
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Save the state with checkpoints collected. """
        to_pickle['checkpoint_collected'] = self.checkpoints_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Restore the state with checkpoints collected. """
        from_pickle = self.env.set_state(state)
        self.checkpoints_collected = from_pickle['checkpoint_collected']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": np.zeros(3)}
        
        if observation is None:
            return reward, components

        # Rewards for strategic positioning, shooting, and passing based on customized scenarios
        for i in range(len(reward)):
            agent_obs = observation[i]

            # Encourage possession near the opponent's goal:
            if agent_obs['ball_owned_team'] == 1 and agent_obs['right_team'][agent_obs['active']][0] > 0.7:
                components['checkpoint_reward'][i] += 0.1
                reward[i] += 0.1

            # Rewards for accurate, goal-facing shots
            if agent_obs['game_mode'] == 14 and agent_obs['ball_direction'][0] > 0:
                components['checkpoint_reward'][i] += 0.3
                reward[i] += 0.3

            # Rewards for successful passes:
            if agent_obs['game_mode'] == 13 and agent_obs['ball_direction'][0] > 0 and agent_obs['right_team_active'][agent_obs['ball_owned_player']]:
                components['checkpoint_reward'][i] += 0.2
                reward[i] += 0.2
                
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
                self.sticky_actions_counter[i] = (self.sticky_actions_counter[i] + action) % 2
        return observation, reward, done, info
