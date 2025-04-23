import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward for specific defensive actions and strategies in football."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_efficiency_reward = 0.05
        self.tackle_reward = 0.2
        self.passing_under_pressure_reward = 0.15

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_efficiency_reward": [0.0] * len(reward),
                      "tackle_reward": [0.0] * len(reward),
                      "passing_under_pressure_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward defensive position effectiveness
            if 'right_team' in o and 'active' in o:
                # Calculate distance from goal to encourage defensive positioning
                own_goal_x = -1  # Assumption for left team defense
                player_x = o['right_team'][o['active']][0]
                dist_to_goal = abs(own_goal_x - player_x)
                if dist_to_goal < 0.2:  # Close to goal
                    components['defensive_efficiency_reward'][rew_index] = self.defensive_efficiency_reward
                    reward[rew_index] += components['defensive_efficiency_reward'][rew_index]

            # Reward successful tackles
            if 'ball_owned_player' in o:
                # If this agent just tackled the ball
                if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0:
                    components['tackle_reward'][rew_index] = self.tackle_reward
                    reward[rew_index] += components['tackle_reward'][rew_index]

            # Reward for passing under pressure
            if 'game_mode' in o and o['game_mode'] == 2:  # FreeKick is a pressured situation
                if 'sticky_actions' in o and o['sticky_actions'][9]:  # Assuming index 9 is pass
                    components['passing_under_pressure_reward'][rew_index] = self.passing_under_pressure_reward
                    reward[rew_index] += components['passing_under_pressure_reward'][rew_index]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
