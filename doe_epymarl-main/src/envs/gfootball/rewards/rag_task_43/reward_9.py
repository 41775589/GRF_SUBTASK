import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that awards defensive strategy and counterattack abilities."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initializing counters for defensive positions and ball recoveries
        self.defensive_actions = 0
        self.ball_recoveries = 0
        self.positional_discount_factor = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_actions = 0
        self.ball_recoveries = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['Conditions'] = {
            'defensive_actions': self.defensive_actions,
            'ball_recoveries': self.ball_recoveries
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        conditions = from_pickle['Conditions']
        self.defensive_actions = conditions['defensive_actions']
        self.ball_recoveries = conditions['ball_recoveries']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        for agent_index, rew in enumerate(reward):
            o = observation[agent_index]
            defense_bonus = 0

            # Award based on defensive strategy: Position and responsiveness.
            if o['ball_owned_team'] == 1 and o['active'] in o['right_team']:
                # Being in a position to prevent opponent's from progressing is good
                for pos in o['right_team']:
                    if pos[0] < o['ball'][0]:  # Player behind the ball
                        defense_bonus += 0.01  # Increasing for potential covering

            # Counterattack potential: Ball recoveries.
            if o['ball_owned_team'] == -1 and self.env.prev_ball_owned_team == 1:
                self.ball_recoveries += 1
                reward[agent_index] += 0.1  # Reward for recovering the ball

            # Apply positional discount to diminish overflow of points and make it realistic
            defense_bonus -= self.positional_discount_factor * self.defensive_actions
            reward[agent_index] += defense_bonus
            self.defensive_actions += 1

        components["defense_bonus"] = [defense_bonus] * len(reward)
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += action
        return observation, reward, done, info
