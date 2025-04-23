import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized defensive and counterattack rewards."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_position_history = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_position_history = []
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['ball_position_history'] = self.ball_position_history
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.ball_position_history = from_pickle['ball_position_history']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, {}

        assert len(reward) == len(observation)
        for i in range(len(reward)):
            o = observation[i]
            components.setdefault("defensive_reward", [0.0] * len(reward))
            components.setdefault("counterattack_reward", [0.0] * len(reward))

            ball_x, ball_y = o['ball'][0:2]
            ball_position = (ball_x, ball_y)

            if o['ball_owned_team'] == 0 and len(self.ball_position_history) > 1:
                # Euro example: 1.5 * reward if ball moves towards opponent's side when owned by team
                prev_ball_x = self.ball_position_history[-2][0]
                if ball_x > prev_ball_x:  # Progressing rightward towards opponent
                    components["defensive_reward"][i] += 0.1 * reward[i]
                    reward[i] += components["defensive_reward"][i]
        
            # Reward for successful counterattacks, i.e., transitioning from defense to attack
            # Assume active player is in defensive half and we check move to offensive half
            if (o['ball_owned_team'] == 0 and ball_x > 0 
                and o['left_team'][o['active']][0] < 0):
                reward[i] += 0.3
                components["counterattack_reward"][i] += 0.3

            self.sticky_actions_counter += o['sticky_actions']
        
        # Update ball position history
        self.ball_position_history.append(ball_position)

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
