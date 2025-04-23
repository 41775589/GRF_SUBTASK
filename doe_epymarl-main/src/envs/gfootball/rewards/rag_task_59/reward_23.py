import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper specialized for goalkeeper coordination tasks, 
    focusing on high-pressure clears and accurate ball distribution."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_actions = {'good_catch': 0.5, 'good_clear': 1.0}
        self.previous_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward

        agent_idx = observation['active']  # Index of the agent in control
        own_team = observation['left_team'] if observation['ball_owned_team'] == 0 else observation['right_team']
        opponent_team = observation['right_team'] if observation['ball_owned_team'] == 0 else observation['left_team']
        goalkeeper_idx = np.argmin(own_team[:, 1])  # Assuming goalkeeper is at the minimum y-position

        ball = observation['ball'][:2]
        if self.previous_ball_position is None:
            ball_direction = np.zeros(2)
        else:
            ball_direction = ball - self.previous_ball_position

        components['goalkeeper_bonus'] = 0.0
        if agent_idx == goalkeeper_idx:
            # Assessing the performance based on goalie's position and ball ownership
            if observation['ball_owned_player'] == goalkeeper_idx:
                # Check if the goalkeeper is clearing the ball towards an outfield player
                clearing_direction = np.sign(ball_direction)
                targeted_teammate_positions = [teammate for teammate in own_team if np.sign(teammate - ball) == clearing_direction]
                distance_to_teammates = [np.linalg.norm(teammate - ball) for teammate in targeted_teammate_positions]

                if distance_to_teammates and min(distance_to_teammates) < 0.3:  # Reward for effective clear
                    components['goalkeeper_bonus'] += self.goalkeeper_actions['good_clear']
                elif np.linalg.norm(ball_direction) > 0.1:  # Bonus for catching and controlling the ball
                    components['goalkeeper_bonus'] += self.goalkeeper_actions['good_catch']

        reward[agent_idx] += components['goalkeeper_bonus']
        
        # Update previous ball position for the next frame
        self.previous_ball_position = observation['ball'][:2]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value if isinstance(value, list) else [value])

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action

        return observation, reward, done, info
