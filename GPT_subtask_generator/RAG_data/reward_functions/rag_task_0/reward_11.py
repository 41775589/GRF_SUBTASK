import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward for developing offensive strategies including shooting, dribbling, and passing."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_reward = 0.8
        self.dribbling_reward = 0.5
        self.passing_reward = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned_team = o['ball_owned_team']
            active_player = o['active']

            # Check if the ball is owned by the controlling team and active player has the ball
            if ball_owned_team == 0 and o.get('ball_owned_player') == active_player:
                # Reward for dribbling
                if o['sticky_actions'][9]:  # Dribble action
                    reward[rew_index] += self.dribbling_reward
                    components["dribbling_reward"][rew_index] = self.dribbling_reward

                # Reward for passes
                if o['game_mode'] in [2, 5]:  # Modes related to kicking the ball (GoalKick, ThrowIn)
                    reward[rew_index] += self.passing_reward
                    components["passing_reward"][rew_index] = self.passing_reward

                # Reward for shooting towards goal
                ball_direction = o['ball_direction']
                goal_direction = [1, 0] if ball_owned_team == 0 else [-1, 0]  # Right or left goal based on team
                shooting_alignment = np.dot(ball_direction[:2], goal_direction)

                if shooting_alignment > 0.5:  # implies the ball is moving towards the opponent's goal
                    reward[rew_index] += self.shooting_reward
                    components["shooting_reward"][rew_index] = self.shooting_reward

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
