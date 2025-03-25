import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for controlling the sweeper role effectively"""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components['tackle_reward'] = 0.0
            components['clear_ball_reward'] = 0.0

            # Reward for effective tackling: Tackle opponents close to your own goal
            own_goal_x = -1 if o['right_team_active'] else 1
            if o['game_mode'] in {3, 4, 6} and abs(o['ball'][0] - own_goal_x) < 0.3:
                if o['ball_owned_team'] == (1 if o['right_team_active'] else 0) and \
                   'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                    components['tackle_reward'] = 0.5
                    reward[rew_index] += components['tackle_reward']

            # Reward for clearing the ball out of the danger zone
            if abs(o['ball'][0] - own_goal_x) < 0.3 and 'ball_direction' in o:
                ball_direction_towards_own_goal = np.sign(o['ball_direction'][0]) == np.sign(own_goal_x)
                if not ball_direction_towards_own_goal:
                    components['clear_ball_reward'] = 0.2
                    reward[rew_index] += components['clear_ball_reward']

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
