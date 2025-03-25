import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies rewards based on defensive maneuvers and midfield control.
    The function focuses on rewarding strategically preventing scoring opportunities
    and controlling the ball in the midfield.
    """
    def __init__(self, env):
        super().__init__(env)
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
        components = {
            "base_score_reward": reward.copy(),
            "defensive_move_bonus": [0.0] * len(reward),
            "midfield_control_bonus": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for proper positioning and interception in the defensive half
            if o['active'] in o['left_team'] or o['active'] in o['right_team']:
                player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            
                # Considered good defensive position if closer to their goal
                if player_pos[0] < 0 and o['ball_owned_team'] == 1:  # left team defending against right team
                    components["defensive_move_bonus"][rew_index] = 0.05 * np.clip(1 - abs(player_pos[0]), 0, 1)
                elif player_pos[0] > 0 and o['ball_owned_team'] == 0:  # right team defending against left team
                    components["defensive_move_bonus"][rew_index] = 0.05 * np.clip(1 - abs(player_pos[0]), 0, 1)

                reward[rew_index] += components["defensive_move_bonus"][rew_index]

            # Reward for controlling the ball in the midfield
            if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                ball_pos = o['ball']
                if -0.4 < ball_pos[0] < 0.4:  # The ball is considered to be in midfield
                    components["midfield_control_bonus"][rew_index] = 0.1
                    reward[rew_index] += components["midfield_control_bonus"][rew_index]

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
