import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    This class is a wrapper for adding specialized rewards focusing on enhancing defending strategies.
    The rewards are based on tackling proficiency, efficient movement control for interception,
    and pressured passing tactics.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Modify coefficients as needed to emphasize on different aspects of the defending.
        self.tackle_coeff = 2.0
        self.interception_coeff = 1.5
        self.pressure_pass_coeff = 1.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # save any additional state information your reward function needs
        to_pickle['Wrapper_sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['Wrapper_sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        # This assumes observations are accessible via 'self.env.unwrapped.observation()' and are dictionary form for each agent.
        observations = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "interception_reward": [0.0] * len(reward),
                      "pressure_pass_reward": [0.0] * len(reward)}

        for i, obs in enumerate(observations):
            ball_owned_team = obs['ball_owned_team']
            active_player = obs['active']
            ball_owned_player = obs['ball_owned_player']

            # Reward for successful tackles (assume a successful tackle when ball ownership changes near the player).
            # Note: Implementation of logic to check ownership change is assumed.
            if 'game_mode' in obs and obs['game_mode'] == 2 and ball_owned_team == 0 and active_player != ball_owned_player and np.any(obs['sticky_actions'][0:2]):
                components['tackle_reward'][i] = self.tackle_coeff
            
            # Reward for successful interceptions (here, consider a good positional play as a proxy).
            distance_to_ball = np.linalg.norm(obs['ball'][:2] - obs['left_team'][active_player])
            if distance_to_ball < 0.1 and obs['ball_owned_team'] == 1:  # Assuming ball possession team ID is +1 for opponents.
                components['interception_reward'][i] = self.interception_coeff

            # Reward for making accurate passes under pressure.
            # Assuming 'pressure' is conceptualized by the number of opposing players nearby.
            if np.sum(obs['right_team_active']) > 3 and ball_owned_player == active_player:
                # This could further be refined based on the result of the pass (successful reception by teammate).
                components['pressure_pass_reward'][i] = self.pressure_pass_coeff

            # Combine rewards
            reward[i] += components['tackle_reward'][i] + components['interception_reward'][i] + components['pressure_pass_reward'][i]
        
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
                self.sticky_actions_counter[action] += 1
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
