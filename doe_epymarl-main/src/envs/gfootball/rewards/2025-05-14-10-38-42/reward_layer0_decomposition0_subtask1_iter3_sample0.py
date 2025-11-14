import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive play by emphasizing teamwork and effective area coverage."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Reward parameters
        self.team_defensive_positioning_reward = 0.2
        self.ball_interception_reward = 0.35
        self.successful_tackle_reward = 0.45
        
        # Configuration for defensive zones
        self.defensive_zones = {
            'high_defensive_zone': 0.75,
            'mid_defensive_zone': 0.5,
            'low_defensive_zone': 0.25
        }

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
            "team_defensive_positioning_reward": [0.0] * len(reward),
            "ball_interception_reward": [0.0] * len(reward),
            "successful_tackle_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for idx, obs in enumerate(observation):
            player_pos = obs['left_team'][obs['active']]
            ball_pos = obs['ball']
             
            # Reward based on defensive position relative to the ball
            dist_from_ball = np.linalg.norm(np.array(player_pos[:2]) - np.array(ball_pos[:2]))
            
            # Encourage players to cover crucial defensive zones effectively
            for zone, distance_threshold in self.defensive_zones.items():
                if dist_from_ball < distance_threshold and obs['ball_owned_team'] == 1:
                    components["team_defensive_positioning_reward"][idx] = self.team_defensive_positioning_reward
                    reward[idx] += components["team_defensive_positioning_reward"][idx]

            # Reward for intercepting the ball effectively in defensive posture
            if obs['ball_owned_team'] == 1 and dist_from_ball < 0.1:  # ball is close and owned by opponent
                components["ball_interception_reward"][idx] = self.ball_interception_reward
                reward[idx] += components["ball_interception_reward"][idx]

            # Additional reward for successful tackles
            if 'tackle' in obs['sticky_actions'] and obs['sticky_actions']['tackle']:
                components["successful_tackle_reward"][idx] = self.successful_tackle_reward
                reward[idx] += components["successful_tackle_reward"][idx]

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
