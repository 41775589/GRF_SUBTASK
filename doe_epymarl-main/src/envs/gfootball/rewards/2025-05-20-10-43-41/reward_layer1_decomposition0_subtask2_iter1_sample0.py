import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defensive maneuvers and control in wide areas."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Additional tracking for long ball interceptions
        self.interceptions = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'interceptions': self.interceptions}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.interceptions = from_pickle['CheckpointRewardWrapper']['interceptions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {"base_score_reward": reward.copy(), "defensive_skill_reward": [0.0]}
        
        assert len(reward) == 1  # Since the training involves only one agent.
        o = observation[0]

        components = {"base_score_reward": reward.copy(), "defensive_skill_reward": [0.0]}
        defensive_skill_bonus = 0

        # Encourage intercepting long balls in wide areas with appropriate defensive actions
        if o['ball_owned_team'] == -1 and np.linalg.norm(o['ball_direction'][:2]) > 0.1:
            interceptor_pos = np.array(o['left_team'][o['active']] if o['designated'] == 0 else o['right_team'][o['active']])
            ball_pos = np.array(o['ball'][:2])
            if np.linalg.norm(interceptor_pos - ball_pos) < 0.2:
                self.interceptions += 1
                defensive_skill_bonus += 0.3

        # Reward sliding tackles in wide areas
        if 'sliding' in o['sticky_actions']:
            defensive_skill_bonus += 0.5
        
        # Adding rewards for maintaining position using Stop-Dribble in control zones
        if 'stop_dribble' in o['sticky_actions']:
            player_pos = np.array(o['left_team'][o['active']])
            # More reward if this action happens on sidelines (wide areas)
            if np.abs(player_pos[1]) > 0.7:
                defensive_skill_bonus += 0.5

        components["defensive_skill_reward"][0] = defensive_skill_bonus
        reward[0] += defensive_skill_bonus

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
