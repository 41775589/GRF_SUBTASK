import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the defensive and tackling capabilities 
    of agents by rewarding aggressive defensive moves and timely reactions to opposing attacks.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parametrize rewards for defensive actions
        self.tackle_reward = 0.2
        self.slide_reward = 0.3
        self.positioning_reward_coef = 0.1

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
            "defensive_action_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Increase reward for tackle (action 9 - slide) and dribble interception
            if o['sticky_actions'][9]:  # slide action
                components["defensive_action_reward"][rew_index] = self.slide_reward
            elif o['sticky_actions'][0] or o['sticky_actions'][4]:  # left or right tackle
                components["defensive_action_reward"][rew_index] = self.tackle_reward

            # Position defensive players closer to opposing attackers with the ball
            if o['left_team_roles'][o['active']] in [1, 4, 5] and o['ball_owned_team'] == 1:  # Defensive indices, 1=CENTER_BACK etc.
                # Calculate distance to the ball carrier
                ball_pos = o['ball'][:2]
                def_pos = o['left_team'][o['active']]
                distance = np.linalg.norm(ball_pos - def_pos)
                # Reward for being close to ball carrier
                components["positioning_reward"][rew_index] = max(0, self.positioning_reward_coef * (1 - distance))

            reward[rew_index] += components["defensive_action_reward"][rew_index] + components["positioning_reward"][rew_index]

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
