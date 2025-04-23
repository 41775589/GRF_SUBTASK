import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering long passes."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_completion_bonus = 0.2  # Reward bonus for completing a pass
        self.distance_threshold_bonus = 0.15  # Additional bonus for long passes
        self.long_pass_minimum_distance = 0.5  # Minimum distance for a pass to be considered "long"
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['StickyActionsCounter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['StickyActionsCounter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_completion_bonus": [0.0] * len(reward),
                      "long_pass_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        # Loop over each agent
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            ball_ownership = o.get('ball_owned_team')
            ball_end_pos = o['ball']

            if ball_ownership == -1:  # Ball is not owned
                continue
            
            pass_distance = np.linalg.norm(ball_end_pos[:2] - active_player_pos)
            
            # Check if there was a successful pass
            if pass_distance > 0 and ball_ownership == o['ball_owned_team']:
                components['pass_completion_bonus'][rew_index] = self.pass_completion_bonus
                reward[rew_index] += self.pass_completion_bonus
                
                # Check if the pass is a long pass
                if pass_distance > self.long_pass_minimum_distance:
                    components['long_pass_bonus'][rew_index] = self.distance_threshold_bonus
                    reward[rew_index] += self.distance_threshold_bonus

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
