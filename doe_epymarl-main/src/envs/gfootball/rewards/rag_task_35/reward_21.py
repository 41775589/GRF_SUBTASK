import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward mechanism to emphasize strategic positioning between defense and attack."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_penalty = {}
        self.max_penalty = -0.1
        self.position_reward = {}
        self.max_reward = 0.1
        self.set_reward_penalty_checkpoints()

    def reset(self):
        self.position_penalty = {}
        self.position_reward = {}
        self.set_reward_penalty_checkpoints()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def set_reward_penalty_checkpoints(self):
        # Define discrete regions in the field where strategic positioning is crucial
        self.position_penalty[1] = {'zone': (-0.75, 0.0, -0.42, 0.42), 'applied': False}  # Defensive left half
        self.position_penalty[2] = {'zone': (0.75, 1.0, -0.42, 0.42), 'applied': False}   # Attacking right close to the goal
        self.position_reward[1] = {'zone': (-0.5, 0.5, -0.42, 0.42), 'applied': False}   # Midfield control

    def get_state(self, to_pickle):
        pickleable_state = {
            'position_penalty': self.position_penalty,
            'position_reward': self.position_reward,
            'sticky_actions_counter': self.sticky_actions_counter.tolist()  # Convert numpy array to list for pickling
        }
        to_pickle.update(pickleable_state)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.position_penalty = from_pickle['position_penalty']
        self.position_reward = from_pickle['position_reward']
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, player_obs in enumerate(observation):
            player_pos = player_obs['right_team'][player_obs['active']] if player_obs['ball_owned_team'] == 1 else player_obs['left_team'][player_obs['active']]
            for key, val in self.position_penalty.items():
                if not val['applied'] and self.is_within_zone(player_pos, val['zone']):
                    reward[rew_index] += self.max_penalty
                    components[f"penalty_zone_{key}"] = self.max_penalty
                    val['applied'] = True
            
            for key, val in self.position_reward.items():
                if not val['applied'] and self.is_within_zone(player_pos, val['zone']):
                    reward[rew_index] += self.max_reward
                    components[f"reward_zone_{key}"] = self.max_reward
                    val['applied'] = True

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            if isinstance(value, list):
                info[f"component_{key}"] = sum(value)
            else:
                info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_status in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_status
        return observation, reward, done, info

    def is_within_zone(self, position, zone):
        """Check if position (x, y) falls within the defined zone (xmin, xmax, ymin, ymax)."""
        x, y = position
        xmin, xmax, ymin, ymax = zone
        return xmin <= x <= xmax and ymin <= y <= ymax
