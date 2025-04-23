import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for executing accurate long passes between predefined zones on the playfield.
    The reward function encourages agents to improve their vision, timing, and precision in ball distribution.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define 4 main zones for the playfield for simplicity: [defensive, midfield, attacking, goal area]
        self.pass_zones = {
            0: ([-1.0, -0.5]),  # Defensive
            1: ([-0.5, 0.0]),   # Midfield
            2: ([0.0, 0.5]),    # Attacking
            3: ([0.5, 1.0])     # Goal Area
        }
        self.last_ball_zone = -1
        self.pass_bonus = 0.2  # Reward increment for successful pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_zone = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'last_ball_zone': self.last_ball_zone
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_zone = from_pickle['CheckpointRewardWrapper']['last_ball_zone']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, {}

        assert len(reward) == len(observation)
        
        components = {
            "base_score_reward": [r.copy() for r in reward],
            "passing_reward": [0.0] * len(reward)
        }

        for idx, r in enumerate(reward):
            player_obs = observation[idx]

            if 'ball_owned_team' in player_obs:
                ball_zone = self._determine_zone(player_obs['ball'][0])

                if self.last_ball_zone != -1 and ball_zone != self.last_ball_zone:
                    if 'ball_owned_player' in player_obs and player_obs['ball_owned_player'] == player_obs['active']:
                        distance = abs(ball_zone - self.last_ball_zone)
                        # Reward only for longer passes (i.e., passing across at least one zone)
                        if distance > 1:
                            components["passing_reward"][idx] = self.pass_bonus * distance
                            reward[idx] += components["passing_reward"][idx]

                self.last_ball_zone = ball_zone

        return reward, components

    def _determine_zone(self, x_pos):
        """Determine football field zone based on ball's x position."""
        for idx, zone in self.pass_zones.items():
            if zone[0] <= x_pos <= zone[1]:
                return idx
        return -1

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        
        # Track sticky actions activations
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_state
        
        return observation, reward, done, info
