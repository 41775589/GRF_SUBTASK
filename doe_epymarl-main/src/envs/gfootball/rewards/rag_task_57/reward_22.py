import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to promote offensive strategies involving midfielders and strikers."""
    
    def __init__(self, env):
        super().__init__(env)
        # Initialization of game element tracking
        self.midfielder_positions = []
        self.striker_positions = []
        self.ball_position = None
        self.last_ball_owner = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the game state trackers."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfielder_positions = []
        self.striker_positions = []
        self.ball_position = None
        self.last_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of the reward wrapper."""
        to_pickle['midfielder_positions'] = self.midfielder_positions
        to_pickle['striker_positions'] = self.striker_positions
        to_pickle['ball_position'] = self.ball_position
        to_pickle['last_ball_owner'] = self.last_ball_owner
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the reward wrapper."""
        from_pickle = self.env.set_state(state)
        self.midfielder_positions = from_pickle['midfielder_positions']
        self.striker_positions = from_pickle['striker_positions']
        self.ball_position = from_pickle['ball_position']
        self.last_ball_owner = from_pickle['last_ball_owner']
        return from_pickle

    def reward(self, reward):
        """Modify reward based on offensive plays by midfielders and strikers."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "offensive_play_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check roles; assuming midfielder roles are [4,5,6], striker roles are [8,9]
            is_midfielder = o['left_team_roles'][o['active']] in [4, 5, 6]
            is_striker = o['left_team_roles'][o['active']] in [8, 9]

            # Track ball and player positions
            if is_midfielder:
                self.midfielder_positions.append(o['left_team'][o['active']])
            elif is_striker:
                self.striker_positions.append(o['left_team'][o['active']])
            
            self.ball_position = o['ball'][:2]  # Assuming 3D ball position, ignore z

            # Reward connection between midfielders and strikers
            if self.last_ball_owner in self.midfielder_positions and self.ball_position in self.striker_positions:
                components["offensive_play_reward"][rew_index] = 0.1
                reward[rew_index] += components["offensive_play_reward"][rew_index]

            # Update ball owner
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                self.last_ball_owner = o['left_team'][o['active']]

        return reward, components

    def step(self, action):
        """Process environment step and modify rewards."""
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
