import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward focused on defending strategies including tackling,
    efficient movement control, and pressured passing tactics."""

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
        components = {"base_score_reward": reward.copy(),
                      "defending_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            components["defending_bonus"][rew_index] = 0
            if o is not None:
                player_role = o.get('left_team_roles', [])[o['active']]
                possession_team = o['ball_owned_team']
                player_team = 0  # assuming the left team is our focus for training

                # Tackling reward: positive feedback when the opponent loses the ball
                if possession_team == 1:  # ball is with the right team (opponent)
                    # Simulated tackling effectiveness
                    components["defending_bonus"][rew_index] += 0.05

                # Efficiency in movement: reduced movement when not necessary helps preserve stamina
                if np.all(np.abs(o['left_team_direction'][o['active']]) < 0.01):  # minimal player movement
                    components["defending_bonus"][rew_index] += 0.03

                # Pressured passing: bonus for having the ball and passing under pressure
                if possession_team == player_team and 'ball_owned_player' in o \
                        and o['ball_owned_player'] == o['active']:
                    # Simulating pressure by proximity of opponent players
                    opponent_distances = np.linalg.norm(
                        o['right_team'] - o['left_team'][o['active']],
                        axis=1
                    )
                    # If opponents are close, assume pressure is higher
                    if np.any(opponent_distances < 0.2):  # Threshold distance to consider "pressure"
                        components["defending_bonus"][rew_index] += 0.07

            # Update rewards
            reward[rew_index] += components["defending_bonus"][rew_index]

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
