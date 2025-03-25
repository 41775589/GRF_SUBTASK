import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive actions, focused on the role of a 'stopper'."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.opponent_approaches_counter = [0]*2
        self.block_counter = [0]*2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.opponent_approaches_counter = [0]*2
        self.block_counter = [0]*2
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'opponent_approaches_counter': self.opponent_approaches_counter,
            'block_counter': self.block_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = state_data['sticky_actions_counter']
        self.opponent_approaches_counter = state_data['opponent_approaches_counter']
        self.block_counter = state_data['block_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, {'base_score_reward': reward}

        assert len(reward) == len(observation)

        # Initialize reward components dictionary
        components = {'base_score_reward': reward.copy(),
                      'defense_effort': [0.0] * len(reward),
                      'blocks': [0.0] * len(reward)}

        for idx, _ in enumerate(reward):
            obs = observation[idx]
            right_players_pos = obs.get('right_team', [])
            ball_pos = obs.get('ball', [])

            # Determine if opponent is approaching with the ball
            opponent_approaching = any(
                np.linalg.norm(ball_pos[:2] - player_pos) < 0.03 for player_pos in right_players_pos)

            if opponent_approaching:
                self.opponent_approaches_counter[idx] += 1
                if obs.get('ball_owned_team') == 0 and obs.get('active') == obs.get('ball_owned_player'):
                    # Agent blocked the opponent's approach
                    self.block_counter[idx] += 1
                    components['blocks'][idx] += 0.5  # Reward for successful block

            # Calculate cumulative defense effort score
            components['defense_effort'][idx] = self.block_counter[idx] * 0.1

            # Update cumulative reward with the additional components
            reward[idx] = components['base_score_reward'][idx] + \
                          components['defense_effort'][idx] + \
                          components['blocks'][idx]

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
