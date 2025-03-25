import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward system focused on the skills of 'stopper'."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset wrapper state upon starting a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serializes the state of the wrapper."""
        to_pickle.update({'sticky_actions_counter': self.sticky_actions_counter})
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserializes the state into the wrapper."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Adjusts the rewards based on the players' stopper actions.
        
        This includes bonuses for man-marking, blocking shots, and intercepting forward moves.
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'man_marking_reward': [0.0] * len(reward),
                      'block_shot_reward': [0.0] * len(reward),
                      'stopping_forward_move_reward': [0.0] * len(reward)}

        # Iterate over each agent in the environment (len(reward) typically = 2)
        for agent_index, o in enumerate(observation):
            ball_owned_team = o.get('ball_owned_team', -1)
            agent_team = 0 if agent_index < len(reward) / 2 else 1
            opponent_has_ball = (ball_owned_team != agent_team and ball_owned_team != -1)

            # Reward for man-marking
            if opponent_has_ball and not o['ball_owned_player'] == o['designated']:
                # Add reward for effectively marking the man with the ball
                components['man_marking_reward'][agent_index] += 0.05

            # Reward for blocking shots
            if opponent_has_ball and 'game_mode' in o and o['game_mode'] == 3:  # Assuming game_mode 3 is a shooting mode
                components['block_shot_reward'][agent_index] += 0.1

            # Reward for stalling opponent forward moves
            if opponent_has_ball:
                for delta in o['right_team_direction']:
                    if np.hypot(delta[0], delta[1]) < 0.01:  # minimal movement detected
                        components['stopping_forward_move_reward'][agent_index] += 0.05

            # Final computation of reward for agent
            reward[agent_index] += sum(components[name][agent_index] for name in components)
        
        return reward, components

    def step(self, action):
        """Execute a step using the wrapped environment, adjust rewards, and return results."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
