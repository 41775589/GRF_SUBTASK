import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for defensive and midfield play by examining possession changes,
    position of ball interception, and teamwork.

    It focuses on reinforcing behaviors related to strategic defense positioning and tactical midfield control,
    aiming to reward successful interceptions and transitions that involve multiple players.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To keep track of sticky actions.

    def reset(self):
        """Reset the environment and counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Reset sticky actions counter.
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save state for resuming the game."""
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions': self.sticky_actions_counter.copy()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the environment state from a saved state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions']
        return from_pickle

    def reward(self, reward):
        """Customize reward based on defense and midfield interplay."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'interception_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            previous_possession = o['prev_ball_owned_team']
            current_possession = o['ball_owned_team']
            
            # Detect change of possession potentially due to interception on defense or strategic midfield play.
            if previous_possession != current_possession:
                if current_possession == 0:  # The agent's team has taken possession.
                    components['interception_reward'][rew_index] = 0.3  # Reward for gaining possession.
                    reward[rew_index] += components['interception_reward'][rew_index]

                # Additional rewards for maintaining possession with effective passes within midfield,
                # stimulating cooperative play.
                if 'ball_owned_player' in o and o['ball_owned_team'] == 0:
                    active_player = o['ball_owned_player']
                    # Check if the ball is in midfield areas.
                    if -0.1 <= o['ball'][0] <= 0.1:
                        components['interception_reward'][rew_index] += 0.2
                        reward[rew_index] += components['interception_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Execute a step in the environment, add reward parts, and stick actions to info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)  # Sum of rewards for debugging or logging.
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
                info[f'sticky_actions_{i}'] = action
                
        return observation, reward, done, info
