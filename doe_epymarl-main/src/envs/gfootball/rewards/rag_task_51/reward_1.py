import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for goalkeeping training."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for shot-stopping and quick reflexes training
        self.save_threshold = 0.1  # Reward for stopping close shots
        self.reflex_reward = 0.05  # Reward for quick reaction to fast-moving balls
        self.pass_accuracy_reward = 0.2  # Reward for accurate passes starting counter-attacks

    def reset(self):
        # Reset sticky actions tracker on new episode starts
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Saving states if required for resuming training sessions
        to_pickle['CheckpointRewardWrapper_StickyActions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Setting state from saved state if available
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_StickyActions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        # Get the observation from the environment to calculate additional rewards
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "save_reward": [0.0, 0.0],
                      "reflex_reward": [0.0, 0.0],
                      "passing_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components

        # Apply the specialized goalkeeper training rewards
        for index, o in enumerate(observation):
            if o['game_mode'] in [4, 5]:  # Consider only in-game active play
                # Determine shot-stopping performance
                if o['ball_owned_team'] == 0 and o['left_team_roles'][o['active']] == 0:  # Is the goalkeeper
                    ball_speed = np.linalg.norm(o['ball_direction'])
                    if ball_speed > 0.1 and np.abs(o['ball'][0] + 1) < self.save_threshold:  # Close to goal line
                        components["save_reward"][index] = self.save_threshold
                        
                    # Checking reflexes on quick ball direction changes
                    if ball_speed > 0.2:
                        components["reflex_reward"][index] = self.reflex_reward * ball_speed
            
            # Reward for initiating accurate counter-attacks
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # Check if the pass leads to a counter-attack setup
                if 'action_bottom_right' in o['sticky_actions'] or 'action_bottom_left' in o['sticky_actions']:
                    components['passing_reward'][index] = self.pass_accuracy_reward

        # Apply the specialized rewards to the original rewards
        for idx in range(len(reward)):
            reward[idx] += components["save_reward"][idx] + components["reflex_reward"][idx] + components["passing_reward"][idx]

        return reward, components

    def step(self, action):
        # Execute environment step
        observation, reward, done, info = self.env.step(action)
        # Adjust reward by wrapping it with the specified goalkeeper rewards
        reward, components = self.reward(reward)
        # Add final and component reward values into info dictionary for tracking
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Monitor sticky actions for informational purposes
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
