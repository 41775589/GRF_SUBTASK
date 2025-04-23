import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the effectiveness of collaborative plays between shooters and passers."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.passing_bonus = 0.3
        self.shooting_bonus = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.has_passed = False

    def reset(self):
        """Reset the environment and internal counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.has_passed = False
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of the reward wrapper along with environment state."""
        to_pickle['CheckpointRewardWrapper'] = {'has_passed': self.has_passed}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Load the state of the reward wrapper along with environment state."""
        from_pickle = self.env.set_state(state)
        self.has_passed = from_pickle.get('has_passed', False)
        return from_pickle

    def reward(self, reward):
        """Modify the reward to emphasize effective collaborative plays between shooters and passers."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 
                      'passing_bonus': [0.0] * len(reward),
                      'shooting_bonus': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # # Check if the action taken was a pass, and the current agent's team owns the ball
            if self.sticky_actions_counter[7] or self.sticky_actions_counter[1]:  # Assuming indices 7 or 1 are for passing
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                    self.has_passed = True
                    components['passing_bonus'][rew_index] = self.passing_bonus
                    reward[rew_index] += components['passing_bonus'][rew_index]
            
            # Check if there is a shooting opportunity, the ball is close to opponent's goal
            if np.abs(o['ball'][0]) > 0.8 and o['ball_owned_team'] == 0:
                if self.has_passed and (o['ball_owned_player'] == o['active']):  # Condition that active player has passed before shooting
                    components['shooting_bonus'][rew_index] = self.shooting_bonus
                    reward[rew_index] += components['shooting_bonus'][rew_index]

            # Reset the passing flag after reward calculation to avoid continuous reward without subsequent passes
            self.has_passed = False
                
        return reward, components

    def step(self, action):
        """Step through the environment, modify rewards, and provide additional info in diagnostics."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
