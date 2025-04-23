import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for mastering close-range attacks, 
    focusing on shot precision, dribble effectiveness, and quick decision-making against goalkeepers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track sticky actions usage
        self.dribble_effectiveness_counter = 0
        self.precision_shot_counter = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_effectiveness_counter = 0
        self.precision_shot_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state.update({
            'dribble_effectiveness': self.dribble_effectiveness_counter,
            'precision_shot': self.precision_shot_counter
        })
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.dribble_effectiveness_counter = from_pickle['dribble_effectiveness']
        self.precision_shot_counter = from_pickle['precision_shot']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        # Handle the reward and components based on the observation
        components = {
            "base_score_reward": reward.copy(),
            "dribble_reward": [0.0] * len(reward),
            "precision_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        # Process each agent's reward based on the presence of the ball and its proximity to the goal
        for rew_index in range(len(reward)):
            o = observation[rew_index]
  
            # Check if the active player has the ball
            if o['ball_owned_team'] == 1 and o['active'] == o['ball_owned_player']:
                dribbling = any(o['sticky_actions'][8:10])  # considering action_dribble and action_sprint
                if dribbling:
                    # Dribble effectively towards opponent's goal
                    self.dribble_effectiveness_counter += 1
                    components["dribble_reward"][rew_index] = 0.1 * self.dribble_effectiveness_counter

                # Check proximity to goal for precise shooting reward
                ball_pos_x = o['ball'][0]
                if ball_pos_x > 0.8:  # Close to opponent's goal on x-axis
                    self.precision_shot_counter += 1
                    components["precision_reward"][rew_index] = 0.2 * self.precision_shot_counter

            # Combine the rewards with their components
            reward[rew_index] += components["dribble_reward"][rew_index] + components["precision_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions usage from observations for reporting
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
