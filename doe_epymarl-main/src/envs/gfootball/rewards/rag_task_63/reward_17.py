import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to add reward contributions for goalkeeper training, focusing on shot stopping, positioning under pressure, and communication."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_goalkeeping_quality = {}

    def reset(self):
        """Resets the environment and clear goalkeeper training metrics."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.player_goalkeeping_quality = {}
        return self.env.reset()
        
    def get_state(self, to_pickle):
        """State serialization to include goalkeeper quality metrics."""
        to_pickle['goalkeeper_metrics'] = self.player_goalkeeping_quality
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state back with goalkeeper metrics."""
        from_pickle = self.env.set_state(state)
        self.player_goalkeeping_quality = from_pickle['goalkeeper_metrics']
        return from_pickle
        
    def reward(self, reward):
        """Compute reward given goalkeeper actions, positioning, and decision-making under pressure."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_efficiency": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            # Identify goalkeeper actions and quality.
            if o['active'] == o['left_team_roles'].index(0):  # Assuming 0 is the role index for goalkeeper
                keeper_quality = self.evaluate_goalkeeper(o)
                reward[i] += keeper_quality
                components["goalkeeper_efficiency"][i] = keeper_quality

        return reward, components

    def evaluate_goalkeeper(self, obs):
        """Evaluate the goalkeeper based on proximity to goal, ball ownership, and clearances under pressure."""
        quality = 0

        # Close to the goal and facing an attack
        if obs['left_team'][obs['active']][0] < -0.5: # if the goalkeeper is towards own goal
            quality += 0.1  # positional awareness

        if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']: 
            quality += 0.2  # ball handling under pressure

        # Communication or clearances can be inferred from game mode changes
        if obs['game_mode'] in {4, 5, 6} and obs['active'] in obs['left_team_roles'][0]:  # modes indicating set pieces or pressure
            quality += 0.3  # effectiveness in pressure situations

        return quality

    def step(self, action):
        """Applies an action, steps the environment, and augments reward and info with reward decomposition."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
