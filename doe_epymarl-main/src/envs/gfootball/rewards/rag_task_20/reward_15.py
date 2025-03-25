import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on offensive strategies, optimizing team coordination and reaction 
    by incentivizing passing, positioning, and shooting at the opponent's goal."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters to control various incentive strengths
        self.pass_reward = 0.02
        self.positioning_threshold = 0.1
        self.shot_on_target_reward = 0.4

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle = self.env.get_state(to_pickle)
        return to_pickle

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Store the baseline reward values
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": 0.0,
                      "positioning_reward": 0.0,
                      "shot_on_goal_reward": 0.0}
        
        observation = self.env.unwrapped.observation()
        # Handle the case where the observation isn't available
        if observation is None:
            return reward, components
          
        assert len(reward) == len(observation)

        for i in range(len(reward)):
            obs = observation[i]
            components["pass_reward"] += self.evaluate_pass_effectiveness(obs)
            components["positioning_reward"] += self.evaluate_positioning(obs)
            components["shot_on_goal_reward"] += self.evaluate_shots_on_target(obs)
            
            # Update the reward for this agent
            reward[i] += (components["pass_reward"] + 
                          components["positioning_reward"] + 
                          components["shot_on_goal_reward"])

        return reward, components

    def evaluate_pass_effectiveness(self, obs):
        """Calculates reward for effective passing, particularly forward passes towards the opponent's goal."""
        if obs['ball_owned_team'] == 0:
            if self.sticky_actions_counter[9] > 0: # action_dribble is on
                # Assume forward passes towards the goal are valuable, check if passing action is performed
                if self.sticky_actions_counter[3] > 0 or self.sticky_actions_counter[4] > 0:  # Right or Right-Forward actions
                    return self.pass_reward
        return 0.0

    def evaluate_positioning(self, obs):
        """Encourages positioning close to the opponent's goal for potential shooting opportunities."""
        ball_pos = obs['ball'][0] # x-coordinate of the ball
        if ball_pos > 0:  # Ball is on the opponent's half
            return max(0, ball_pos - self.positioning_threshold) * self.positioning_threshold
        return 0.0

    def evaluate_shots_on_target(self, obs):
        """Rewards shots directed towards the opponent's goal."""
        if obs['game_mode'] == 0 and obs['ball_owned_team'] == 0: # Normal play and ball owned by the agent's team
            if self.sticky_actions_counter[2] > 0:  # action_shoot
                return self.shot_on_target_reward
        return 0.0

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
