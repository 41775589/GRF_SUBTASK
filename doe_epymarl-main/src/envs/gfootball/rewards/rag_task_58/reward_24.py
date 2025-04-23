import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive performance and effective transitions from defense to attack."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Distance thresholds mapping improvement in defense-to-attack transitions to extra rewards.
        self.defensive_efficiency_reward = 0.05

    def reset(self):
        """Reset the environment."""
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Retrieve the state to enable pickling."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state based on unpickled data."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Compute additional reward for defensive actions and smooth transitioning."""
        components = {"base_score_reward": reward.copy(),
                      "defense_to_attack_transition_reward": [0.0] * len(reward)}
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Enhance rewards linked to defense action patterns
            if o['ball_owned_team'] == 1:  # Considering our agent's team as default team 1 (defensive team)
                opponent_goals = o['score'][1]  # Assuming team 0 is the opponent
                my_team_goals = o['score'][0]
                
                # Reward for ball clearance (defensive action) by checking the ball's proximity to our goal-zone
                ball_x, ball_y, _ = o['ball']
                if ball_x < -0.8 and np.abs(ball_y) < 0.2:  # Close to our goal line
                    components['defense_to_attack_transition_reward'][rew_index] += self.defensive_efficiency_reward

                # Encourage transitioning to attack without losing possession
                if o['ball_owned_team'] == 1 and o['ball'][0] > 0:  # Ball is in the opponent's half and we own the ball
                    components['defense_to_attack_transition_reward'][rew_index] += 2 * self.defensive_efficiency_reward
            
            # Combine new reward components with the base reward
            reward[rew_index] += components['defense_to_attack_transition_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Execute a step in the environment."""
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
