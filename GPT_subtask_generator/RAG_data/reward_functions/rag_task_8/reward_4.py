import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for ball recovery and quick counter-attacks."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_ball_owned_team = -1  # Initializing as no team has possession
        self.recovery_positions = {}
        self.counter_attack_reward = 1.0

    def reset(self):
        """Reset the environment and the tracking variables for the new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_ball_owned_team = -1
        self.recovery_positions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Add this wrapper's state to the pickle object."""
        to_pickle['CheckpointRewardWrapper'] = {
            'prev_ball_owned_team': self.prev_ball_owned_team,
            'recovery_positions': self.recovery_positions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Retrieve the state of this wrapper from the pickle object."""
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self.prev_ball_owned_team = wrapper_state['prev_ball_owned_team']
        self.recovery_positions = wrapper_state['recovery_positions']
        return from_pickle

    def reward(self, reward):
        """Reward agent for recovering the ball and initiating a quick counter-attack."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "recovery_reward": [0.0] * len(reward)}

        # Loop through agent observations
        for agent_idx in range(len(observation)):
            o = observation[agent_idx]
            current_ball_team = o['ball_owned_team']
            
            # Check if possession has changed to this agent's team
            if self.prev_ball_owned_team != -1 and current_ball_team != self.prev_ball_owned_team:
                if current_ball_team == 0:  # Assuming agent's team is 0
                    components['recovery_reward'][agent_idx] = self.counter_attack_reward
                    reward[agent_idx] += components['recovery_reward'][agent_idx]
                    self.recovery_positions[agent_idx] = o['ball']
            
            # Check for quick ball movement towards opposition goal post-recovery
            if agent_idx in self.recovery_positions:
                ball_pos = o['ball'][0]  # x-coordinate of the ball
                if ball_pos > self.recovery_positions[agent_idx][0] + 0.3:
                    components['recovery_reward'][agent_idx] += self.counter_attack_reward
                    reward[agent_idx] += components['recovery_reward'][agent_idx]
                    del self.recovery_positions[agent_idx]  # Reward once per recovery

        # Update the previous ball ownership status
        self.prev_ball_owned_team = current_ball_team
        return reward, components

    def step(self, action):
        """Perform a step using the provided action, augmenting info dictionary with reward components."""
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
