import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that encourages improved defensive behavior including tackling, efficient movement, and passing under pressure."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward modifiers
        self.tackle_reward = 0.2
        self.pass_reward = 0.15
        self.stop_reward = 0.1

    def reset(self):
        """Reset the environment's state and any internal counters or states."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Save the current state of this reward wrapper."""
        to_pickle['sticky_actions'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of this reward wrapper."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle

    def reward(self, reward):
        """Computes the augmented reward based on defensive actions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward),
                      "stop_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for i, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # If the ball is owned by our team (left team in this case)
                # Check for pass effectiveness
                if self.sticky_actions_counter[9]:  # Assuming index 9 denotes passing action
                    components["pass_reward"][i] = self.pass_reward
                    reward[i] += components["pass_reward"][i]
                
                # Check for stopping effectiveness
                if np.linalg.norm(o['left_team_direction'][o['active']]) < 0.01:
                    components["stop_reward"][i] = self.stop_reward
                    reward[i] += components["stop_reward"][i]

                # Tackling opponent who has the ball
                if 'right_team_active' in o and not o['right_team_active'][o['ball_owned_player']]:
                    components["tackle_reward"][i] = self.tackle_reward
                    reward[i] += components["tackle_reward"][i]

        return reward, components

    def step(self, action):
        """Perform a step using the given action, augment reward based on defense efficacy, and report back the new state."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update the sticky actions tracker
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for idx, action_state in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{idx}"] = action_state
        
        return observation, reward, done, info
