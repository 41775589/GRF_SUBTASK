import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a tailored defensive training reward to promote fast transitions between movement states."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the sticky actions counter and any other stateful elements related to reward calculation."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Include custom state elements for serialization."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore state from serialized format."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on the agent's ability to effectively stop and start against offensive moves."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
        
        components = {"base_score_reward": reward.copy(),
                      "stop_start_reward": [0.0] * len(reward)}
        
        for index, player_obs in enumerate(observation):
            # Apply a positive reward for transitioning rapidly and appropriately between moving and stationary states.
            if player_obs['sticky_actions'][8] == 1:  # Action 8 refers to "action_sprint"
                # Reward the agent for quickly transitioning into a sprint from a non-sprint state if near an opponent who possesses the ball.
                if player_obs['ball_owned_team'] != 0 and np.sum(player_obs['sticky_actions'][:8]) == 0:
                    components["stop_start_reward"][index] = 0.3  # Reward for starting to sprint appropriately
            if player_obs['sticky_actions'][8] == 0:
                # Reward for stopping sprint effectively when near the ball carrier from the opposing team and shifting to defense position.
                proximity_to_ball = np.linalg.norm(np.array(player_obs['ball'][:2]) - player_obs['left_team'][player_obs['active']])
                if player_obs['ball_owned_team'] == 1 and proximity_to_ball < 0.1:
                    components["stop_start_reward"][index] = 0.5  # Reward for stopping sprint at the right time

            # Calculate the final reward by combining base game rewards with the defensive reward component.
            reward[index] = components["base_score_reward"][index] + components["stop_start_reward"][index]

        return reward, components

    def step(self, action):
        """Execute environment steps and apply modified rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Pack the final reward and components into info returned from the environment.
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Populate extra info with sticky action counts.
        obs = self.env.unwrapped.observation()
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] = action
                    info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
