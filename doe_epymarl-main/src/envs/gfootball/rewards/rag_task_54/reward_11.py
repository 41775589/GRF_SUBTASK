import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards effective collaboration between shooters and passers in football gameplay."""

    def __init__(self, env):
        super().__init__(env)
        self.collaboration_reward = 0.2  # Reward increment for effective collaboration
        self.shooter_bonus = 1.0  # Additional bonus for scoring through collaboration
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Stores states useful for checkpointing collaboration metrics."""
        to_pickle['CheckpointRewardCollaboration'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Loads states relevant for checkpointing collaboration metrics."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Enhances rewards by encouraging collaboration between passers and shooters."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "collaboration_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Collaborative reward mechanism: Passes that result in shots on goal
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the ball is with the active player of the team
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and 'ball_owned_player' in o:
                active_player_index = o['active']
                ball_owner_index = o['ball_owned_player']

                # Reward for collaboration: ball passed and immediately shot by another player
                if active_player_index != ball_owner_index and 'action' in o:
                    action = o['sticky_actions'][9]  # Assuming index 9 corresponds to the shooting action
                    if action:
                        # If the shooter scores, add collaboration reward and shooter bonus
                        if reward[rew_index] > 0:
                            components["collaboration_reward"][rew_index] = self.shooter_bonus
                            reward[rew_index] += components["collaboration_reward"][rew_index]
                        else:
                            # Reward for attempts that might have been initiated from a pass
                            components["collaboration_reward"][rew_index] = self.collaboration_reward
                            reward[rew_index] += components["collaboration_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Executes a step in the environment, appending additional reward information."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action_item in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_item
        return observation, reward, done, info
