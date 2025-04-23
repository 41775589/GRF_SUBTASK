import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering stop-and-sprint techniques."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stop_actions = np.zeros(10, dtype=int)  # Count the stops for each agent
        self.move_sprint_transitions = np.zeros(10, dtype=int)  # Count transitions from moving to sprinting

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stop_actions = np.zeros(10, dtype=int)
        self.move_sprint_transitions = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'stop_actions': self.stop_actions,
            'move_sprint_transitions': self.move_sprint_transitions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.stop_actions = from_pickle['CheckpointRewardWrapper']['stop_actions']
        self.move_sprint_transitions = from_pickle['CheckpointRewardWrapper']['move_sprint_transitions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "stop_reward": np.zeros_like(reward), 
                      "sprint_transition_reward": np.zeros_like(reward)}

        if observation is None:
            return reward, components

        for i, (obs, rew) in enumerate(zip(observation, reward)):
            active_player = obs['active']
            sticky_actions = obs['sticky_actions']
            
            # Reward for stopping after movement (stop when moving quickly)
            if not sticky_actions[8] and any(sticky_actions[:8]):  # no sprint and any directional is active
                if self.sticky_actions_counter[i] > 1:
                    components["stop_reward"][i] += 0.1 
                    self.stop_actions[i] += 1

            # Reward transitions from moving to sprint
            if sticky_actions[8] and not self.sticky_actions_counter[i]:  # sprint now and was not sprinting
                if any(sticky_actions[:8]):  # any direction action is active
                    components["sprint_transition_reward"][i] += 0.2
                    self.move_sprint_transitions[i] += 1

            self.sticky_actions_counter[i] = sticky_actions[8]  # Update sprint action tracker

            # Summing up the rewards
            reward[i] = rew + components["stop_reward"][i] + components["sprint_transition_reward"][i]

        return reward, components

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
