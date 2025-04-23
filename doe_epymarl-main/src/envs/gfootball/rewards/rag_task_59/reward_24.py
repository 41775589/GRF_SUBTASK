import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized goalkeeper coordination reward."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Initialize the number of checkpoints and their reward value
        self.backup_checkpoints = {}
        self.clearance_checkpoints = {}
        self.num_backup_checkpoints = 5  # Example: five stages of backup near the goal
        self.num_clearance_checkpoints = 3  # Example: three target players to clear ball to
        self.checkpoint_increment = 0.05
        self.clear_ball_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.backup_checkpoints = {}
        self.clearance_checkpoints = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['backup_checkpoints'] = self.backup_checkpoints
        to_pickle['clearance_checkpoints'] = self.clearance_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.backup_checkpoints = from_pickle['backup_checkpoints']
        self.clearance_checkpoints = from_pickle['clearance_checkpoints']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "checkpoint_backup_reward": [0.0] * len(reward),
            "checkpoint_clearance_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            team_role = o.get('left_team_roles') if rew_index in o.get('left_team') else o.get('right_team_roles')

            # Reward for backup support near goalkeeper (assuming goalkeeper role index is 0)
            if o['active'] in team_role and team_role[o['active']] == 0:
                distance = np.linalg.norm(o['ball'][:2])
                # Example logic to reward moving closer to the goal to support goalkeeper
                if distance < 0.5:  # Within 50% distance to goal
                    steps_to_backup = int((0.5 - distance) / 0.1)
                    if steps_to_backup > self.backup_checkpoints.get(rew_index, 0):
                        components["checkpoint_backup_reward"][rew_index] = self.checkpoint_increment * steps_to_backup
                        reward[rew_index] += components["checkpoint_backup_reward"][rew_index]
                        self.backup_checkpoints[rew_index] = steps_to_backup

            # Reward for efficient ball clearing
            if o['ball_owned_team'] == 0:
                clear_target_distance = min([np.linalg.norm(o['ball'][:2] - player) for player in o['left_team']])
                # Example logic for ball clearing when a player has the ball
                if clear_target_distance < 0.3:  # Assuming clearance to a player within 30% distance 
                    steps_to_clear = int((0.3 - clear_target_distance) / 0.1)
                    if steps_to_clear > self.clearance_checkpoints.get(rew_index, 0):
                        components["checkpoint_clearance_reward"][rew_index] = self.clear_ball_reward * steps_to_clear
                        reward[rew_index] += components["checkpoint_clearance_reward"][rew_index]
                        self.clearance_checkpoints[rew_index] = steps_to_clear

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
            for i, action_elem in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_elem
        return observation, reward, done, info
