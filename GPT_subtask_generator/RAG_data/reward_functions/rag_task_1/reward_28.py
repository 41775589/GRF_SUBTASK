import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on offensive maneuvers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize checkpoints with different phases of the match
        self.checkpoint_rewards = {'attack': 0.1, 'midfield_control': 0.05, 'defense_transition': 0.05}
        self.collected_checkpoints = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.collected_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy()}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            game_stage = self.determine_game_stage(o)
            if game_stage in self.checkpoint_rewards:
                if game_stage not in self.collected_checkpoints:
                    # Rewarding for entering a new game stage
                    reward[rew_index] += self.checkpoint_rewards[game_stage]
                    components[game_stage] = self.checkpoint_rewards[game_stage]
                    self.collected_checkpoints[game_stage] = True

        return reward, components

    def determine_game_stage(self, observation):
        """Determine the stage of the game based on ball position and team possession."""
        ball_pos = observation['ball'][0]  # ball x position
        if observation['ball_owned_team'] == 0:  # controlled by left team
            if ball_pos > 0.5:
                return 'attack'
            elif -0.5 < ball_pos <= 0.5:
                return 'midfield_control'
            else:
                return 'defense_transition'
        return None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            if isinstance(value, list):
                info[f"component_{key}"] = sum(value)
            else:
                info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
