import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a custom reward for offensive strategies between midfielders and strikers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control_checkpoints = set()
        self.coordinated_attack_bonus = 0.5  # Bonus reward for coordinated actions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control_checkpoints = set()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'midfield_checkpoints': self.midfield_control_checkpoints
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_info = from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = state_info['sticky_actions_counter']
        self.midfield_control_checkpoints = state_info['midfield_checkpoints']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "coordinated_attack_bonus": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        midfield_positions = [1, 2, 4]  # Role indices that typically correspond to midfielders

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 0:  # Our team controls the ball
                ball_owner = o['ball_owned_player']
                player_role = o['left_team_roles'][ball_owner]
                if player_role in midfield_positions:
                    self.midfield_control_checkpoints.add(ball_owner)
                elif len(self.midfield_control_checkpoints) > 0 and player_role == 9: # Striker has the ball after midfielder
                    components["coordinated_attack_bonus"][rew_index] = self.coordinated_attack_bonus
                    reward[rew_index] += self.coordinated_attack_bonus

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
                self.sticky_actions_counter[i] += int(action)
        return observation, reward, done, info
