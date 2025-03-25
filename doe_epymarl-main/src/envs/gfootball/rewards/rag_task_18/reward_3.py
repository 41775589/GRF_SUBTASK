import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to enhance the synergistic effectiveness of central midfield play."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.episode_player_statistics = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.episode_player_statistics = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.episode_player_statistics
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.episode_player_statistics = from_pickle.get('CheckpointRewardWrapper', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_synergy_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for i, rew in enumerate(reward):
            o = observation[i]
            ball_owner_team = o.get('ball_owned_team', -1)
            if ball_owner_team == 0: # Left team has the ball
                active_player = o.get('active', -1)
                player_role = o['left_team_roles'][active_player] if active_player != -1 else None
                if player_role in [4, 5, 6]: # 4 = DM, 5 = CM, 6 = LM
                    # Increment managed moves by midfield players
                    player_stats = self.episode_player_statistics.setdefault(i, {'controlled_transitions': 0})
                    player_stats['controlled_transitions'] += 1
                    # Reward synergy for controlled midfield plays (for simplicity: each 5 controlled transitions)
                    if player_stats['controlled_transitions'] % 5 == 0:
                        components["midfield_synergy_bonus"][i] = 0.5
                        reward[i] += components["midfield_synergy_bonus"][i]
        
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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
