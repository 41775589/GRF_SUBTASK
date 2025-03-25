import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense dribbling and evasion reward to promote advanced ball control under pressure."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.evasion_progress_flag = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.evasion_progress_flag = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.evasion_progress_flag
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.evasion_progress_flag = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "dribble_evasion_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        # Setting parameters for dribbling and evasion-based rewards
        evasion_bonus = 0.05
        dribble_sticky = 9

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if reward[rew_index] == 1:
                # When a goal is scored - agent is rewarded
                reward[rew_index] += 0.5
                continue

            # Check if this player has the ball and is dribbling
            if 'ball_owned_player' in o and 'active' in o and o['ball_owned_player'] == o['active']:
                if o['sticky_actions'][dribble_sticky]:
                    components["dribble_evasion_reward"][rew_index] = evasion_bonus
                    reward[rew_index] += 1.5 * components["dribble_evasion_reward"][rew_index]

                    # Check if the agent is successfully evading a nearby opponent
                    player_pos = o['left_team'] if o['ball_owned_team'] == 0 else o['right_team']
                    opponents = o['right_team'] if o['ball_owned_team'] == 0 else o['left_team']
                    player_idx = o['active']

                    # Calculate distance to nearest opponent
                    min_distance = min(np.linalg.norm(player_pos[player_idx] - opp, ord=2) for opp in opponents)
                    if min_distance < 0.1:  # If really close to an opponent
                        if rew_index not in self.evasion_progress_flag:
                            self.evasion_progress_flag[rew_index] = True
                            reward[rew_index] += 2.0  # Reward more for evading very close defenders

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
