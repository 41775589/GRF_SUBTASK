import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a specialized reward for goalkeeper training focused on
    shot-stopping, quick reflexes, and initiating counter-attacks with accurate passes.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward based on goalkeeper performance metrics.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_reward": [0.0, 0.0]}  # Assuming two agents with index 0 being the goalkeeper

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            goalkeeper_index = 0 if rew_index == 0 else None  # Assuming the first agent is the goalkeeper if there are two agents

            if goalkeeper_index is not None:
                # Reward goalkeepers for successful saves and distribution quality
                # Assuming the goalkeeper is on the left team (team index 0)
                if o['ball_owned_team'] == 1 and np.linalg.norm(o['ball'] - o['left_team'][goalkeeper_index][:2]) < 0.1:
                    # Ball is near the goalkeeper, potentially a save scenario
                    components['goalkeeper_reward'][rew_index] = 0.5  # Reward for being in position to save

                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == goalkeeper_index:
                    # Ball is controlled by the goalkeeper
                    if 'action_dribble' in o['sticky_actions'] or 'action_bottom' in o['sticky_actions']:
                        # Good distribution potentially starting a counter-attack
                        components['goalkeeper_reward'][rew_index] += 1.0

                # Total reward for this agent is modified by the goalkeeper components
                reward[rew_index] += components['goalkeeper_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        return observation, reward, done, info
