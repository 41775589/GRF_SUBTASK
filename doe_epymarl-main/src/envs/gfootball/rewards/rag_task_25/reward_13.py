import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward focused on dribbling and sprinting."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track sticky actions usage

    def reset(self):
        """ Reset sticky actions counter on each episode start. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Package up state to pickle. State pertains to sticky actions for learning state reproduction. """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Set state from unpickled state, primarily sticky actions states. """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """ Reward agents for effective dribbling and sprinting practices. """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_bonus": [0.0]*len(reward),
                      "sprint_bonus": [0.0]*len(reward)}

        if observation is None:
            return reward, components
        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                if o['sticky_actions'][9]:  # action_dribble is indexed at 9
                    components["dribbling_bonus"][rew_index] += 0.05
                
                if o['sticky_actions'][8]:  # action_sprint is indexed at 8
                    components["sprint_bonus"][rew_index] += 0.1

            # Combining base reward and additional bonuses for dribbling and sprinting
            reward[rew_index] += components["dribbling_bonus"][rew_index] + components["sprint_bonus"][rew_index]

        return reward, components

    def step(self, action):
        """ Execute environment step and apply reward wrapper adjustments. """
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
