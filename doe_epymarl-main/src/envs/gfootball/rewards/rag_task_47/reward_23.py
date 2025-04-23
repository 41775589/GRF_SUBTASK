import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for performing good sliding tackles 
       during counter-attacks and high-pressure situations near the team's defensive third."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.appropriate_tackles_counter = {0: 0, 1: 0}
        self.incentive_for_tackle = 0.5

    def reset(self):
        """Resets the environment and the rewards related counters."""

        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.appropriate_tackles_counter = {0: 0, 1: 0}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Custom state retrieval to save the added state components."""

        to_pickle['AppropriateTacklesCounter'] = self.appropriate_tackles_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Custom state setter to set the added state components."""

        from_pickle = self.env.set_state(state)
        self.appropriate_tackles_counter = from_pickle.get('AppropriateTacklesCounter', {0: 0, 1: 0})
        return from_pickle

    def reward(self, reward):
        """Calculate the additional reward for successful sliding tackles in high-pressure 
           situations based on ball possession in the defensive area."""

        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        # Process each agent's actions and positions
        for idx, rew in enumerate(reward):
            o = observation[idx]
            if o['game_mode'] != 0:  # Skip modifications if it's not normal game play
                continue

            # Check if the player is in the defensive third and performs a slide
            active_player_pos = o['right_team'][o['active']] if o['ball_owned_team'] == 1 else o['left_team'][o['active']]
            if active_player_pos[0] <= -0.5 and o['sticky_actions'][6]:  # Action 6 is supposed to be the sliding tackle
                # Only reward if the ball is in the defensive third and owned by the attacking team
                if o['ball'][0] <= -0.5 and o['ball_owned_team'] != o['active']:
                    reward[idx] += self.incentive_for_tackle
                    components['tackle_reward'][idx] = self.incentive_for_tackle
                    self.appropriate_tackles_counter[idx] += 1

        return reward, components

    def step(self, action):
        """Apply an action, modified reward calculations, obtain observations and return the result."""

        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Gather additional info for rendering or debugging
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
                    
        return observation, reward, done, info
