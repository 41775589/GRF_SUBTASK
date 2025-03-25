import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the synergy and pace management of central midfielders."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._synergy_points = 0.05
        self._pace_management_points = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_picle['CheckpointRewardWrapper']["sticky_actions_counter"])
        return from_picle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "synergy_points": [0.0] * len(reward),
                      "pace_management_points": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Synergy when centrally controlled players pass accurately within a region
            if o['game_mode'] == 0:  # Normal play mode
                # Identify if the player is a CM (central midfield)
                if o['active'] in (4, 5):  # Assuming central midfield roles are 4 and 5
                    # Check if the ball is passed and controlled by a central midfield teammate
                    team_owning_ball = o['ball_owned_team']
                    player_owning_ball = o['ball_owned_player']
                    if team_owning_ball == (0 if o['active'] < len(o['left_team']) / 2 else 1) \
                        and player_owning_ball != -1 and abs(o['active'] - player_owning_ball) <= 2:
                        components['synergy_points'][rew_index] = self._synergy_points
                        reward[rew_index] += components['synergy_points'][rew_index]
            
            # Pace management by slowing down when on lead
            score_diff = o['score'][0] - o['score'][1] if o['active'] < len(o['left_team'])/2 else o['score'][1] - o['score'][0]
            if score_diff > 0:
                sticky_actions = o['sticky_actions']
                if sticky_actions[7] or sticky_actions[3]:  # Check for 'action_bottom_left' or 'action_top_right'
                    components['pace_management_points'][rew_index] = self._pace_management_points
                    reward[rew_index] += components['pace_management_points'][rew_index]

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
