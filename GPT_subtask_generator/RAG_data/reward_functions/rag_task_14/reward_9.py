import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a rewards for 'sweeper' actions, which include:
    1. Clearing the ball from the defensive zone,
    2. Performing critical last-man tackles,
    3. Supporting the stopper by covering positions and executing fast recoveries.
    """

    def __init__(self, env):
        super().__init__(env)
        # Initialize counters for the number of effective defenses and clearances made.
        self.effective_defense_count = 0
        self.effective_clearance_count = 0
        self.covering_position_count = 0
        # The rewards for each of the actions.
        self.defense_reward = 0.5
        self.clearance_reward = 0.7
        self.covering_reward = 0.3
        # Action counters to avoid double counting within a single step.
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        # Reset the sticky action counters and action effectiveness counters on environment reset.
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.effective_defense_count = 0
        self.effective_clearance_count = 0
        self.covering_position_count = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        # Additional state information related to this reward wrapper can be added to the pickle.
        state = self.env.get_state(to_pickle)
        state.update({
            'effective_defense_count': self.effective_defense_count,
            'effective_clearance_count': self.effective_clearance_count,
            'covering_position_count': self.covering_position_count
        })
        return state

    def set_state(self, state):
        # Load the state for the additional information related to this reward wrapper.
        from_pickle = self.env.set_state(state)
        self.effective_defense_count = from_pickle['effective_defense_count']
        self.effective_clearance_count = from_pickle['effective_clearance_count']
        self.covering_position_count = from_pickle['covering_position_count']

    def reward(self, reward):
        # Fetch the latest observation to evaluate 'sweeper' actions.
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components
       
        # Initialize the reward component dictionary.
        components.update({
            "defense_reward": 0.0,
            "clearance_reward": 0.0,
            "covering_reward": 0.0
        })

        # Evaluate the actions based on observations.
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if player has executed an effective defensive action.
            if self.is_effective_defense(o):
                components["defense_reward"] += self.defense_reward
                self.effective_defense_count += 1

            # Check if player has cleared the ball effectively from the defensive zone.
            if self.is_effective_clearance(o):
                components["clearance_reward"] += self.clearance_reward
                self.effective_clearance_count += 1
            
            # Check if player is effectively covering positions.
            if self.is_effective_covering(o):
                components["covering_reward"] += self.covering_reward
                self.covering_position_count += 1

            # Sum up the components to define the final reward for this time step.
            reward[rew_index] += (
                components["defense_reward"] +
                components["clearance_reward"] +
                components["covering_reward"]
            )
        
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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info

    def is_effective_defense(self, observation):
        # Placeholder function: Define logic to determine if an effective defense has been conducted.
        return False

    def is_effective_clearance(self, observation):
        # Placeholder function: Define logic to determine if an effective clearance has been conducted.
        return False

    def is_effective_covering(self, observation):
        # Placeholder function: Define logic to determine if the player is covering positions effectively.
        return False
