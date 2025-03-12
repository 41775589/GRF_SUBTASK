from . import *
def build_scenario(builder):
    # Setting the basic configuration for the training scenario
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Setting initial ball position near the opposing team's goal area to simulate high pressure defense situations
    builder.SetBallPosition(0.7, 0.0)

    # Configure the left team (defensive training team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)  # Goalkeeper
    # Defensive player being trained on sliding tackles
    builder.AddPlayer(0.65, 0.0, e_PlayerRole_CB, controllable=True)

    # Configure the right team (opposing attackers)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, lazy=True)  # Lazy goalkeeper
    # Opposing forward positioned to challenge the defender
    builder.AddPlayer(0.75, 0.0, e_PlayerRole_CF, lazy=False, controllable=False)
