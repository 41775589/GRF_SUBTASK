from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Set initial ball position near the agent
    builder.SetBallPosition(0.45, 0.0)

    # Setting up the left team (controlled team in the environment)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.45, 0.0, e_PlayerRole_LB)  # This is the controllable agent being trained for dribbling

    # Setting up the right team (opponent, automated by the environment)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    # Opponents added to pressure the agent from different directions
    builder.AddPlayer(0.5, 0.1, e_PlayerRole_CB)
    builder.AddPlayer(0.5, -0.1, e_PlayerRole_CB)
    builder.AddPlayer(0.55, 0.0, e_PlayerRole_CM)
