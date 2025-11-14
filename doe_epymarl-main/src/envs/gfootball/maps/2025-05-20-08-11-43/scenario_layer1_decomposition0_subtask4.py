from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(0.0, 0.0)  # Start with the ball in the center

    # Setting up the left team with training agent
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.0, 0.0, e_PlayerRole_CM, controllable=True)  # Our main agent

    # Setting up the right team (opponent)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    # Opponent players
    builder.AddPlayer(-0.2, 0.1, e_PlayerRole_CM)
    builder.AddPlayer(-0.2, -0.1, e_PlayerRole_CM)
    builder.AddPlayer(-0.3, 0.2, e_PlayerRole_CB)
    builder.AddPlayer(-0.3, -0.2, e_PlayerRole_CB)
