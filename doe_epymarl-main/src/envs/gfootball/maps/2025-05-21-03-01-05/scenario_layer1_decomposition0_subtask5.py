from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(0.5, 0.1)  # Ball starts on the flank

    # Left team setup 
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(0.5, 0.1, e_PlayerRole_LM, controllable=True)  # Our single player (left midfielder)

    # Right team setup
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent Goalkeeper
    builder.AddPlayer(0.5, 0.2, e_PlayerRole_CB)  # One center back near our player
    builder.AddPlayer(0.6, -0.1, e_PlayerRole_RB)  # One right back inside to help with flanks
