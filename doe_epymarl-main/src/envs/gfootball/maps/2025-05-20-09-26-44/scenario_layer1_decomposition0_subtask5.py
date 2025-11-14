from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    builder.SetBallPosition(-0.3, 0.0)  # Central, slightly towards the left team's goal

    # Setting up the left team (trained agent's team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    # Main trained player, positioned to start attacks from the back and work on dribbles and passes
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_LB, controllable=True)

    # Extra AI-controlled players on the left team to simulate realistic game conditions
    builder.AddPlayer(-0.7, 0.15, e_PlayerRole_CB)
    builder.AddPlayer(-0.7, -0.15, e_PlayerRole_CB)
    builder.AddPlayer(-0.5, 0.2, e_PlayerRole_RB)

    # Setting up the right team (opponent team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    # Opponent players are positioned to challenge the trained player's ability in passing and dribbling
    builder.AddPlayer(-0.3, 0.1, e_PlayerRole_CB)
    builder.AddPlayer(-0.3, -0.1, e_PlayerRole_CB)
    builder.AddPlayer(-0.2, 0.2, e_PlayerRole_CM)
    builder.AddPlayer(-0.2, -0.2, e_PlayerRole_CM)
