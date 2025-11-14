from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    
    builder.SetBallPosition(0.1, 0.0)  # Ball is closer to the defense to start the defense to counter-attack transition

    # Setting up the left team (agent's team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.5, -0.2, e_PlayerRole_LB)  # Left Back, one of the agents
    builder.AddPlayer(-0.5, 0.2, e_PlayerRole_RB)   # Right Back, one of the agents
    # Other players (AI-controlled)
    builder.AddPlayer(-0.7, -0.1, e_PlayerRole_CB)
    builder.AddPlayer(-0.7, 0.1, e_PlayerRole_CB)
    builder.AddPlayer(-0.2, -0.3, e_PlayerRole_CM)
    builder.AddPlayer(-0.2, 0.3, e_PlayerRole_CM)
    builder.AddPlayer(0.0, -0.1, e_PlayerRole_LM)
    builder.AddPlayer(0.0, 0.1, e_PlayerRole_RM)
    builder.AddPlayer(0.2, 0.0, e_PlayerRole_CF)

    # Setting up the right team (opponent)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(-0.8, -0.25, e_PlayerRole_LB)
    builder.AddPlayer(-0.8, 0.25, e_PlayerRole_RB)
    builder.AddPlayer(-0.6, -0.15, e_PlayerRole_CB)
    builder.AddPlayer(-0.6, 0.15, e_PlayerRole_CB)
    builder.AddPlayer(-0.4, -0.2, e_PlayerRole_CM)
    builder.AddPlayer(-0.4, 0.2, e_PlayerRole_CM)
    builder.AddPlayer(-0.2, -0.25, e_PlayerRole_LM)
    builder.AddPlayer(-0.2, 0.25, e_PlayerRole_RM)
    builder.AddPlayer(0.0, 0.0, e_PlayerRole_CF)
