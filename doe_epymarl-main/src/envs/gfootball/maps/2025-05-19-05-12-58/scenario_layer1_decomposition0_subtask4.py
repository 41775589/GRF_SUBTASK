from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Setting the ball position near the right team attacker to encourage defensive scenario.
    builder.SetBallPosition(0.5, 0.0)

    # Left team setup (our trained agent is on this side)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.4, 0.0, e_PlayerRole_CB)  # Our trained player specialized in sliding tackles 

    # Right team setup (opponent team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    # Adding multiple strikers close to midfield encourages scenarios where the agent must engage defensively
    builder.AddPlayer(0.4, 0.1, e_PlayerRole_CF)  
    builder.AddPlayer(0.4, -0.1, e_PlayerRole_CF)
    builder.AddPlayer(0.45, 0.0, e_PlayerRole_CF)  # Central forward with the ball, ready to attack
