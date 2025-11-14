from . import *
def build_scenario(builder):
    builder.config().game_duration = 400
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True

    # Set the ball position near the center, but slightly towards the opponent's goal
    builder.SetBallPosition(0.2, 0.0)

    # Setting up the left team (our team)
    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Goalkeeper
    builder.AddPlayer(-0.5, 0.0, e_PlayerRole_CB)  # Central defender having the main training goal

    # Setting up the right team (opponent team)
    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)  # Opponent's Goalkeeper
    # Placing two opponent attackers to challenge our central defender
    builder.AddPlayer(0.3, -0.1, e_PlayerRole_CF, lazy=False)  # Center forward near the ball
    builder.AddPlayer(0.3, 0.1, e_PlayerRole_CF, lazy=False)  # Another Center forward near the ball

    # This setup ensures that the central defender (agent) frequently interacts with opponent
    # forwards in a realistic match situation, providing ample opportunities to train on
    # dispossessing skills, like Sliding and Stop-Moving, effectively in the game context.
