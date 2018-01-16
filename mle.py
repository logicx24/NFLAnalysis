from __future__ import division
import numpy as np
import nflgame
import random

class NflMaximumLikelihoodEstimator:

	teams_array = ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAC', 'JAX', 'KC', 'LA', 'LAC', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'OAK', 'PHI', 'PIT', 'SD', 'SEA', 'SF', 'STL', 'TB', 'TEN', 'WAS']

	"""
	current_weights: [0, ..., n]: n weights
	games_matrix = w[i][j]: games i played against j
	wins_array = [w_0, ..., w_n]: wins_array[i] = total wins by team i

	Function:

	R_a = (Number of Wins by A) / ( (Games A played against B) / (R_a + R_b) + ... + (Games A played against N) / (R_a + R_n))
	"""
	def optimization_function(self, current_weights, games_matrix, wins_array):
		out = np.array([i for i in current_weights])
		for i, curr_weight in enumerate(current_weights):
			running_sum = 0
			for j, opposing_weight in enumerate(current_weights):
				if curr_weight + opposing_weight > 0:
					running_sum += games_matrix[i][j] / (curr_weight + opposing_weight)
			if running_sum > 0:
				out[i] = wins_array[i] / running_sum
		return out


	"""
	Essentially, run numerical optimization on the above function until the weights stop changing.
	"""
	def iterate(self, games_matrix, wins_array, initial_weights=None):
		if not initial_weights:
			initial_weights = np.array([1.0] * len(wins_array))

		current_weights = np.array([0.0] * len(wins_array))
		next_weights = initial_weights
		while not np.allclose(current_weights, next_weights):
			current_weights, next_weights = next_weights, self.optimization_function(next_weights, games_matrix, wins_array)
		return next_weights


	def generate_matrices(self, season, weeks, kind="REG"):
		games = nflgame.games(season, week=weeks, kind=kind)
		games_matrix = np.identity(len(self.teams_array))

		team_index_map = {team : team_index for team_index, team in enumerate(self.teams_array)}
		game_wins_dict = {}

		for game in games:
			if game.winner not in game_wins_dict:
				game_wins_dict[game.winner] = 0
			game_wins_dict[game.winner] += self.wins_update_formula(game)

			games_matrix[team_index_map[game.winner]][team_index_map[game.loser]] += 1
			games_matrix[team_index_map[game.loser]][team_index_map[game.winner]] += 1

		wins_array = np.array([0] * len(self.teams_array))
		for team in game_wins_dict:
			wins_array[team_index_map[team]] = game_wins_dict[team]

		return games_matrix, wins_array

	def wins_update_formula(self, game):
		score_difference = abs(game.score_away - game.score_home)
		score_sum = game.score_away + game.score_home

		return 0.6 + (0.4 * (score_difference / score_sum))

	def generate_rankings(self, season, week):
		games_matrix, wins_array = self.generate_matrices(season, week)
		weights = self.iterate(games_matrix, wins_array)

		return {team : weights[i] for i, team in enumerate(self.teams_array)}


def main():
	w = NflMaximumLikelihoodEstimator()
	inter_weights = w.generate_rankings(2017, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])

	print(inter_weights)
	print([x[0] for x in sorted(inter_weights.items(), key=lambda x: -x[1])])




if __name__ == "__main__":
	main()