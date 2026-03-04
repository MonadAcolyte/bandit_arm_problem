from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import numpy as np
from scipy.special import softmax


##############
# ESTIMATORS #
##############

class Estimator(ABC):
	"""
	Abstract base class for reward estimators.

	:param N: The number of arms in the environment.
	:type N: int
	"""

	def __init__(self, N: int) -> None:
		self.N: int = N
		self.estimation: np.ndarray = np.zeros(N, dtype=float)

	def estimate(self, arm: int) -> float:
		"""
		Returns the current estimated reward for a specific arm.

		:param arm: The index of the arm.
		:type arm: int
		:return: The estimated reward.
		:rtype: float
		"""
		return self.estimation[arm]

	def clear(self) -> None:
		"""
		Resets the estimations to zero.
		"""
		self.estimation = np.zeros(self.N, dtype=float)

	@abstractmethod
	def update(self, arm: int, reward: float) -> None:
		"""
		Updates the estimation based on the received reward.

		:param arm: The index of the pulled arm.
		:type arm: int
		:param reward: The reward received from the arm.
		:type reward: float
		"""
		pass


class AverageEstimator(Estimator):
	"""
	Estimator that calculates the simple average of all received rewards for each arm.

	:param N: The number of arms in the environment.
	:type N: int
	"""

	def __init__(self, N: int) -> None:
		super().__init__(N)
		self.cnt: np.ndarray = np.zeros(N, dtype=int)
		self.sum: np.ndarray = np.zeros(N, dtype=float)

	def update(self, arm: int, reward: float) -> None:
		self.cnt[arm] += 1
		self.sum[arm] += reward
		self.estimation[arm] = self.sum[arm] / self.cnt[arm]

	def clear(self) -> None:
		super().clear()
		self.cnt = np.zeros(self.N, dtype=int)
		self.sum = np.zeros(self.N, dtype=float)


class IncrementalUpdateEstimator(Estimator):
	"""
	Estimator that updates values incrementally using a fixed step size.

	:param N: The number of arms in the environment.
	:type N: int
	:param step_size: The learning rate or step size for the update.
	:type step_size: float
	"""

	def __init__(self, N: int, step_size: float) -> None:
		super().__init__(N)
		self.step_size: float = step_size

	def update(self, arm: int, reward: float) -> None:
		self.estimation[arm] += self.step_size * (reward - self.estimation[arm])


class MovingAverageEstimator(Estimator):
	"""
	Estimator that calculates the average reward over a recent rolling window.

	:param N: The number of arms in the environment.
	:type N: int
	:param window_size: The size of the moving window.
	:type window_size: int
	"""

	def __init__(self, N: int, window_size: int) -> None:
		super().__init__(N)
		self.window_size: int = window_size
		self.queues: list[deque[float]] = [deque(maxlen=window_size) for _ in range(N)]

	def update(self, arm: int, reward: float) -> None:
		self.queues[arm].append(reward)
		self.estimation[arm] = np.mean(self.queues[arm])

	def clear(self) -> None:
		super().clear()
		self.queues = [deque(maxlen=self.window_size) for _ in range(self.N)]


##############
# STRATEGIES #
##############

class Strategy(ABC):
	"""
	Abstract base class for action selection strategies.
	"""

	@abstractmethod
	def select(self, estimation: np.ndarray) -> int:
		"""
		Selects an arm based on the current estimations.

		:param estimation: Array of current reward estimates for each arm.
		:type estimation: np.ndarray
		:return: The index of the chosen arm.
		:rtype: int
		"""
		pass


class EpsilonGreedyStrategy(Strategy):
	"""
	Epsilon-greedy selection strategy.

	:param epsilon: The probability of exploring a random arm.
	:type epsilon: float
	"""

	def __init__(self, epsilon: float) -> None:
		self.epsilon: float = epsilon

	def select(self, estimation: np.ndarray) -> int:
		if np.random.random() < self.epsilon:
			return int(np.random.randint(len(estimation)))
		else:
			return int(np.argmax(estimation))


class SoftmaxStrategy(Strategy):
	"""
	Softmax (Boltzmann) selection strategy.

	:param temperature: The temperature parameter controlling exploration vs exploitation.
	:type temperature: float
	"""

	def __init__(self, temperature: float) -> None:
		self.temperature: float = temperature

	def select(self, estimation: np.ndarray) -> int:
		return int(np.random.choice(
			len(estimation), p=softmax(estimation / self.temperature)
		))


#########
# AGENT #
#########

class Agent:
	"""
	Agent that interacts with the Multi-Armed Bandit environment.

	:param strategy: The action selection strategy.
	:type strategy: Strategy
	:param estimator: The reward estimation method.
	:type estimator: Estimator
	"""

	def __init__(self, strategy: Strategy, estimator: Estimator) -> None:
		self.strategy: Strategy = strategy
		self.estimator: Estimator = estimator
		self.cumulative_reward: float = 0.0
		self.cumulative_reward_over_time: list[float] = []
		self.total_selection: int = 0
		self.best_arm_selection: int = 0
		self.percentage_best_arm_selection_over_time: list[float] = []

	def select(self) -> int:
		"""
		Selects an arm using the underlying strategy.

		:return: The index of the chosen arm.
		:rtype: int
		"""
		ret: int = self.strategy.select(self.estimator.estimation)
		self.total_selection += 1
		return ret

	def update(self, arm: int, reward: float) -> None:
		"""
		Updates the agent's estimator and cumulative metrics.

		:param arm: The index of the arm that was pulled.
		:type arm: int
		:param reward: The received reward.
		:type reward: float
		"""
		self.estimator.update(arm, reward)
		self.cumulative_reward += reward
		self.cumulative_reward_over_time.append(self.cumulative_reward)

	def clear(self) -> None:
		"""
		Resets all tracking metrics and the estimator for a new run.
		"""
		self.cumulative_reward = 0.0
		self.cumulative_reward_over_time = []
		self.total_selection = 0
		self.best_arm_selection = 0
		self.percentage_best_arm_selection_over_time = []
		self.estimator.clear()
