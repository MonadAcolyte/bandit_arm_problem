from abc import ABC, abstractmethod

import numpy as np


########
# ARMS #
########

class Arm(ABC):
	"""
	Abstract base class for an Arm in the Multi-Armed Bandit problem.
	"""

	def __init__(self):
		pass

	@abstractmethod
	def pull(self) -> float:
		"""
		Simulates pulling the arm and returns a reward.

		:return: The generated reward.
		:rtype: float
		"""
		pass

	@abstractmethod
	def expectation(self) -> float:
		"""
		Returns the mathematical expectation of the arm's reward distribution.

		:return: The expected value of the reward.
		:rtype: float
		"""
		pass


class DynamicArm(Arm, ABC):
	"""
	Abstract base class for dynamic arms whose reward distribution drifts over time.

	Extends :class:`Arm` with :meth:`update` (which advances the distribution
	one time-step) and :meth:`reset` (which restores the initial parameters
	between independent runs).
	"""

	@abstractmethod
	def update(self):
		"""
		Advances the arm's distribution parameters by one time-step.
		Called once per simulation step when the environment is dynamic.
		"""
		pass

	@abstractmethod
	def reset(self):
		"""
		Resets all mutable distribution parameters to their initial values.
		Called between independent simulation runs to ensure reproducibility.
		"""
		pass


####################
# STATIC ARM TYPES #
####################

class NormalArm(Arm):
	"""
	Arm with a Normal (Gaussian) reward distribution.

	:param mu: Mean of the normal distribution.
	:type mu: float
	:param sigma: Standard deviation of the normal distribution.
	:type sigma: float
	"""

	def __init__(self, mu: float, sigma: float):
		super().__init__()
		self.mu = mu
		self.sigma = sigma

	def pull(self) -> float:
		return float(np.random.normal(self.mu, self.sigma))

	def expectation(self) -> float:
		return self.mu


class UniformArm(Arm):
	"""
	Arm with a Uniform reward distribution.

	:param lower: Lower bound.
	:type lower: float
	:param upper: Upper bound.
	:type upper: float
	"""

	def __init__(self, lower: float, upper: float):
		super().__init__()
		self.lower = lower
		self.upper = upper

	def pull(self) -> float:
		return float(np.random.uniform(self.lower, self.upper))

	def expectation(self) -> float:
		return (self.lower + self.upper) / 2


class TriangularArm(Arm):
	"""
	Arm with a Triangular reward distribution.

	:param lower: Lower limit.
	:type lower: float
	:param mode: Peak of the distribution.
	:type mode: float
	:param upper: Upper limit.
	:type upper: float
	"""

	def __init__(self, lower: float, mode: float, upper: float):
		super().__init__()
		self.lower = lower
		self.mode = mode
		self.upper = upper

	def pull(self) -> float:
		return float(np.random.triangular(self.lower, self.mode, self.upper))

	def expectation(self) -> float:
		return (self.lower + self.mode + self.upper) / 3


class LogNormalArm(Arm):
	"""
	Arm with a Lognormal reward distribution.

	:param mean: Mean of the underlying normal distribution.
	:type mean: float
	:param sigma: Standard deviation of the underlying normal distribution.
	:type sigma: float
	"""

	def __init__(self, mean: float, sigma: float):
		super().__init__()
		self.mean = mean
		self.sigma = sigma

	def pull(self) -> float:
		return float(np.random.lognormal(self.mean, self.sigma))

	def expectation(self) -> float:
		return np.exp(self.mean + self.sigma ** 2 / 2)


class RayleighArm(Arm):
	"""
	Arm with a Rayleigh reward distribution.

	:param scale: The scale parameter of the distribution.
	:type scale: float
	"""

	def __init__(self, scale: float):
		super().__init__()
		self.scale = scale

	def pull(self) -> float:
		return float(np.random.rayleigh(self.scale))

	def expectation(self) -> float:
		return self.scale * np.sqrt(np.pi / 2)


#####################
# DYNAMIC ARM TYPES #
#####################

class NormalArmDynamic(DynamicArm):
	"""
	Dynamic Arm with a Normal reward distribution that drifts over time.

	:param mu: Initial mean.
	:type mu: float
	:param sigma: Initial standard deviation.
	:type sigma: float
	:param dmu: Step-wise change in mean.
	:type dmu: float
	:param dsigma: Step-wise change in standard deviation.
	:type dsigma: float
	"""

	def __init__(self, mu: float, sigma: float, dmu: float, dsigma: float):
		super().__init__()
		self.mu = mu
		self.sigma = sigma
		self.dmu = dmu
		self.dsigma = dsigma
		self.raw_mu = mu
		self.raw_sigma = sigma

	def pull(self) -> float:
		return float(np.random.normal(self.mu, self.sigma))

	def update(self):
		self.mu += self.dmu
		self.sigma += self.dsigma

	def reset(self):
		self.mu = self.raw_mu
		self.sigma = self.raw_sigma

	def expectation(self) -> float:
		return self.mu


class UniformArmDynamic(DynamicArm):
	"""
	Dynamic Arm with a Uniform reward distribution that drifts over time.

	:param lower: Initial lower bound.
	:type lower: float
	:param upper: Initial upper bound.
	:type upper: float
	:param dlower: Step-wise change in lower bound.
	:type dlower: float
	:param dupper: Step-wise change in upper bound.
	:type dupper: float
	"""

	def __init__(self, lower: float, upper: float, dlower: float, dupper: float):
		super().__init__()
		self.lower = lower
		self.upper = upper
		self.dlower = dlower
		self.dupper = dupper
		self.raw_lower = lower
		self.raw_upper = upper

	def pull(self) -> float:
		return float(np.random.uniform(self.lower, self.upper))

	def update(self):
		self.lower += self.dlower
		self.upper += self.dupper

	def reset(self):
		self.lower = self.raw_lower
		self.upper = self.raw_upper

	def expectation(self) -> float:
		return (self.lower + self.upper) / 2


class TriangularArmDynamic(DynamicArm):
	"""
	Dynamic Arm with a Triangular reward distribution whose peak drifts over time.

	:param lower: Lower limit (fixed).
	:type lower: float
	:param mode: Initial peak.
	:type mode: float
	:param upper: Upper limit (fixed).
	:type upper: float
	:param dmode: Step-wise change in the peak.
	:type dmode: float
	"""

	def __init__(self, lower: float, mode: float, upper: float, dmode: float):
		super().__init__()
		self.lower = lower
		self.mode = mode
		self.upper = upper
		self.dmode = dmode
		self.raw_mode = mode

	def pull(self) -> float:
		return float(np.random.triangular(self.lower, self.mode, self.upper))

	def update(self):
		self.mode += self.dmode

	def reset(self):
		self.mode = self.raw_mode

	def expectation(self) -> float:
		return (self.lower + self.mode + self.upper) / 3


class LogNormalArmDynamic(DynamicArm):
	"""
	Dynamic Arm with a Lognormal reward distribution that drifts over time.

	:param mean: Initial mean of the underlying normal distribution.
	:type mean: float
	:param sigma: Initial standard deviation.
	:type sigma: float
	:param dmean: Step-wise change in mean.
	:type dmean: float
	:param dsigma: Step-wise change in standard deviation.
	:type dsigma: float
	"""

	def __init__(self, mean: float, sigma: float, dmean: float, dsigma: float):
		super().__init__()
		self.mean = mean
		self.sigma = sigma
		self.dmean = dmean
		self.dsigma = dsigma
		self.raw_mean = mean
		self.raw_sigma = sigma

	def pull(self) -> float:
		return float(np.random.lognormal(self.mean, self.sigma))

	def update(self):
		self.mean += self.dmean
		self.sigma += self.dsigma

	def reset(self):
		self.mean = self.raw_mean
		self.sigma = self.raw_sigma

	def expectation(self) -> float:
		return np.exp(self.mean + self.sigma ** 2 / 2)


class RayleighArmDynamic(DynamicArm):
	"""
	Dynamic Arm with a Rayleigh reward distribution that drifts over time.

	:param scale: Initial scale parameter.
	:type scale: float
	:param dscale: Step-wise change in scale.
	:type dscale: float
	"""

	def __init__(self, scale: float, dscale: float):
		super().__init__()
		self.scale = scale
		self.dscale = dscale
		self.raw_scale = scale

	def pull(self) -> float:
		return float(np.random.rayleigh(self.scale))

	def update(self):
		self.scale += self.dscale

	def reset(self):
		self.scale = self.raw_scale

	def expectation(self) -> float:
		return self.scale * np.sqrt(np.pi / 2)
