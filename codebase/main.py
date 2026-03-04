import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from agent import (Agent,
                   AverageEstimator,
                   EpsilonGreedyStrategy,
                   IncrementalUpdateEstimator,
                   MovingAverageEstimator,
                   SoftmaxStrategy)
from arm import (Arm,
                 DynamicArm,
                 LogNormalArm,
                 LogNormalArmDynamic,
                 NormalArm,
                 NormalArmDynamic,
                 RayleighArm,
                 RayleighArmDynamic,
                 TriangularArm,
                 TriangularArmDynamic,
                 UniformArm,
                 UniformArmDynamic)

plt.style.use("ggplot")

# Directory where all output figures are saved
OUTPUT_DIR: Path = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_experiment(
    environment: list[Arm],
    agents: list[Agent],
    steps: int,
    runs: int,
    is_dynamic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Runs the multi-armed bandit simulation.

	:param environment: List of :class:`~arm.Arm` objects representing the environment.
	                    Pass a ``list[DynamicArm]`` when *is_dynamic* is ``True``.
	:type environment: list[Arm]
	:param agents: List of :class:`~agent.Agent` objects to test.
	:type agents: list[Agent]
	:param steps: Number of steps per run.
	:type steps: int
	:param runs: Number of independent runs to average over.
	:type runs: int
	:param is_dynamic: Flag indicating whether the environment changes over time.
	                   When ``True``, *environment* must contain
	                   :class:`~arm.DynamicArm` instances.
	:type is_dynamic: bool
	:return: A tuple of 3-D numpy arrays containing the history of metrics:
	         *(cumulative_rewards, average_rewards, percentage_best_arm)*.
	         Each array has shape ``(runs, len(agents), steps)``.
	:rtype: tuple[np.ndarray, np.ndarray, np.ndarray]
	"""
	num_agents: int = len(agents)

	# Pre-allocate arrays for performance tracking — shape: (runs, agents, steps)
	cum_rewards_history: np.ndarray = np.zeros((runs, num_agents, steps), dtype=float)
	avg_rewards_history: np.ndarray = np.zeros((runs, num_agents, steps), dtype=float)
	pct_best_arm_history: np.ndarray = np.zeros((runs, num_agents, steps), dtype=float)

	for run in range(runs):
		for agent in agents:
			agent.clear()

		for i in range(steps):
			# For static env the best arm is fixed; for dynamic, recompute each step
			best_arm_idx: int = int(np.argmax([arm.expectation() for arm in environment]))

			for a_idx, agent in enumerate(agents):
				selection: int = agent.select()
				reward: float = environment[selection].pull()
				agent.update(arm=selection, reward=reward)

				# Track whether the selected arm was optimal
				if selection == best_arm_idx:
					agent.best_arm_selection += 1

				# Record metrics at the current step
				cum_rewards_history[run, a_idx, i] = agent.cumulative_reward
				avg_rewards_history[run, a_idx, i] = agent.cumulative_reward / (i + 1)
				pct_best_arm_history[run, a_idx, i] = agent.best_arm_selection / (i + 1)

			# Advance the dynamic environment one time-step
			if is_dynamic:
				dynamic_env: list[DynamicArm] = environment  # type: ignore[assignment]
				for arm in dynamic_env:
					arm.update()

		# Reset dynamic arms between runs
		if is_dynamic:
			dynamic_env = environment  # type: ignore[assignment]
			for arm in dynamic_env:
				arm.reset()

	return cum_rewards_history, avg_rewards_history, pct_best_arm_history


def plot_metric(
    data: np.ndarray,
    steps: int,
    labels: list[str],
    colors: list[str],
    title: str,
    ylabel: str,
    filename: str,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
	"""
	Plots the mean of a specific metric over time with standard deviation bounds
	and saves the figure to disk.

	:param data: 3-D numpy array of shape ``(runs, agents, steps)``.
	:type data: np.ndarray
	:param steps: Total number of steps.
	:type steps: int
	:param labels: List of legend labels for each agent.
	:type labels: list[str]
	:param colors: List of colours for each agent line.
	:type colors: list[str]
	:param title: Plot title.
	:type title: str
	:param ylabel: Label for the Y-axis.
	:type ylabel: str
	:param filename: Name of the output file (without directory, with extension).
	:type filename: str
	:param ylim: Optional ``(y_min, y_max)`` axis limits.
	:type ylim: tuple[float, float] or None
	"""
	mean_data: np.ndarray = np.mean(data, axis=0)
	std_data: np.ndarray = np.std(data, axis=0)

	x: range = range(steps)
	plt.figure(figsize=(10, 6))

	for i in range(len(labels)):
		plt.plot(x, mean_data[i], alpha=0.8, color=colors[i], label=labels[i])
		plt.fill_between(
			x, mean_data[i] - std_data[i], mean_data[i] + std_data[i],
			alpha=0.15, color=colors[i]
		)

	plt.xlabel("Steps")
	plt.ylabel(ylabel)
	plt.title(title, color="dodgerblue")
	if ylim is not None:
		plt.ylim(ylim)
	plt.grid(True, alpha=0.8)
	plt.legend()

	outpath: Path = OUTPUT_DIR / filename
	plt.savefig(outpath, dpi=150, bbox_inches="tight")
	plt.close()
	print(f"  Saved: {outpath}")


def main() -> None:
	steps: int = 2000
	runs: int = 50

	# ---------------------------------------------------------
	# STATIC ENVIRONMENT — only Arm (non-dynamic) instances
	# ---------------------------------------------------------
	env_static: list[Arm] = [
		NormalArm(mu=0.1, sigma=1),
		TriangularArm(lower=-0.4, mode=-0.2, upper=0),
		LogNormalArm(mean=0.3, sigma=2),
		RayleighArm(scale=0.4),
		UniformArm(lower=-0.1, upper=1.1),
	]

	agents_static: list[Agent] = [
		Agent(EpsilonGreedyStrategy(epsilon=0.1), AverageEstimator(N=len(env_static))),
		Agent(EpsilonGreedyStrategy(epsilon=0.3), AverageEstimator(N=len(env_static))),
		Agent(EpsilonGreedyStrategy(epsilon=0.5), AverageEstimator(N=len(env_static))),
		Agent(EpsilonGreedyStrategy(epsilon=0.7), AverageEstimator(N=len(env_static))),
		Agent(EpsilonGreedyStrategy(epsilon=0.9), AverageEstimator(N=len(env_static))),
		Agent(SoftmaxStrategy(temperature=1), AverageEstimator(N=len(env_static))),
	]

	labels_static: list[str] = [
		"epsilon=0.1", "epsilon=0.3", "epsilon=0.5",
		"epsilon=0.7", "epsilon=0.9", "softmax",
	]
	colors_static: list[str] = ["lime", "yellow", "orange", "midnightblue", "darkorchid", "brown"]

	print("Running Static Environment Simulations...")
	stat_cum, stat_avg, stat_pct = run_experiment(
		env_static, agents_static, steps, runs, is_dynamic=False
	)

	# ---------------------------------------------------------
	# DYNAMIC ENVIRONMENT — only DynamicArm instances
	# ---------------------------------------------------------
	env_dynamic: list[DynamicArm] = [
		NormalArmDynamic(mu=1, sigma=1, dmu=-0.001, dsigma=0),
		TriangularArmDynamic(lower=-3, mode=-1, upper=3, dmode=0.001),
		LogNormalArmDynamic(mean=2, sigma=2, dmean=-0.003, dsigma=-0.001),
		RayleighArmDynamic(scale=4, dscale=-0.001),
		UniformArmDynamic(lower=-1, upper=11, dlower=0.002, dupper=-0.003),
	]

	agents_dynamic: list[Agent] = [
		Agent(SoftmaxStrategy(temperature=1), AverageEstimator(N=len(env_dynamic))),
		Agent(SoftmaxStrategy(temperature=1), MovingAverageEstimator(N=len(env_dynamic), window_size=20)),
		Agent(SoftmaxStrategy(temperature=1), IncrementalUpdateEstimator(N=len(env_dynamic), step_size=0.1)),
	]

	labels_dynamic: list[str] = [
		"Average Estimator",
		"Moving Average (window=20)",
		"Incremental Update (step=0.1)",
	]
	colors_dynamic: list[str] = ["lime", "yellow", "orange"]

	print("Running Dynamic Environment Simulations...")
	dyn_cum, dyn_avg, dyn_pct = run_experiment(
		env_dynamic, agents_dynamic, steps, runs, is_dynamic=True  # type: ignore[arg-type]
	)

	# ---------------------------------------------------------
	# SAVE THE 5 REQUIRED CHARTS
	# ---------------------------------------------------------
	print("Saving figures...")

	plot_metric(stat_cum, steps, labels_static, colors_static,
	            "Static Environment: Cumulative Reward over Time",
	            "Cumulative Reward", "static_cumulative_reward.png")

	plot_metric(stat_avg, steps, labels_static, colors_static,
	            "Static Environment: Average Reward over Time",
	            "Average Reward", "static_average_reward.png")

	plot_metric(stat_pct, steps, labels_static, colors_static,
	            "Static Environment: Percentage of Best Arm Selection",
	            "% Optimal Selection", "static_pct_best_arm.png", ylim=(0.0, 1.0))

	plot_metric(dyn_cum, steps, labels_dynamic, colors_dynamic,
	            "Dynamic Environment: Cumulative Reward over Time",
	            "Cumulative Reward", "dynamic_cumulative_reward.png")

	plot_metric(dyn_pct, steps, labels_dynamic, colors_dynamic,
	            "Dynamic Environment: Percentage of Best Arm Selection",
	            "% Optimal Selection", "dynamic_pct_best_arm.png", ylim=(0.0, 1.0))


if __name__ == "__main__":
	main()
