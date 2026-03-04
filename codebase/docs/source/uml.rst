UML Class Diagrams
==================

Arms Hierarchy
--------------

.. uml::

   @startuml

   skinparam classAttributeIconSize 0
   skinparam classFontStyle bold
   skinparam backgroundColor #FAFAFA
   skinparam class {
       BackgroundColor #EEF2FF
       BorderColor #4F5BD5
       ArrowColor #4F5BD5
   }

   abstract class Arm {
       + pull() : float
       + expectation() : float
   }

   class NormalArm {
       + mu : float
       + sigma : float
       + pull() : float
       + expectation() : float
   }

   class UniformArm {
       + lower : float
       + upper : float
       + pull() : float
       + expectation() : float
   }

   class TriangularArm {
       + lower : float
       + mode : float
       + upper : float
       + pull() : float
       + expectation() : float
   }

   class LogNormalArm {
       + mean : float
       + sigma : float
       + pull() : float
       + expectation() : float
   }

   class RayleighArm {
       + scale : float
       + pull() : float
       + expectation() : float
   }

   abstract class DynamicArm {
       + update() : void
       + reset() : void
   }

   class NormalArmDynamic {
       + mu : float
       + sigma : float
       + dmu : float
       + dsigma : float
       + raw_mu : float
       + raw_sigma : float
       + pull() : float
       + update() : void
       + reset() : void
       + expectation() : float
   }

   class UniformArmDynamic {
       + lower : float
       + upper : float
       + dlower : float
       + dupper : float
       + raw_lower : float
       + raw_upper : float
       + pull() : float
       + update() : void
       + reset() : void
       + expectation() : float
   }

   class TriangularArmDynamic {
       + lower : float
       + mode : float
       + upper : float
       + dmode : float
       + raw_mode : float
       + pull() : float
       + update() : void
       + reset() : void
       + expectation() : float
   }

   class LogNormalArmDynamic {
       + mean : float
       + sigma : float
       + dmean : float
       + dsigma : float
       + raw_mean : float
       + raw_sigma : float
       + pull() : float
       + update() : void
       + reset() : void
       + expectation() : float
   }

   class RayleighArmDynamic {
       + scale : float
       + dscale : float
       + raw_scale : float
       + pull() : float
       + update() : void
       + reset() : void
       + expectation() : float
   }

   Arm <|-- NormalArm
   Arm <|-- UniformArm
   Arm <|-- TriangularArm
   Arm <|-- LogNormalArm
   Arm <|-- RayleighArm

   Arm <|-- DynamicArm
   DynamicArm <|-- NormalArmDynamic
   DynamicArm <|-- UniformArmDynamic
   DynamicArm <|-- TriangularArmDynamic
   DynamicArm <|-- LogNormalArmDynamic
   DynamicArm <|-- RayleighArmDynamic

   @enduml


Estimators Hierarchy
--------------------

.. uml::

   @startuml

   skinparam classAttributeIconSize 0
   skinparam classFontStyle bold
   skinparam backgroundColor #FAFAFA
   skinparam class {
       BackgroundColor #E8F5E9
       BorderColor #388E3C
       ArrowColor #388E3C
   }

   abstract class Estimator {
       + N : int
       + estimation : ndarray
       + estimate(arm: int) : float
       + clear() : void
       + update(arm: int, reward: float) : void
   }

   class AverageEstimator {
       + cnt : ndarray
       + sum : ndarray
       + update(arm: int, reward: float) : void
       + clear() : void
   }

   class IncrementalUpdateEstimator {
       + step_size : float
       + update(arm: int, reward: float) : void
   }

   class MovingAverageEstimator {
       + window_size : int
       + queues : list
       + update(arm: int, reward: float) : void
       + clear() : void
   }

   Estimator <|-- AverageEstimator
   Estimator <|-- IncrementalUpdateEstimator
   Estimator <|-- MovingAverageEstimator

   @enduml


Strategies Hierarchy
--------------------

.. uml::

   @startuml

   skinparam classAttributeIconSize 0
   skinparam classFontStyle bold
   skinparam backgroundColor #FAFAFA
   skinparam class {
       BackgroundColor #FFF3E0
       BorderColor #E65100
       ArrowColor #E65100
   }

   abstract class Strategy {
       + select(estimation: ndarray) : int
   }

   class EpsilonGreedyStrategy {
       + epsilon : float
       + select(estimation: ndarray) : int
   }

   class SoftmaxStrategy {
       + temperature : float
       + select(estimation: ndarray) : int
   }

   Strategy <|-- EpsilonGreedyStrategy
   Strategy <|-- SoftmaxStrategy

   @enduml


Agent and Component Interactions
---------------------------------

.. uml::

   @startuml

   skinparam classAttributeIconSize 0
   skinparam classFontStyle bold
   skinparam backgroundColor #FAFAFA
   skinparam class {
       BackgroundColor #F3E5F5
       BorderColor #6A1B9A
       ArrowColor #6A1B9A
   }

   class Agent {
       + strategy : Strategy
       + estimator : Estimator
       + cumulative_reward : float
       + cumulative_reward_over_time : list
       + total_selection : int
       + best_arm_selection : int
       + percentage_best_arm_selection_over_time : list
       + select() : int
       + update(arm: int, reward: float) : void
       + clear() : void
   }

   abstract class Strategy {
       + select(estimation: ndarray) : int
   }

   abstract class Estimator {
       + N : int
       + estimation : ndarray
       + estimate(arm: int) : float
       + clear() : void
       + update(arm: int, reward: float) : void
   }

   abstract class Arm {
       + pull() : float
       + expectation() : float
   }

   Agent "1" *-- "1" Strategy : uses
   Agent "1" *-- "1" Estimator : uses
   Arm "N" -- "1" Agent : pulled by

   @enduml
