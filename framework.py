"""Professional training framework for reinforcement learning from human feedback."""

import uuid
import asyncio
from typing import Any, Callable, List, Tuple, Optional
from itertools import batched
from dataclasses import dataclass
import art
from art.trajectories import History
from .llm_wrapper import add_thread
from .logging import FileLogger
from .message_utils import convert_langgraph_messages


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int
    batch_size: int
    group_size: int
    validation_samples: int = 2
    
    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.validation_samples < 0:
            raise ValueError("validation_samples must be non-negative")


class TrainingFramework:
    """
    A professional training framework for running reinforcement learning experiments.
    
    This framework orchestrates the training process by managing:
    - Model rollouts and trajectory collection
    - Reward computation and evaluation
    - Batch processing and training steps
    - Logging and monitoring
    """
    
    def __init__(self):
        """Initialize the training framework."""
        pass
    
    async def execute_rollout(
        self, 
        *,
        model: art.Model, 
        scenario: Any, 
        agent_function: Callable
    ) -> Tuple[Any, str]:
        """
        Execute a single rollout for a given model and scenario.
        
        Args:
            model: The model to execute the rollout with
            scenario: The scenario/prompt to run
            agent_function: The agent function that processes the scenario
            
        Returns:
            Tuple of (result, log_path)
        """
        thread_id = str(uuid.uuid4())
        add_thread(
            thread_id, 
            model.inference_base_url, 
            model.inference_api_key, 
            model.inference_model_name
        )
        
        result = await agent_function(scenario, thread_id)
        log_path = f'.art/langgraph/{thread_id}'
        
        return result, log_path
    
    def create_trajectory_from_logs(self, log_path: str, reward: float) -> art.Trajectory:
        """
        Convert logged interactions into a trajectory object.
        
        Args:
            log_path: Path to the log file
            reward: Reward score for this trajectory
            
        Returns:
            art.Trajectory object containing the conversation history
        """
        logs = FileLogger(log_path).load_logs()
        
        trajectory = art.Trajectory(messages_and_choices=[], reward=reward)
        
        for log_entry in logs:
            output = log_entry[1]['output']
            raw_output = output.get('raw') if hasattr(output, 'get') else output
            
            messages = convert_langgraph_messages(log_entry[1]['input'] + [raw_output])
            
            if not trajectory.messages_and_choices:
                trajectory.messages_and_choices = messages
            else:
                trajectory.additional_histories.append(
                    History(messages_and_choices=messages, tools=None)
                )
        
        return trajectory
    
    async def process_scenario(
        self,
        *,
        model: art.Model,
        scenario: Any,
        group_size: int,
        agent_function: Callable,
        reward_function: Callable,
        validation_model: Optional[art.Model] = None,
        validation_samples: int = 2
    ) -> art.TrajectoryGroup:
        """
        Process a single scenario by running multiple rollouts and computing rewards.
        
        Args:
            model: The model being trained
            scenario: The scenario to process
            group_size: Number of rollouts to execute
            agent_function: Function that processes scenarios
            reward_function: Function that computes rewards
            validation_model: Optional validation model for comparison
            validation_samples: Number of validation samples to include
            
        Returns:
            TrajectoryGroup containing all rollouts and their rewards
        """
        # Execute main model rollouts
        rollout_tasks = [
            self.execute_rollout(model=model, scenario=scenario, agent_function=agent_function) 
            for _ in range(group_size)
        ]
        results_and_logs = await asyncio.gather(*rollout_tasks)
        results, log_paths = zip(*results_and_logs)
        
        # Execute validation rollouts if validation model provided
        all_results = list(results)
        if validation_model:
            val_tasks = [
                self.execute_rollout(model=validation_model, scenario=scenario, agent_function=agent_function)
                for _ in range(validation_samples)
            ]
            val_results_and_logs = await asyncio.gather(*val_tasks)
            val_results, _ = zip(*val_results_and_logs)
            all_results.extend(val_results)
        
        # Compute rewards for all results
        rewards = await reward_function(scenario, all_results)
        
        # Create trajectories (only for main model, not validation)
        trajectories = [
            self.create_trajectory_from_logs(log_path, reward)
            for log_path, reward in zip(log_paths, rewards[:len(log_paths)])
        ]
        
        return art.TrajectoryGroup(trajectories=trajectories)
    
    async def execute_training_step(
        self,
        *,
        model: art.Model,
        scenarios: List[Any],
        group_size: int,
        agent_function: Callable,
        reward_function: Callable,
        validation_model: Optional[art.Model] = None,
        validation_samples: int = 2
    ) -> None:
        """
        Execute a single training step across multiple scenarios.
        
        Args:
            model: The model being trained
            scenarios: List of scenarios to process
            group_size: Number of rollouts per scenario
            agent_function: Function that processes scenarios
            reward_function: Function that computes rewards
            validation_model: Optional validation model
            validation_samples: Number of validation samples to include
        """
        scenario_tasks = [
            self.process_scenario(
                model=model,
                scenario=scenario,
                group_size=group_size,
                agent_function=agent_function,
                reward_function=reward_function,
                validation_model=validation_model,
                validation_samples=validation_samples
            )
            for scenario in scenarios
        ]
        
        trajectory_groups = await asyncio.gather(*scenario_tasks)
        await model.train(trajectory_groups)
    
    async def run_training_epoch(
        self,
        *,
        model: art.Model,
        scenarios: List[Any],
        batch_size: int,
        group_size: int,
        agent_function: Callable,
        reward_function: Callable,
        validation_model: Optional[art.Model] = None,
        validation_samples: int = 2
    ) -> None:
        """
        Run a complete training epoch by processing scenarios in batches.
        
        Args:
            model: The model being trained
            scenarios: List of all scenarios
            batch_size: Number of scenarios per batch
            group_size: Number of rollouts per scenario
            agent_function: Function that processes scenarios
            reward_function: Function that computes rewards
            validation_model: Optional validation model
            validation_samples: Number of validation samples to include
        """
        for batch in batched(scenarios, batch_size):
            await self.execute_training_step(
                model=model,
                scenarios=batch,
                group_size=group_size,
                agent_function=agent_function,
                reward_function=reward_function,
                validation_model=validation_model,
                validation_samples=validation_samples
            )
    
    async def run_training(
        self,
        *,
        model: art.Model,
        scenarios: List[Any],
        agent_function: Callable,
        reward_function: Callable,
        config: TrainingConfig,
        validation_model: Optional[art.Model] = None
    ) -> None:
        """
        Run the complete training process for the specified number of epochs.
        
        Args:
            model: The model being trained
            scenarios: List of training scenarios
            agent_function: Function that processes scenarios
            reward_function: Function that computes rewards
            config: Training configuration containing epochs, batch_size, group_size, etc.
            validation_model: Optional validation model
        """
        start_step = await model.get_step()
        
        for epoch in range(config.epochs):
            await self.run_training_epoch(
                model=model,
                scenarios=scenarios,
                batch_size=config.batch_size,
                group_size=config.group_size,
                agent_function=agent_function,
                reward_function=reward_function,
                validation_model=validation_model,
                validation_samples=config.validation_samples
            )


# Backward compatibility functions
async def train(model, epochs, scenarios, batch_size, group_size, do_ai, do_reward, val_model):
    """Legacy training function for backward compatibility."""
    framework = TrainingFramework()
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        group_size=group_size
    )
    await framework.run_training(
        model=model,
        scenarios=scenarios,
        agent_function=do_ai,
        reward_function=do_reward,
        config=config,
        validation_model=val_model
    )