# LangGraph Training

A training framework for LangGraph agents using reinforcement learning.

## Installation

```bash
pip install langgraph-training
```

## Quick Start

```python
import asyncio
from langgraph_training import TrainingFramework, TrainingConfig

async def my_agent(scenario, thread_id):
    # Your LangGraph agent logic here
    return "Generated response"

async def my_reward_function(scenario, results):
    # Your reward computation logic here
    return [0.8, 0.9, 0.7]  # Scores for each result

async def main():
    # Configure training parameters
    config = TrainingConfig(
        epochs=5,
        batch_size=2,
        group_size=4,
        validation_samples=2
    )
    
    # Initialize framework
    framework = TrainingFramework()
    
    # Run training
    await framework.run_training(
        model=your_model,
        scenarios=your_scenarios,
        agent_function=my_agent,
        reward_function=my_reward_function,
        config=config,
        validation_model=your_validation_model  # optional
    )

if __name__ == "__main__":
    asyncio.run(main())
```

## Requirements

- Python ≥ 3.10
- LangChain Core ≥ 0.3.0
- OpenAI ≥ 1.0.0
- Pydantic ≥ 2.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.