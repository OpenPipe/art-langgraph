# A Quick Start Guide

We are going to set up tavily's search enabled chat agent to run a training run using ART.

# First Steps

Clone the repo

`git clone https://github.com/tavily-ai/tavily-chat`

And add this pyproject.toml (ART needs a pyproject to successfuly run)

```
[project]
name = "backend"
version = "0.1.0"
description = "Your project description"
readme = "README.md"
requires-python = ">=3.10"

dependencies = [
  "langchain ==0.3.25",
  "langchain-core ==0.3.65",
  "langchain-openai ==0.3.23",
  "langchain-tavily ==0.2.2",
  "langgraph ==0.4.8",
  "langgraph-checkpoint ==2.0.26",
  "langgraph-prebuilt ==0.2.2",
  "langgraph-sdk ==0.1.70",
  "pydantic ==2.11.7",
  "pydantic_core ==2.33.2",
  "python-dotenv ==1.1.0",
  "tavily-python ==0.7.6",
  "uvicorn ==0.34.3",
  "fastapi ==0.115.12",
  "openpipe-art ==0.3.13",
  "langgraph-training @ git+https://github.com/OpenPipe/art-langgraph.git@ffc3dec52f65cee14b587f4e41729f672e2ec0be",
  "skypilot ==0.8.0",
  "skypilot[aws,cudo,do,fluidstack,gcp,lambda,paperspace,runpod] >=0.8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.metadata]
allow-direct-references = true
```

These are the requirements from the original project with the ART packages added

```
"openpipe-art ==0.3.13",
"langgraph-training @ git+https://github.com/OpenPipe/art-langgraph.git@ffc3dec52f65cee14b587f4e41729f672e2ec0be",
"skypilot ==0.8.0",
"skypilot[aws,cudo,do,fluidstack,gcp,lambda,paperspace,runpod] >=0.8.0",
```

# Getting Your Agent Ready For Training

In order to train your agent, we need to be able to capture its outputs.

In your `backend/agent.py` file add our llm hook import ```from langgraph_training import init_chat_model`

Next we just have to replace all the spots we initialize our model in with our new model function. Luckily this is just one spot in our case.

Change

```
class WebAgent:
    def __init__(
        self,
    ):
        self.llm = ChatOpenAI(
            model="gpt-4.1-nano", api_key=os.getenv("OPENAI_API_KEY")
        ).with_config({"tags": ["streaming"]})
```

to

```
class WebAgent:
    def __init__(
        self,
    ):
        self.llm = init_chat_model()
```

> **_NOTE:_**  If you are already using `init_chat_model` you can just replace your import with the training import and it should just work!

## Training Loop

Below is the full training loop, broken down into parts so you can see how it works.

### 1. Set Up Cluster and Backend

This spins up a training machine on your chosen cloud using SkyPilot.

```py
backend = await SkyPilotBackend.initialize_cluster(
    art_version=".",
    cluster_name=CLUSTER_NAME,
    env_path=".env",
    gpu="H100",
)
```

### 2. Register Your Trainable Model

Create a `TrainableModel` that ART will fine-tune.

```py
model = art.TrainableModel(
    name=AGENT_NAME,
    project=PROJECT_NAME,
    base_model="Qwen/Qwen2.5-14B-Instruct",
)

await model.register(backend)
```

### 3. Define Topics

These are the tasks or prompts the agent will handle. Each topic becomes a scenario for training.
You can pass complex data shapes as well, but for this project we'll just use user questions.

```py
topics = [
    "What are all the iPhone models for sale right now",
    "What country is currently making the most nuclear power plants?",
    "What country has the largest percentage use of public transport?"
]
```

### 4. Define the Agent Function

This function runs your agent on an input topic. It spins up your chat agent, runs the query, and streams the response.

```py
async def do_ai(topic, thread_id):
    agent = WebAgent()
    compiled_agent = agent.build_graph()

    try:
        inputs = {"messages": [HumanMessage(content=topic)]}
        async for s in compiled_agent.astream(inputs, config={"thread_id": thread_id}, stream_mode="values"):
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
            report = message
    except Exception as e:
        print(e)
        report = "An error occurred."
    
    return report
```

### 5. Define the Reward Function

This function scores the agent's outputs. It shows multiple runs to a judge model that returns scores.

```py
async def do_reward(topic, reports, trajectories):
    serialized_rollouts: List[str] = []

    for idx, traj in enumerate(trajectories, start=1):
        serialized_rollouts.append(
            f'<rollout id="{idx}">\n' + json.dumps(traj.messages()) + "\n</rollout>"
        )
        
    rubric_text = dedent(
        f"""
        All of the rollouts below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgement of the agent's goal.
        Please respond in a json format that matches this shcema.
        {json.dumps(JudgeGroupResponse.model_json_schema(), indent=2)}

        Grading standards:
            - A rollout that achieves its goal should always get a significantly higher score than a rollout that does not achieve its goal.
            - A rollout that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a rollout that achieves its goal less efficiently.
            - If one rollout is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
            - You may give some partial credit for a rollout that makes progress towards its goal but does not complete it.
        """
    )

    user_text = "Rollouts:\n\n" + "\n\n".join(serialized_rollouts)

    messages = [
        {"role": "system", "content": rubric_text},
        {"role": "user", "content": user_text},
    ]

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
    )

    result = completion = await client.chat.completions.create(
        messages=messages,
        model="o3",
        temperature=1.0,
    )
    result = completion.choices[0].message.content.strip()

    parsed = JudgeGroupResponse.model_validate_json(result)

    print(user_text)
    print(result)

    return [s.score for s in parsed.scores]
```

### 6. Create the Training Config

This sets your training parameters.

```py
config = TrainingConfig(
    epochs=4,
    batch_size=1,
    group_size=8,
    validation_samples=2
)
```

### 7. Add a Validation Model (Optional)

Use a separate model to evaluate performance during training. This will cause the agent to run with model as well as the one you're training to help you see if your model is improving compared to some baseline.

```py
gpt41 = art.Model(
    name="gpt-4.1",
    project=PROJECT_NAME,
    inference_model_name="gpt-4.1",
    inference_api_key=os.getenv("OPENAI_API_KEY"),
    inference_base_url="https://api.openai.com/v1",
)
```

### 8. Run the Training

This puts it all together and starts the training loop.

```py
framework = TrainingFramework()

await framework.run_training(
    model=model,
    scenarios=topics,
    agent_function=do_ai,
    reward_function=do_reward,
    config=config,
    validation_model=gpt41,
)
```

# Full Training Script

Below is the complete example training script based on the breakdown above. Save this as `train.py` or similar.

```python
from backend.agent import WebAgent
from dotenv import load_dotenv
from langgraph_training import JudgeGroupResponse, get_judge_completion_msg, TrainingFramework, TrainingConfig
from textwrap import dedent
from art.skypilot import SkyPilotBackend
from langchain.schema import HumanMessage

import art
import json
import os
import asyncio

load_dotenv()

AGENT_NAME = "tavily-agent-001"
PROJECT_NAME = "tavily-chat-agent"
CLUSTER_NAME = "tavily-chat-training-machine"

async def main():
    backend = await SkyPilotBackend.initialize_cluster(
        art_version=".",
        cluster_name=CLUSTER_NAME,
        env_path=".env",
        gpu="H100",
    )

    model = art.TrainableModel(
        name=AGENT_NAME,
        project=PROJECT_NAME,
        base_model="Qwen/Qwen2.5-14B-Instruct",
    )

    await model.register(backend)

    topics = [
        "What are all the iPhone models for sale right now",
        "What country is currently making the most nuclear power plants?",
        "What country has the largest percentage use of public transport?"
    ]

    async def do_ai(topic, thread_id):
        agent = WebAgent()
        compiled_agent = agent.build_graph()

        try:
            inputs = {"messages": [HumanMessage(content=topic)]}
            async for s in compiled_agent.astream(inputs, config={"thread_id": thread_id}, stream_mode="values"):
                message = s["messages"][-1]
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()
                report = message
        except Exception as e:
            print(e)
            report = "An error occurred."
        
        return report

    async def do_reward(topic, reports, trajectories):
        serialized_rollouts: List[str] = []

        for idx, traj in enumerate(trajectories, start=1):
            serialized_rollouts.append(
                f'<rollout id="{idx}">\n' + json.dumps(traj.messages()) + "\n</rollout>"
            )
            
        rubric_text = dedent(
            f"""
            All of the rollouts below have been given the same goal. Your job is to consider each of them and give them a score between 0 and 1. Take into consideration your best judgement of the agent's goal.
            Please respond in a json format that matches this shcema.
            {json.dumps(JudgeGroupResponse.model_json_schema(), indent=2)}

            Grading standards:
                - A rollout that achieves its goal should always get a significantly higher score than a rollout that does not achieve its goal.
                - A rollout that achieves its goal more efficiently (eg. by avoiding unproductive detours) should get a higher score than a rollout that achieves its goal less efficiently.
                - If one rollout is only slightly better than another, the difference in scores should be small. If it is significantly better, the difference in scores should be large.
                - You may give some partial credit for a rollout that makes progress towards its goal but does not complete it.
            """
        )

        user_text = "Rollouts:\n\n" + "\n\n".join(serialized_rollouts)

        messages = [
            {"role": "system", "content": rubric_text},
            {"role": "user", "content": user_text},
        ]

        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
        )

        result = completion = await client.chat.completions.create(
            messages=messages,
            model="o3",
            temperature=1.0,
        )
        result = completion.choices[0].message.content.strip()

        parsed = JudgeGroupResponse.model_validate_json(result)

        print(user_text)
        print(result)

        return [s.score for s in parsed.scores]

    config = TrainingConfig(
        epochs=4,
        batch_size=1,
        group_size=8,
        validation_samples=2
    )

    gpt41 = art.Model(
        name="gpt-4.1",
        project=PROJECT_NAME,
        inference_model_name="gpt-4.1",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
        inference_base_url="https://api.openai.com/v1",
    )

    framework = TrainingFramework()

    await framework.run_training(
        model=model,
        scenarios=topics,
        agent_function=do_ai,
        reward_function=do_reward,
        config=config,
        validation_model=gpt41,
    )

if __name__ == "__main__":
    asyncio.run(main())
```
