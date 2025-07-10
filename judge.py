"""Judge/scoring system for evaluating model outputs."""

import asyncio
import os
from typing import List
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1",
)

class JudgeGroupScore(BaseModel):
    rollout_id: str = Field(description="The id of the rollout being scored.")
    explanation: str = Field(
        description="A short explanation of why you gave this score."
    )
    score: float = Field(description="A score between 0 and 1.")

class JudgeGroupResponse(BaseModel):
    scores: List[JudgeGroupScore]

def strip_json_backticks(s: str) -> str:
    if s.startswith("```json"):
        s = s[len("```json"):].lstrip("\n")
    if s.endswith("```"):
        s = s[: -len("```")].rstrip()
    return s

async def get_judge_completion_msg(
    messages, temperature=0.0, max_tokens=1500, retries=3, timeout=10
) -> str:
    for attempt in range(1, retries + 1):
        try:
            completion = await client.chat.completions.create(
                messages=messages,
                model="o3",
                temperature=1.0,
                # max_tokens=max_tokens,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries:
                print(
                    f"[Retry {attempt}/{retries}] get_judge_completion failed: {e}. Retrying..."
                )
                await asyncio.sleep(3)
            else:
                print(
                    f"[Failure] get_judge_completion failed after {retries} attempts: {e}"
                )
                return "ERROR: Get judge completion failed"