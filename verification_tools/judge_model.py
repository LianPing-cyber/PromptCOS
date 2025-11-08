import os
import openai
import asyncio
import aiohttp
from openai import OpenAI


class JudgeModel:
    def __init__(self, api_key: str, base_url: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
    async def judge(self, answer: str, reference: str) -> bool:
        system_prompt = f"""You are an answer evaluator tasked with determining whether the provided answer is correct.
You will receive: 1. The question and the answer. 2. The standard reference answer.
Please determine whether the answer is correct. Note that if the answer contains multiple responses, it is considered correct if at least one response is accurate.
-- Please only return "Correct" or "Incorrect", without any additional explanations.
"""
        query = f"""The question and the answer:
{answer}
--The standard reference answer：{reference}
\"Correct\" or \"Incorrect\"?"""
        response = await self.fetch_answer(system_prompt, query)
        response = response.strip().lower()
        if "incorrect" in response:
            return 0
        else:
            return 1

    async def fetch_answer(self, system_prompt: str, user_query: str) -> bool:
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            stream=False
        )
        return response.choices[0].message.content