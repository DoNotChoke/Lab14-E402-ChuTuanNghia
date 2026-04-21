import asyncio
import copy
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import Faithfulness, AnswerRelevancy


class RetrievalEvaluator:
    def __init__(
        self,
        top_k: int = 3,
        enable_ragas: bool = True,
        ragas_llm_model: str = "gpt-4o-mini",
        ragas_embedding_model: str = "text-embedding-3-small"
    ):
        self.top_k = top_k
        self.enable_ragas = enable_ragas
        self.ragas_llm_model = ragas_llm_model
        self.ragas_embedding_model = ragas_embedding_model
        self.client = AsyncOpenAI()

    def calculate_hit_rate(
        self,
        expected_ids: List[str],
        retrieved_ids: List[str],
        top_k: Optional[int] = None,
    ) -> float:
        k = top_k or self.top_k

        if not expected_ids:
            return 1.0

        top_retrieved = retrieved_ids[:k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        if not expected_ids:
            return 1.0

        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def score(self, test_case: Dict, response: Dict) -> Dict:
        """
        Return the same shape expected by BenchmarkRunner:
        {
          "faithfulness": float | None,
          "relevancy": float | None,
          "retrieval": {"hit_rate": float, "mrr": float}
        }
        """
        expected_ids = test_case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])

        retrieval_scores = {
            "hit_rate": self.calculate_hit_rate(expected_ids, retrieved_ids),
            "mrr": self.calculate_mrr(expected_ids, retrieved_ids),
            "expected_ids": expected_ids,
            "retrieved_ids": retrieved_ids,
            "is_applicable": bool(expected_ids),
        }

        return {
            "faithfulness": await self.calculate_faithfulness(test_case, response),
            "relevancy": await self.calculate_relevancy(test_case, response),
            "retrieval": retrieval_scores,
        }

    async def evaluate_batch(self, records: List[Dict]) -> Dict:
        """
        Evaluate a list of records containing either:
        - {"test_case": dict, "response": dict}
        - {"case": dict, "response": dict}
        """
        if not records:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}

        scores = []
        for record in records:
            test_case = record.get("test_case") or record.get("case") or record
            response = record.get("response") or record.get("agent_response") or {}
            scores.append(await self.score(test_case, response))

        return {
            "avg_hit_rate": sum(item["retrieval"]["hit_rate"] for item in scores) / len(scores),
            "avg_mrr": sum(item["retrieval"]["mrr"] for item in scores) / len(scores),
            "avg_faithfulness": sum(item["faithfulness"] for item in scores) / len(scores),
            "avg_relevancy": sum(item["relevancy"] for item in scores) / len(scores),
        }

    async def calculate_faithfulness(self, test_case: Dict, response: Dict):
        llm = llm_factory(model=self.ragas_llm_model, client=self.client)
        scorer = Faithfulness(llm=llm)

        result = await scorer.ascore(
            user_input=test_case["question"],
            response=response["answer"],
            retrieved_contexts=response["contexts"]
        )
        return result.value

    async def calculate_relevancy(self, test_case: Dict, response: Dict):
        llm = llm_factory(model=self.ragas_llm_model, client=self.client)
        embeddings = embedding_factory("openai", model=self.ragas_embedding_model, client=self.client)

        scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
        result = await scorer.ascore(
            user_input=test_case["question"],
            response=response["answer"],
        )
        return result.value
