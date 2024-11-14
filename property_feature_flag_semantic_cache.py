# Code explaination - https://docs.google.com/document/d/1d5-M8-MXbhpJ4Tw262pOKw43lf7pRZxuJvnr1uESycg/edit?usp=sharing
  
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import time
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import uuid

class FeatureFlag(Enum):
    """
    Enumeration of possible feature flags for properties.
    """
    HAS_DOORMAN = "has_doorman"
    # PETS_ALLOWED = "pets_allowed"
    # HAS_MICROWAVE = "has_microwave"
    # HAS_GYM = "has_gym"
    # HAS_POOL = "has_pool"

@dataclass
class FeatureConfig:
    """
    Configuration for each feature flag.
    """
    relevant_fields: List[str]
    similarity_threshold: float
    prompt_template: str

class MockLLMClient:
    """
    Mock LLM client to simulate LLM responses for testing purposes.
    """
    def get_boolean_response(self, prompt: str) -> bool:
        """
        Simulate an LLM response based on the prompt content.

        Args:
            prompt (str): The input prompt for the LLM.

        Returns:
            bool: The simulated boolean response.
        """
        prompt_lower = prompt.lower()
        if "doorman" in prompt_lower or "concierge" in prompt_lower:
            return "24/7" in prompt_lower or "full-time" in prompt_lower
        elif "pet" in prompt_lower:
            return "allowed" in prompt_lower and "no pet" not in prompt_lower
        elif "microwave" in prompt_lower:
            return "microwave" in prompt_lower
        return False

class FeatureFlagCache:
    """
    A semantic cache that stores feature flag determinations using embeddings
    and a vector database (Qdrant) to avoid redundant LLM calls.
    """
    def __init__(self, llm_client=None, qdrant_client=None):
        """
        Initialize the FeatureFlagCache with an embedding model, LLM client,
        cache clients, feature configurations, and cache statistics.

        Args:
            llm_client: An instance of an LLM client. Defaults to MockLLMClient.
        """
        # Initialize the embedding model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        # Use provided LLM client or default to MockLLMClient
        self.llm_client = llm_client or MockLLMClient()
        # Initialize feature configurations
        self.feature_configs = self._initialize_feature_configs()
        # Initialize cache statistics
        self.cache_stats = {feature: {"hits": 0, "misses": 0} for feature in FeatureFlag}
        self.qdrant_client = qdrant_client or QdrantClient(url="http://localhost:6333")
        self.create_qdrant_collections_if_not_exist()

    def create_qdrant_collections_if_not_exist(self):
        """
        Create a new Qdrant collection if it does not already exist.
        """
        vector_size = 384  # Dimension of your vectors
        distance = Distance.COSINE  # Similarity measure (COSINE, EUCLID, etc.)
        for feature in FeatureFlag:
            collection_name = feature.value
            if collection_name not in list(map(lambda l: l.name, self.qdrant_client.get_collections().collections)):
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=distance)
                )

    def _initialize_feature_configs(self) -> Dict[FeatureFlag, FeatureConfig]:
        """
        Set up configurations for each feature flag, including relevant fields,
        similarity thresholds, and prompt templates.

        Returns:
            Dict[FeatureFlag, FeatureConfig]: A dictionary of feature configurations.
        """
        return {
            FeatureFlag.HAS_DOORMAN: FeatureConfig(
                relevant_fields=["security_features", "building_features"],
                similarity_threshold=0.85,  # Adjusted threshold for new embedding model
                prompt_template="""
                Based on these property details, determine if there is a doorman:

                security_features: {security_features}
                Building Features: {building_features}

                Return only 'true' or 'false'.
                """
            ),
            # FeatureFlag.PETS_ALLOWED: FeatureConfig(
            #     relevant_fields=["pet_policy", "rules_and_regulations"],
            #     similarity_threshold=0.80,  # Adjusted threshold for new embedding model
            #     prompt_template="""
            #     Based on these property details, determine if pets are allowed:

            #     Pet Policy: {pet_policy}
            #     Rules: {rules_and_regulations}

            #     Return only 'true' or 'false'.
            #     """
            # ),
            # FeatureFlag.HAS_MICROWAVE: FeatureConfig(
            #     relevant_fields=["kitchen_features", "interior_features"],
            #     similarity_threshold=0.80,
            #     prompt_template="""
            #     Based on these property details, determine if there is a microwave:

            #     Kitchen Features: {kitchen_features}
            #     Interior Features: {interior_features}

            #     Return only 'true' or 'false'.
            #     """
            # )
            # Additional feature flags can be added here
        }

    def _prepare_feature_text(self, property_data: Dict[str, str], feature: FeatureFlag) -> str:
        """
        Combine relevant fields from the property data to create a text representation
        for embedding and cache lookup.

        Args:
            property_data (Dict[str, str]): The property details.
            feature (FeatureFlag): The feature flag being queried.

        Returns:
            str: The combined text of relevant fields.
        """
        config = self.feature_configs[feature]
        feature_text_parts = []

        for field in config.relevant_fields:
            value = property_data.get(field, "")
            feature_text_parts.append(f"{field}: {value}")

        return "\n".join(feature_text_parts)

    def get_feature_value(
        self,
        property_data: Dict[str, str],
        feature: FeatureFlag,
    ) -> Tuple[bool, bool, float]:
        """
        Get the value of a feature flag for a given property, utilizing the cache
        if a similar entry exists; otherwise, query the LLM and cache the result.

        Args:
            property_data (Dict[str, str]): The property details.
            feature (FeatureFlag): The feature flag to determine.

        Returns:
            Tuple[bool, bool, float]: A tuple containing:
                - feature_value (bool): The determined feature value.
                - is_cache_hit (bool): Whether the result was from cache.
                - similarity_score (float): The similarity score of the cached entry.
        """
        feature_text = self._prepare_feature_text(property_data, feature)
        embedding = self.encoder.encode(feature_text)

        # Search the cache for a similar entry
        results = self.qdrant_client.search(
            collection_name=feature.value,
            query_vector=embedding,
            limit=1,
            with_payload=True,
            with_vectors=False,
            score_threshold=self.feature_configs[feature].similarity_threshold
        )

        if results:
            # Cache hit
            self.cache_stats[feature]["hits"] += 1
            return (
                results[0].payload["feature_value"],
                True,
                results[0].score
            )

        # Cache miss - query LLM
        self.cache_stats[feature]["misses"] += 1
        config = self.feature_configs[feature]
        prompt = config.prompt_template.format(**property_data)
        feature_value = self.llm_client.get_boolean_response(prompt)

        # Cache the new result
        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, str(time.time()) + property_data["listing_id"])),
            vector=embedding.tolist(),  # Ensure vector is list
            payload={
                "feature_text": feature_text,
                "feature_value": feature_value,
                "timestamp": time.time()
            }
        )

        self.qdrant_client.upsert(
            collection_name=feature.value,
            points=[point]
        )

        return feature_value, False, 1.0

    def get_stats(self) -> Dict:
        """
        Retrieve cache statistics including hits, misses, and hit rate for each feature.

        Returns:
            Dict: A dictionary containing cache statistics per feature flag.
        """
        return {
            feature.value: {
                "hits": self.cache_stats[feature]["hits"],
                "misses": self.cache_stats[feature]["misses"],
                "hit_rate": self.cache_stats[feature]["hits"] /
                            (self.cache_stats[feature]["hits"] + self.cache_stats[feature]["misses"])
                            if (self.cache_stats[feature]["hits"] + self.cache_stats[feature]["misses"]) > 0
                            else 0
            }
            for feature in FeatureFlag
        }
    