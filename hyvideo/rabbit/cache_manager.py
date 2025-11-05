"""
RabbitVideo Temporal Cache Manager
Caches and reuses computations for similar video frames.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
from collections import OrderedDict
import numpy as np
import time

class TemporalCacheManager:
    """Manages caching of attention computations for temporal redundancy."""

    def __init__(self, similarity_threshold: float = 0.95, max_cache_size: int = 100):
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size

        # Cache storage
        self.attention_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.key_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.value_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.frame_features: Dict[int, torch.Tensor] = {}

        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
        self.memory_saved = 0  # in bytes
        self.compute_saved = 0  # in FLOPs (approximate)

        # Frame similarity tracking
        self.frame_similarities: Dict[Tuple[int, int], float] = {}
        self.similarity_matrix: Optional[np.ndarray] = None

    def _compute_feature_hash(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> str:
        """Compute a hash for the QKV tensors."""
        # Use a combination of tensor statistics as hash
        with torch.no_grad():
            q_stats = (q.mean().item(), q.std().item(), q.shape)
            k_stats = (k.mean().item(), k.std().item(), k.shape)
            v_stats = (v.mean().item(), v.std().item(), v.shape)

        return f"{q_stats}_{k_stats}_{v_stats}"

    def _compute_similarity(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        """Compute cosine similarity between two tensors."""
        with torch.no_grad():
            # Flatten tensors
            t1_flat = tensor1.flatten()
            t2_flat = tensor2.flatten()

            # Compute cosine similarity
            similarity = F.cosine_similarity(t1_flat.unsqueeze(0), t2_flat.unsqueeze(0))

            return similarity.item()

    def _find_similar_cached_entry(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Optional[str]:
        """Find a similar cached entry based on similarity threshold."""
        # Check recent cache entries
        for cache_key in list(self.key_cache.keys())[-10:]:  # Check last 10 entries
            cached_k = self.key_cache[cache_key]
            cached_v = self.value_cache[cache_key]

            # Check if shapes match
            if cached_k.shape != k.shape or cached_v.shape != v.shape:
                continue

            # Compute similarity
            k_sim = self._compute_similarity(k, cached_k)
            v_sim = self._compute_similarity(v, cached_v)

            avg_similarity = (k_sim + v_sim) / 2

            if avg_similarity >= self.similarity_threshold:
                return cache_key

        return None

    def get_cached_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            frame_idx: int) -> Optional[torch.Tensor]:
        """Retrieve cached attention if available for similar inputs."""
        self.total_queries += 1

        # Generate cache key
        cache_key = self._compute_feature_hash(q, k, v)

        # Direct cache hit
        if cache_key in self.attention_cache:
            self.cache_hits += 1
            self._update_statistics(q, k, v, hit=True)
            return self.attention_cache[cache_key]

        # Look for similar cached entry
        similar_key = self._find_similar_cached_entry(q, k, v)
        if similar_key:
            self.cache_hits += 1
            self._update_statistics(q, k, v, hit=True)

            # Use the similar entry's attention
            cached_attention = self.attention_cache[similar_key]

            # Optionally apply small adjustment based on Q difference
            # This maintains quality while reusing computation
            with torch.no_grad():
                # Simple adjustment - could be more sophisticated
                adjustment = q.mean(dim=-1, keepdim=True) * 0.01
                adjusted_attention = cached_attention + adjustment

            return adjusted_attention

        self.cache_misses += 1
        return None

    def cache_attention(self, attention_output: torch.Tensor, q: torch.Tensor,
                       k: torch.Tensor, v: torch.Tensor, frame_idx: int):
        """Cache attention computation for future reuse."""
        cache_key = self._compute_feature_hash(q, k, v)

        # Add to cache
        self.attention_cache[cache_key] = attention_output.detach()
        self.key_cache[cache_key] = k.detach()
        self.value_cache[cache_key] = v.detach()

        # Store frame features for similarity analysis
        if frame_idx not in self.frame_features:
            # Use K as frame feature representation
            self.frame_features[frame_idx] = k.mean(dim=(0, 1)).detach()

        # Enforce cache size limit
        if len(self.attention_cache) > self.max_cache_size:
            # Remove oldest entries (FIFO)
            oldest_key = next(iter(self.attention_cache))
            del self.attention_cache[oldest_key]
            del self.key_cache[oldest_key]
            del self.value_cache[oldest_key]

    def analyze_temporal_redundancy(self, num_frames: int) -> Dict[str, Any]:
        """Analyze temporal redundancy across frames."""
        if len(self.frame_features) < 2:
            return {}

        # Build similarity matrix
        frame_indices = sorted(self.frame_features.keys())[:num_frames]
        n_frames = len(frame_indices)

        similarity_matrix = np.zeros((n_frames, n_frames))

        for i, frame_i in enumerate(frame_indices):
            for j, frame_j in enumerate(frame_indices):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                elif (frame_i, frame_j) in self.frame_similarities:
                    similarity_matrix[i, j] = self.frame_similarities[(frame_i, frame_j)]
                else:
                    # Compute similarity
                    feat_i = self.frame_features[frame_i]
                    feat_j = self.frame_features[frame_j]
                    sim = self._compute_similarity(feat_i, feat_j)
                    similarity_matrix[i, j] = sim
                    self.frame_similarities[(frame_i, frame_j)] = sim

        self.similarity_matrix = similarity_matrix

        # Analyze redundancy
        high_similarity_pairs = np.sum(similarity_matrix > self.similarity_threshold)
        avg_similarity = np.mean(similarity_matrix)
        temporal_smoothness = np.mean(np.abs(np.diff(similarity_matrix, axis=0)))

        # Identify static segments (consecutive similar frames)
        static_segments = []
        in_static = False
        segment_start = 0

        for i in range(1, n_frames):
            if similarity_matrix[i-1, i] > self.similarity_threshold:
                if not in_static:
                    in_static = True
                    segment_start = i-1
            else:
                if in_static:
                    static_segments.append((segment_start, i-1))
                    in_static = False

        if in_static:
            static_segments.append((segment_start, n_frames-1))

        return {
            'avg_similarity': avg_similarity,
            'high_similarity_pairs': high_similarity_pairs,
            'temporal_smoothness': temporal_smoothness,
            'static_segments': static_segments,
            'redundancy_ratio': high_similarity_pairs / (n_frames * n_frames),
            'potential_speedup': 1.0 / (1.0 - min(0.9, avg_similarity))
        }

    def _update_statistics(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, hit: bool):
        """Update cache statistics."""
        if hit:
            # Estimate saved memory (attention computation intermediate results)
            batch_size, heads, seq_len, dim = q.shape
            saved_memory = batch_size * heads * seq_len * seq_len * 4  # float32
            self.memory_saved += saved_memory

            # Estimate saved compute (attention FLOPs)
            # QK^T: 2 * batch * heads * seq * seq * dim
            # Softmax: batch * heads * seq * seq
            # Attention * V: 2 * batch * heads * seq * seq * dim
            saved_flops = 4 * batch_size * heads * seq_len * seq_len * dim
            self.compute_saved += saved_flops

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = self.cache_hits / max(1, self.total_queries)

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_queries': self.total_queries,
            'cache_size': len(self.attention_cache),
            'memory_saved_gb': self.memory_saved / (1024**3),
            'compute_saved_gflops': self.compute_saved / 1e9,
            'avg_similarity': np.mean(list(self.frame_similarities.values())) if self.frame_similarities else 0
        }

    def clear_old_entries(self, keep_recent: int = 20):
        """Clear old cache entries keeping only recent ones."""
        if len(self.attention_cache) <= keep_recent:
            return

        # Keep only recent entries
        keys_to_remove = list(self.attention_cache.keys())[:-keep_recent]
        for key in keys_to_remove:
            del self.attention_cache[key]
            del self.key_cache[key]
            del self.value_cache[key]

    def clear_all(self):
        """Clear all caches."""
        self.attention_cache.clear()
        self.key_cache.clear()
        self.value_cache.clear()
        self.frame_features.clear()
        self.frame_similarities.clear()
        torch.cuda.empty_cache()

    def optimize_cache_policy(self) -> Dict[str, Any]:
        """Suggest optimal cache policy based on observed patterns."""
        stats = self.get_statistics()
        redundancy = self.analyze_temporal_redundancy(len(self.frame_features))

        recommendations = []

        # Adjust similarity threshold based on hit rate
        if stats['hit_rate'] < 0.2:
            recommendations.append(f"Lower similarity threshold from {self.similarity_threshold:.2f} to {self.similarity_threshold - 0.05:.2f}")
        elif stats['hit_rate'] > 0.8:
            recommendations.append(f"Raise similarity threshold from {self.similarity_threshold:.2f} to {self.similarity_threshold + 0.02:.2f}")

        # Adjust cache size
        if len(self.attention_cache) == self.max_cache_size:
            recommendations.append(f"Increase cache size from {self.max_cache_size} to {self.max_cache_size * 2}")

        # Identify optimization opportunities
        if redundancy.get('redundancy_ratio', 0) > 0.5:
            recommendations.append("High temporal redundancy detected - aggressive caching recommended")

        if redundancy.get('static_segments'):
            recommendations.append(f"Found {len(redundancy['static_segments'])} static segments - consider segment-level caching")

        return {
            'current_policy': {
                'similarity_threshold': self.similarity_threshold,
                'max_cache_size': self.max_cache_size
            },
            'statistics': stats,
            'redundancy_analysis': redundancy,
            'recommendations': recommendations
        }

    def print_summary(self):
        """Print cache summary."""
        stats = self.get_statistics()
        policy = self.optimize_cache_policy()

        print("\n" + "="*60)
        print("Temporal Cache Manager Summary")
        print("="*60)
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Cache misses: {stats['cache_misses']}")
        print(f"Hit rate: {stats['hit_rate']:.1%}")
        print(f"Memory saved: {stats['memory_saved_gb']:.2f} GB")
        print(f"Compute saved: {stats['compute_saved_gflops']:.1f} GFLOPs")

        if policy['redundancy_analysis']:
            redundancy = policy['redundancy_analysis']
            print(f"\nTemporal Redundancy:")
            print(f"  Average similarity: {redundancy['avg_similarity']:.3f}")
            print(f"  Redundancy ratio: {redundancy['redundancy_ratio']:.1%}")
            print(f"  Potential speedup: {redundancy['potential_speedup']:.1f}x")

        if policy['recommendations']:
            print(f"\nRecommendations:")
            for rec in policy['recommendations']:
                print(f"  - {rec}")

        print("="*60)