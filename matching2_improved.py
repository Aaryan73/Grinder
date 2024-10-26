from typing import List, Dict, Optional, Tuple, Set
import numpy as np
from datetime import datetime
import uuid
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_cohere import CohereEmbeddings
import torch
from transformers import AutoTokenizer, AutoModel
import logging
from enum import Enum
from functools import lru_cache
import asyncio
from collections import defaultdict

class MatchingModel(Enum):
    SBERT = "sentence-transformers/all-mpnet-base-v2"
    COHERE = "cohere-embed"
    DOMAIN = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

@dataclass
class GroupCompatibility:
    score: float
    knowledge_balance: float
    topic_similarity: float
    group_id: Optional[str] = None
    details: Optional[Dict] = None

class OptimizedStudyGroupMatcher:
    def __init__(self, cohere_api_key: str):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.logger.info("Initializing embedding models...")
        self.models = {}
        self.models['sbert'] = SentenceTransformer(MatchingModel.SBERT.value)
        self.models['domain'] = SentenceTransformer(MatchingModel.DOMAIN.value)
        self.models['cohere'] = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-english-v3.0"
        )
        
        # Optimized data structures
        self.groups = {}
        self.course_groups = defaultdict(set)
        self.size_groups = defaultdict(set)
        
        # Updated configuration with stricter thresholds
        self.MIN_KNOWLEDGE_AVERAGE = 2.0
        self.MIN_TOPIC_SIMILARITY = 0.75  # Increased from 0.75
        self.MIN_INDIVIDUAL_SIMILARITY = 0.70  # New threshold for individual similarity
        self.MAX_SIMILARITY_VARIANCE = 0.15  # New threshold for similarity variance
        self.SIMILARITY_WEIGHTS = {
            'sbert': 0.5,
            'domain': 0.3,
            'cohere': 0.2
        }
        
        # Cache for knowledge scores
        self.knowledge_scores = {
            'beginner': 1.0,
            'beginner+': 1.5,
            'intermediate': 2.0,
            'intermediate+': 2.5,
            'advanced': 3.0,
            'expert': 3.5
        }    
    @lru_cache(maxsize=1000)
    def get_knowledge_score(self, level: str) -> float:
        """Cached knowledge score conversion"""
        return self.knowledge_scores.get(level.lower(), 1.0)
    
    async def get_embeddings(self, text: str) -> Dict[str, np.ndarray]:
        """Get embeddings with batched processing"""
        embeddings = {}
        
        # Process SBERT models in parallel
        sbert_tasks = [
            self.models['sbert'].encode(text, convert_to_tensor=True),
            self.models['domain'].encode(text, convert_to_tensor=True)
        ]
        sbert_results = await asyncio.gather(*[asyncio.to_thread(lambda: task) for task in sbert_tasks])
        
        embeddings['sbert'] = sbert_results[0]
        embeddings['domain'] = sbert_results[1]
        
        # Get Cohere embeddings
        embeddings['cohere'] = np.array(await self.models['cohere'].aembed_query(text))
        
        return embeddings
    
    def calculate_weighted_similarity(self, emb1: Dict[str, np.ndarray], 
                                   emb2: Dict[str, np.ndarray]) -> float:
        """Optimized similarity calculation"""
        similarities = {}
        
        for model_name, weight in self.SIMILARITY_WEIGHTS.items():
            e1 = emb1[model_name].cpu().numpy() if torch.is_tensor(emb1[model_name]) else emb1[model_name]
            e2 = emb2[model_name].cpu().numpy() if torch.is_tensor(emb2[model_name]) else emb2[model_name]
            
            # Optimize reshape operation
            if e1.ndim == 1:
                e1 = e1.reshape(1, -1)
            if e2.ndim == 1:
                e2 = e2.reshape(1, -1)
            
            similarities[model_name] = cosine_similarity(e1, e2)[0][0] * weight
        
        return sum(similarities.values())
    
    def calculate_group_balance(self, members: List[Dict]) -> Tuple[float, bool]:
        scores = np.array([self.get_knowledge_score(m['knowledge_level']) for m in members])
        avg_score = float(np.mean(scores))  # Convert to Python float
        std_dev = float(np.std(scores))     # Convert to Python float
        
        if len(scores) > 1:
            sorted_scores = np.sort(scores)
            max_gap = float(np.max(np.diff(sorted_scores)))  # Convert to Python float
        else:
            max_gap = 0.0
        
        balance_score = float(avg_score * (1 - 0.5 * std_dev) * (1 - 0.3 * max_gap))  # Ensure float
        return balance_score, bool(balance_score >= self.MIN_KNOWLEDGE_AVERAGE)
    
    async def evaluate_group_compatibility(self, user_data: Dict, 
                                    group: Dict) -> GroupCompatibility:
        if 'embeddings' not in user_data:
            user_data['embeddings'] = await self.get_embeddings(user_data['topics'])
        
        similarities = []
        for member in group['members']:
            if 'embeddings' not in member:
                member['embeddings'] = await self.get_embeddings(member['topics'])
            
            similarity = float(self.calculate_weighted_similarity(  # Convert to Python float
                user_data['embeddings'],
                member['embeddings']
            ))
            similarities.append(similarity)
        
        # Convert numpy types to Python types
        avg_similarity = float(np.mean(similarities))
        similarity_std = float(np.std(similarities))
        min_similarity = float(min(similarities))
        
        # Check if any individual similarity is too low
        if min_similarity < self.MIN_INDIVIDUAL_SIMILARITY:
            return GroupCompatibility(
                score=0.0,  # Use Python float
                knowledge_balance=0.0,
                topic_similarity=avg_similarity,
                group_id=group['id'],
                details={
                    'individual_similarities': [float(s) for s in similarities],  # Convert list elements
                    'rejection_reason': 'Individual similarity too low',
                    'min_similarity': min_similarity
                }
            )
        
        # Check if similarity variance is too high
        if similarity_std > self.MAX_SIMILARITY_VARIANCE:
            return GroupCompatibility(
                score=0.0,
                knowledge_balance=0.0,
                topic_similarity=avg_similarity,
                group_id=group['id'],
                details={
                    'individual_similarities': [float(s) for s in similarities],
                    'rejection_reason': 'Too much topic variance in group',
                    'similarity_std': float(similarity_std)
                }
            )
        
        # Calculate knowledge balance
        temp_members = group['members'] + [user_data]
        balance_score, is_balanced = self.calculate_group_balance(temp_members)
        
        # Weighted compatibility score with higher weight on topic similarity
        compatibility_score = float(0.7 * avg_similarity + 0.3 * balance_score)
        
        return GroupCompatibility(
            score=compatibility_score,
            knowledge_balance=float(balance_score),
            topic_similarity=avg_similarity,
            group_id=group['id'],
            details={
                'individual_similarities': [float(s) for s in similarities],
                'knowledge_distribution': [m['knowledge_level'] for m in temp_members],
                'is_balanced': bool(is_balanced),
                'similarity_std': float(similarity_std)
            }
        )

    async def find_matching_group(self, user_data: Dict) -> Optional[GroupCompatibility]:
        best_compatibility = None
        
        self.logger.info(f"Finding match for user with topics: {user_data['topics']}")
        
        candidate_groups = self.course_groups[user_data['course_code']] & \
                        self.size_groups[user_data['desired_group_size']]
        
        self.logger.info(f"Found {len(candidate_groups)} candidate groups")
        
        compatibility_tasks = [
            self.evaluate_group_compatibility(user_data, self.groups[gid])
            for gid in candidate_groups
        ]
        
        if compatibility_tasks:
            compatibilities = await asyncio.gather(*compatibility_tasks)
            
            for compatibility in compatibilities:
                self.logger.info(f"Group {compatibility.group_id} compatibility:")
                self.logger.info(f"  Topic similarity: {compatibility.topic_similarity:.3f}")
                self.logger.info(f"  Knowledge balance: {compatibility.knowledge_balance:.3f}")
                self.logger.info(f"  Overall score: {compatibility.score:.3f}")
                
                if compatibility.score > 0 and \
                compatibility.topic_similarity >= self.MIN_TOPIC_SIMILARITY and \
                (not best_compatibility or compatibility.score > best_compatibility.score):
                    best_compatibility = compatibility
                    self.logger.info("  Selected as best match so far")
        
        return best_compatibility
    
    async def add_user_to_study_group(self, user_data: Dict) -> Dict:
        """Add user to best matching group with optimized indexing"""
        user_data['embeddings'] = await self.get_embeddings(user_data['topics'])
        best_match = await self.find_matching_group(user_data)
        
        if best_match and best_match.score >= self.MIN_TOPIC_SIMILARITY:
            group = self.groups[best_match.group_id]
            group['members'].append(user_data)
            
            group['metrics'] = {
                'knowledge_balance': best_match.knowledge_balance,
                'topic_similarity': best_match.topic_similarity,
                'overall_compatibility': best_match.score,
                'details': best_match.details
            }
            
            if len(group['members']) == group['max_size']:
                completed_group = group.copy()
                # Update indices
                self.course_groups[group['course_code']].remove(group['id'])
                self.size_groups[group['max_size']].remove(group['id'])
                del self.groups[best_match.group_id]
                return {
                    'status': 'success',
                    'message': 'Group completed',
                    'group': completed_group
                }
            
            return {
                'status': 'success',
                'message': 'Added to existing group',
                'group': group
            }
        else:
            new_group = {
                'id': str(uuid.uuid4()),
                'course_code': user_data['course_code'],
                'max_size': user_data['desired_group_size'],
                'members': [user_data],
                'created_at': datetime.now().isoformat(),
                'metrics': {
                    'knowledge_balance': self.get_knowledge_score(user_data['knowledge_level']),
                    'topic_similarity': 1.0,
                    'overall_compatibility': 1.0
                }
            }
            # Update indices
            self.groups[new_group['id']] = new_group
            self.course_groups[new_group['course_code']].add(new_group['id'])
            self.size_groups[new_group['max_size']].add(new_group['id'])
            
            return {
                'status': 'success',
                'message': 'Created new group',
                'group': new_group
            }

async def main():
    from termcolor import colored
    import random
    import time
    from tabulate import tabulate  # For prettier table output
    
    matcher = OptimizedStudyGroupMatcher(cohere_api_key='WL7FBHcRECYhJqH5USMsO64KsHyfsObKbUxlS4C2')
    
    # Generate larger test dataset
    courses = ['CS101', 'CS201', 'CS301']
    topics = [
        'Graph algorithms, BFS, DFS',
        'Dynamic programming, recursion',
        'Data structures, arrays, linked lists',
        'Machine learning basics',
        'Python programming',
        'Database design',
        'Web development',
        'Algorithm analysis'
    ]
    knowledge_levels = ['beginner', 'beginner+', 'intermediate', 'intermediate+', 'advanced', 'expert']
    group_sizes = [3, 4, 5]
    
    test_users = []
    for i in range(50):  # Test with 50 users
        test_users.append({
            'id': str(uuid.uuid4()),
            'name': f'User_{i}',
            'course_code': random.choice(courses),
            'topics': random.choice(topics),
            'knowledge_level': random.choice(knowledge_levels),
            'desired_group_size': random.choice(group_sizes)
        })
    
    
    for i, user in enumerate(test_users):
        print(colored(f"\n{'='*80}", "blue"))
        print(colored(f"Processing user {i+1}/50: {user['name']}", "cyan"))
        print(colored(f"Course: {user['course_code']}, Knowledge: {user['knowledge_level']}", "cyan"))
        print(colored(f"Topics: {user['topics']}", "cyan"))
        
        result = await matcher.add_user_to_study_group(user)
        
        print(colored(f"\nStatus: {result['status']}", "green"))
        print(colored(f"Message: {result['message']}", "green"))
        
        group = result['group']
        print(colored("\nGroup Details:", "yellow"))
        print(colored(f"Group ID: {group['id']}", "white"))
        print(colored(f"Course: {group['course_code']}", "white"))
        print(colored(f"Current size: {len(group['members'])}/{group['max_size']}", "white"))
        
        # Display members table
        member_data = []
        for member in group['members']:
            member_data.append([
                member['name'],
                member['knowledge_level'],
                member['topics']
            ])
        
        print(colored("\nGroup Members:", "yellow"))
        print(tabulate(member_data, 
                      headers=['Name', 'Knowledge Level', 'Topics'],
                      tablefmt='grid'))
        
        if 'metrics' in group:
            metrics = group['metrics']
            print(colored("\nGroup Metrics:", "yellow"))
            print(colored(f"Knowledge Balance: {metrics['knowledge_balance']:.2f}", "white"))
            print(colored(f"Topic Similarity: {metrics['topic_similarity']:.2f}", "white"))
            print(colored(f"Overall Compatibility: {metrics['overall_compatibility']:.2f}", "white"))
            
            if 'details' in metrics:
                if 'rejection_reason' in metrics['details']:
                    print(colored(f"\nRejection Reason: {metrics['details']['rejection_reason']}", "red"))
                
                if metrics['details'].get('individual_similarities'):
                    print(colored("\nSimilarity Scores:", "yellow"))
                    for idx, sim in enumerate(metrics['details']['individual_similarities']):
                        print(colored(f"With {group['members'][idx]['name']}: {sim:.2f}", "white"))
                
                if 'similarity_std' in metrics['details']:
                    print(colored(f"Similarity Standard Deviation: {metrics['details']['similarity_std']:.2f}", "white"))

if __name__ == "__main__":
    asyncio.run(main())