from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import uuid
from langchain_core.embeddings import Embeddings
from langchain_cohere import CohereEmbeddings

class StudyGroupMatcher:
    def __init__(self, cohere_api_key: str):
        self.embeddings = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model="embed-english-v3.0"
        )
        
        # Store active groups
        self.groups = {}  # Dictionary to store groups
        
        # Configuration
        self.MIN_KNOWLEDGE_AVERAGE = 2.0
        self.MIN_TOPIC_SIMILARITY = 0.7
        
        # Knowledge level scores and requirements
        self.KNOWLEDGE_SCORES = {
            'beginner': 1,
            'intermediate': 2,
            'advanced': 3
        }
        
    async def get_topic_embedding(self, topics: str) -> List[float]:
        """Get embedding vector for topics using Cohere"""
        embeddings = await self.embeddings.aembed_query(topics)
        return embeddings
    
    def calculate_embedding_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def get_knowledge_score(self, level: str) -> int:
        """Convert knowledge level to numeric score"""
        return self.KNOWLEDGE_SCORES.get(level.lower(), 1)
    
    def can_maintain_knowledge_balance(self, group: Dict, new_user: Dict) -> bool:
        """
        Check if adding the new user maintains knowledge balance or allows for future balance
        Returns True if the addition is acceptable, False otherwise
        """
        current_members = group['members']
        max_size = group['max_size']
        remaining_slots = max_size - len(current_members) - 1  # -1 for the new user being considered
        
        # Calculate current group scores
        current_scores = [self.get_knowledge_score(m['knowledge_level']) for m in current_members]
        new_user_score = self.get_knowledge_score(new_user['knowledge_level'])
        
        # Calculate new average with the potential member
        new_scores = current_scores + [new_user_score]
        current_avg = sum(new_scores) / len(new_scores)
        
        # If this fills the group, check if it meets minimum average
        if remaining_slots == 0:
            return current_avg >= self.MIN_KNOWLEDGE_AVERAGE
        
        # If slots remain, calculate if it's possible to achieve minimum average
        best_possible_avg = (sum(new_scores) + (remaining_slots * self.KNOWLEDGE_SCORES['advanced'])) / max_size
        worst_possible_avg = (sum(new_scores) + (remaining_slots * self.KNOWLEDGE_SCORES['beginner'])) / max_size
        
        # Check if we can maintain balance with future additions
        if current_avg < self.MIN_KNOWLEDGE_AVERAGE:
            # If current average is below minimum, we must be able to achieve it with advanced members
            return best_possible_avg >= self.MIN_KNOWLEDGE_AVERAGE
        else:
            # If current average is good, we should be able to maintain minimum even with beginners
            return worst_possible_avg >= self.MIN_KNOWLEDGE_AVERAGE
    
    async def find_matching_group(self, user_data: Dict) -> Optional[str]:
        """Find a matching group for the user or return None"""
        user_embedding = await self.get_topic_embedding(user_data['topics'])
        best_group_id = None
        best_similarity = 0
        
        for group_id, group in self.groups.items():
            # Skip groups with different course codes or desired size
            if (group['course_code'] != user_data['course_code'] or 
                group['max_size'] != user_data['desired_group_size'] or
                len(group['members']) >= group['max_size']):
                continue
            
            # Check if adding this user maintains knowledge balance
            if not self.can_maintain_knowledge_balance(group, user_data):
                continue
            
            # Calculate topic similarity with group members
            group_similarities = []
            for member in group['members']:
                if 'topic_embedding' not in member:
                    member['topic_embedding'] = await self.get_topic_embedding(member['topics'])
                similarity = self.calculate_embedding_similarity(user_embedding, member['topic_embedding'])
                group_similarities.append(similarity)
            
            avg_similarity = sum(group_similarities) / len(group_similarities)
            
            if avg_similarity >= self.MIN_TOPIC_SIMILARITY and avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_group_id = group_id
        
        return best_group_id
    
    async def add_user_to_study_group(self, user_data: Dict) -> Dict:
        """Add user to existing group or create new group"""
        # Get topic embedding for the user
        user_data['topic_embedding'] = await self.get_topic_embedding(user_data['topics'])
        
        # Try to find matching group
        matching_group_id = await self.find_matching_group(user_data)
        
        if matching_group_id:
            # Add user to existing group
            self.groups[matching_group_id]['members'].append(user_data)
            group = self.groups[matching_group_id]
            
            # Check if group is now complete
            if len(group['members']) == group['max_size']:
                completed_group = group.copy()
                del self.groups[matching_group_id]  # Remove completed group from active groups
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
            # Create new group
            new_group = {
                'id': str(uuid.uuid4()),
                'course_code': user_data['course_code'],
                'max_size': user_data['desired_group_size'],
                'members': [user_data],
                'created_at': datetime.now().isoformat()
            }
            self.groups[new_group['id']] = new_group
            return {
                'status': 'success',
                'message': 'Created new group',
                'group': new_group
            }

# Example usage
async def main():
    from termcolor import colored
    
    matcher = StudyGroupMatcher(cohere_api_key='WL7FBHcRECYhJqH5USMsO64KsHyfsObKbUxlS4C2')
    
    # Test users with different group sizes and knowledge levels
    test_users = [
        {
            'id': str(uuid.uuid4()),
            'name': 'John',
            'course_code': 'CS101',
            'topics': 'Graph algorithms, breadth first search, depth first search',
            'knowledge_level': 'beginner',
            'desired_group_size': 3
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Alice',
            'course_code': 'CS101',
            'topics': 'Graph Algorithms',
            'knowledge_level': 'beginner',
            'desired_group_size': 3
        },
        {
            'id': str(uuid.uuid4()),
            'name': 'Bob',
            'course_code': 'CS101',
            'topics':   'Python programming, algorithms',
            'knowledge_level': 'intermediate',
            'desired_group_size': 3 
        },
                {
            'id': str(uuid.uuid4()),
            'name': 'Jane',
            'course_code': 'CS101',
            'topics': 'Python, Data Structures and Algorithms',
            'knowledge_level': 'beginner',
            'desired_group_size': 3
        }
    ]
    
    # Process users sequentially
    for user in test_users:
        print(colored(f"\nProcessing user: {user['name']}", "cyan"))
        result = await matcher.add_user_to_study_group(user)
        print(colored(f"Status: {result['status']}", "yellow"))
        print(colored(f"Message: {result['message']}", "yellow"))
        print(colored(f"Group members: {[m['name'] for m in result['group']['members']]}", "green"))
        if len(result['group']['members']) > 1:
            scores = [matcher.get_knowledge_score(m['knowledge_level']) for m in result['group']['members']]
            avg_score = sum(scores) / len(scores)
            print(colored(f"Group knowledge average: {avg_score:.2f}", "blue"))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())