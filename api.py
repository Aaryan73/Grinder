from fastapi import FastAPI, HTTPException
from typing import Dict, Optional, List
import asyncio
from pydantic import BaseModel, Field, validator
import uuid
from datetime import datetime

from matching2_improved import OptimizedStudyGroupMatcher
from typing import Dict, Optional
import asyncio
import uuid
from datetime import datetime

# Data Models
class UserBase(BaseModel):
    name: str
    phone_number: str
    course_code: str
    topics: str
    knowledge_level: str
    desired_group_size: int

    @validator('knowledge_level')
    def validate_knowledge_level(cls, v):
        valid_levels = ['beginner', 'beginner+', 'intermediate', 'intermediate+', 'advanced', 'expert']
        if v.lower() not in valid_levels:
            raise ValueError(f'Knowledge level must be one of {valid_levels}')
        return v.lower()

    @validator('desired_group_size')
    def validate_group_size(cls, v):
        if v < 2 or v > 5:
            raise ValueError('Group size must be between 2 and 5')
        return v

class GroupMember(BaseModel):
    id: str
    name: str
    phone_number: str
    course_code: str
    topics: str
    knowledge_level: str

class GroupMetrics(BaseModel):
    knowledge_balance: float
    topic_similarity: float
    overall_compatibility: float
    details: Optional[Dict] = None

class StudyGroup(BaseModel):
    id: str
    course_code: str
    max_size: int
    members: List[GroupMember]
    created_at: str
    metrics: Optional[GroupMetrics] = None

class GroupResponse(BaseModel):
    status: str
    message: str
    group: Optional[StudyGroup] = None

class GroupStatusResponse(BaseModel):
    status: str
    message: str
    is_complete: bool
    group: Optional[StudyGroup] = None

class CompletedGroup(BaseModel):
    id: str
    course_code: str
    max_size: int
    members: List[GroupMember]
    created_at: str
    completed_at: str
    metrics: Optional[GroupMetrics] = None

# FastAPI App
app = FastAPI(title="Study Group Matcher API")

# Initialize the matcher with your Cohere API key
matcher = OptimizedStudyGroupMatcher(cohere_api_key='WL7FBHcRECYhJqH5USMsO64KsHyfsObKbUxlS4C2')

# In-memory storage for phone number to group ID mapping and completed groups
phone_to_group = {}
completed_groups = {}  # Store completed groups here

@app.post("/users/", response_model=GroupResponse)
async def create_user(user: UserBase):
    """
    Create a new user and attempt to match them with a study group.
    If no suitable group is found, creates a new group.
    """
    try:
        # Prepare user data for the matcher
        user_data = {
            'id': str(uuid.uuid4()),
            'name': user.name,
            'phone_number': user.phone_number,
            'course_code': user.course_code,
            'topics': user.topics,
            'knowledge_level': user.knowledge_level,
            'desired_group_size': user.desired_group_size
        }

        # Try to add user to a group
        result = await matcher.add_user_to_study_group(user_data)
        
        # Store the phone number to group ID mapping
        phone_to_group[user.phone_number] = result['group']['id']
        
        # Check if the group is complete after adding the user
        if len(result['group']['members']) == result['group']['max_size']:
            # Store the completed group
            completed_groups[result['group']['id']] = {
                **result['group'],
                'completed_at': datetime.now().isoformat()
            }
            # Remove from active groups if needed
            if result['group']['id'] in matcher.groups:
                del matcher.groups[result['group']['id']]

        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/group-status/{phone_number}", response_model=GroupStatusResponse)
async def check_group_status(phone_number: str):
    """
    Check the status of a study group using a user's phone number.
    Returns group details if the group is complete.
    """
    try:
        # Check if phone number exists in our mapping
        if phone_number not in phone_to_group:
            return GroupStatusResponse(
                status="error",
                message="No group found for this phone number",
                is_complete=False
            )

        group_id = phone_to_group[phone_number]
        
        # First check if the group is in completed groups
        if group_id in completed_groups:
            return GroupStatusResponse(
                status="success",
                message="Group is complete",
                is_complete=True,
                group=completed_groups[group_id]
            )
        
        # If not in completed groups, check active groups
        if group_id in matcher.groups:
            group = matcher.groups[group_id]
            is_complete = len(group['members']) == group['max_size']
            
            # If group just completed, move it to completed_groups
            if is_complete:
                completed_groups[group_id] = {
                    **group,
                    'completed_at': datetime.now().isoformat()
                }
                del matcher.groups[group_id]
                return GroupStatusResponse(
                    status="success",
                    message="Group is complete",
                    is_complete=True,
                    group=completed_groups[group_id]
                )
            
            return GroupStatusResponse(
                status="success",
                message="Group found",
                is_complete=False,
                group=group
            )
        
        # Group not found in either active or completed groups
        return GroupStatusResponse(
            status="error",
            message="Group not found",
            is_complete=False
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Additional helper endpoints if needed

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}

@app.get("/stats")
async def get_stats():
    """
    Get current statistics about groups and users
    """
    return {
        "total_active_groups": len(matcher.groups),
        "total_users_mapped": len(phone_to_group),
        "courses_with_active_groups": list(matcher.course_groups.keys())
    }