import streamlit as st
import requests
import re
from typing import Dict

# API endpoint configurations
API_BASE_URL = "http://localhost:8000"  # Adjust if your FastAPI runs on a different port

# Initialize session states
if "login_details" not in st.session_state:
    st.session_state.login_details = []
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm here to help you find study partners. What's your name?"}]
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "student_data" not in st.session_state:
    st.session_state.student_data = {}

# Enhanced questions sequence with expected formats
QUESTIONS = [
    {
        "question": "What's your name?",
        "key": "name",
        "extraction_prompt": "Extract only the name without any extra words:"
    },
    {
        "question": "What's your contact number?",
        "key": "phone_number",  # Changed to match API model
        "extraction_prompt": "Extract only the numeric contact number:"
    },
    {
        "question": "Which course do you need help with? (Please provide the course code)",
        "key": "course_code",
        "extraction_prompt": "Extract only the course code:"
    },
    {
        "question": "What specific topics do you need help with?",
        "key": "topics",
        "extraction_prompt": "Extract the main topics mentioned:"
    },
    {
        "question": "What's your current knowledge level? (beginner/beginner+/intermediate/intermediate+/advanced/expert)",
        "key": "knowledge_level",
        "extraction_prompt": "Extract only the knowledge level:"
    },
    {
        "question": "What is your desired group size? (2-5)",  # Updated to match API constraints
        "key": "desired_group_size",
        "extraction_prompt": "Extract only the numeric group size:"
    }
]

KNOWLEDGE_LEVELS = ['beginner', 'beginner+', 'intermediate', 'intermediate+', 'advanced', 'expert']

def validate_input(question_idx, value):
    """Enhanced validation for all input types"""
    value = str(value).strip().lower()
    
    if question_idx == 0:  # Name
        return bool(re.match(r'^[A-Za-z\s]{2,}$', value))
    elif question_idx == 1:  # Contact
        return bool(re.match(r'^\d{10}$', value))
    elif question_idx == 2:  # Course code
        return bool(re.match(r'^[A-Za-z]{2,4}\s?\d{3}$', value.upper()))
    elif question_idx == 3:  # Topics
        return len(value) >= 3
    elif question_idx == 4:  # Knowledge level
        return value in KNOWLEDGE_LEVELS
    elif question_idx == 5:  # Group size
        try:
            size = int(value)
            return 2 <= size <= 5  # Updated to match API constraints
        except ValueError:
            return False
    return False

def process_input(question_idx, user_input):
    """Process user input without using Cohere"""
    cleaned_input = user_input.strip()
    
    if question_idx == 4:  # Knowledge level
        level = cleaned_input.lower()
        if level in KNOWLEDGE_LEVELS:
            return level
    
    elif question_idx == 5:  # Group size
        try:
            size = int(re.search(r'\d+', cleaned_input).group())
            if 2 <= size <= 5:  # Updated to match API constraints
                return str(size)
        except (ValueError, AttributeError):
            pass
    
    elif question_idx == 1:  # Contact
        digits = ''.join(filter(str.isdigit, cleaned_input))
        if len(digits) == 10:
            return digits
    
    return cleaned_input

def create_user_and_find_group(user_data: Dict):
    """Create user and find study group through API"""
    try:
        response = requests.post(f"{API_BASE_URL}/users/", json=user_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error creating user: {str(e)}")
        return None

def check_group_status(phone_number: str):
    """Check group status through API"""
    try:
        response = requests.get(f"{API_BASE_URL}/group-status/{phone_number}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error checking status: {str(e)}")
        return None

def display_sidebar():
    """Enhanced sidebar display with API integration"""
    if st.session_state.login_details:
        st.sidebar.title("📚 Student Profile")
        fields = ["Name", "Contact", "Course", "Topics", "Knowledge Level", "Group Size"]
        for field, value in zip(fields, st.session_state.login_details):
            st.sidebar.markdown(f"**{field}:** {value}")
        
        if st.sidebar.button("🔍 Find Study Partners"):
            st.sidebar.markdown("### Looking for study partners...")
            
            # Prepare user data for API
            user_data = {
                "name": st.session_state.student_data["name"],
                "phone_number": st.session_state.student_data["phone_number"],
                "course_code": st.session_state.student_data["course_code"],
                "topics": st.session_state.student_data["topics"],
                "knowledge_level": st.session_state.student_data["knowledge_level"],
                "desired_group_size": int(st.session_state.student_data["desired_group_size"])
            }
            
            # Call API to create user and find group
            result = create_user_and_find_group(user_data)
            
            if result:
                st.sidebar.success("Successfully joined the matching queue!")
                st.sidebar.info("""
                Your request has been added to the queue. 
                Please check the Status tab later using your phone number to see if your group is complete.
                """)
            else:
                st.sidebar.error("Failed to join the matching queue. Please try again.")
        
        st.sidebar.divider()
        st.sidebar.caption("*AI-powered matching in progress")

def display_group_status(status_data):
    """Display group status information"""
    if status_data["is_complete"]:
        st.success("🎉 Your group is complete!")
        if status_data.get("group"):
            group = status_data["group"]
            st.write("### Group Details")
            st.write(f"**Course:** {group['course_code']}")
            st.write(f"**Group Size:** {len(group['members'])}/{group['max_size']}")
            st.write(f"**Created At:** {group['created_at']}")
            if 'completed_at' in group:
                st.write(f"**Completed At:** {group['completed_at']}")
            
            st.write("### Group Members")
            for member in group["members"]:
                st.write(f"- **Name:** {member['name']}")
                st.write(f"  **Phone:** {member['phone_number']}")
                st.write(f"  **Knowledge Level:** {member['knowledge_level']}")
                st.write(f"  **Topics:** {member['topics']}")
            
            if group.get('metrics'):
                st.write("### Group Metrics")
                metrics = group['metrics']
                st.write(f"**Knowledge Balance:** {metrics['knowledge_balance']:.2f}")
                st.write(f"**Topic Similarity:** {metrics['topic_similarity']:.2f}")
                st.write(f"**Overall Compatibility:** {metrics['overall_compatibility']:.2f}")
    else:
        st.info("🕒 Your group is still being formed. Please check back later!")
        if status_data.get("group"):
            group = status_data["group"]
            st.write(f"Current members: {len(group['members'])}/{group['max_size']}")
            st.write("### Current Members")
            for member in group["members"]:
                st.write(f"- **Name:** {member['name']}")
                st.write(f"  **Knowledge Level:** {member['knowledge_level']}")
                
def main():
    st.title("🎓 Smart Study Partner Finder")
    
    tab1, tab2 = st.tabs(["💬 Chat Registration", "📊 Status"])
    
    with tab1:
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="😀" if message["role"]=="user" else "🤖"):
                st.markdown(message["content"])
        
        # Handle user input
        if prompt := st.chat_input("Type your response here..."):
            # Add user message to chat
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Process current question
            if st.session_state.current_question < len(QUESTIONS):
                # Process and validate input
                processed_value = process_input(st.session_state.current_question, prompt)
                
                if validate_input(st.session_state.current_question, processed_value):
                    # Store the processed information
                    current_key = QUESTIONS[st.session_state.current_question]["key"]
                    st.session_state.student_data[current_key] = processed_value
                    
                    # Move to next question
                    st.session_state.current_question += 1
                    
                    # Prepare next question or complete registration
                    if st.session_state.current_question < len(QUESTIONS):
                        next_message = QUESTIONS[st.session_state.current_question]["question"]
                    else:
                        next_message = "Great! Registration complete. Check the sidebar for your profile and click 'Find Study Partners' to join the matching queue!"
                        st.session_state.login_details = list(st.session_state.student_data.values())
                    
                    # Add assistant response
                    st.chat_message("assistant").markdown(next_message)
                    st.session_state.messages.append({"role": "assistant", "content": next_message})
                else:
                    error_message = f"That doesn't look right. Please provide a valid {QUESTIONS[st.session_state.current_question]['key']}."
                    st.chat_message("assistant").markdown(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
            
            st.rerun()
    
    with tab2:
        st.write("### Check Your Study Group Status")
        contact = st.text_input("Enter your contact number")
        if contact:
            if re.match(r'^\d{10}$', contact):
                status_data = check_group_status(contact)
                if status_data:
                    display_group_status(status_data)
            else:
                st.error("Please enter a valid 10-digit phone number.")

    display_sidebar()

if __name__ == "__main__":
    main()