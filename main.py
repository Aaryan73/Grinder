import streamlit as st
import cohere
import re

# Initialize Cohere client
co = cohere.Client('WL7FBHcRECYhJqH5USMsO64KsHyfsObKbUxlS4C2')

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
        "key": "contact",
        "extraction_prompt": "Extract only the numeric contact number:"
    },
    {
        "question": "What year are you currently in? (1-4)",
        "key": "year",
        "extraction_prompt": "Extract only the year number (1-4):"
    },
    {
        "question": "What's your gender? (male/female/other)",
        "key": "gender",
        "extraction_prompt": "Extract only the gender (male/female/other):"
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
        "question": "What is your desired group size? (2-9)",
        "key": "group_size",
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
    elif question_idx == 2:  # Year
        return value in ['1', '2', '3', '4']
    elif question_idx == 3:  # Gender
        return value in ['male', 'female', 'other']
    elif question_idx == 4:  # Course code
        return bool(re.match(r'^[A-Za-z]{2,4}\s?\d{3}$', value.upper()))
    elif question_idx == 5:  # Topics
        return len(value) >= 3
    elif question_idx == 6:  # Knowledge level
        return value in KNOWLEDGE_LEVELS
    elif question_idx == 7:  # Group size
        try:
            size = int(value)
            return 2 <= size <= 9
        except ValueError:
            return False
    return False

def process_input(question_idx, user_input):
    """Enhanced input processing with improved LLM integration"""
    cleaned_input = user_input.strip()
    
    if question_idx == 2:  # Year
        match = re.search(r'\d+', cleaned_input)
        if match:
            year = match.group()
            if year in ['1', '2', '3', '4']:
                return year
    
    elif question_idx == 3:  # Gender
        gender = cleaned_input.lower()
        if gender in ['male', 'female', 'other']:
            return gender
    
    elif question_idx == 6:  # Knowledge level
        level = cleaned_input.lower()
        if level in KNOWLEDGE_LEVELS:
            return level
        # Use LLM to interpret unclear responses
        extraction_prompt = f"Map this response to one of these knowledge levels {KNOWLEDGE_LEVELS}: '{cleaned_input}'"
        response = co.generate(
            model='command-r-plus',
            max_tokens=30,
            temperature=0.1,
            prompt=extraction_prompt
        )
        suggested_level = response.generations[0].text.strip().lower()
        if suggested_level in KNOWLEDGE_LEVELS:
            return suggested_level
    
    elif question_idx in [0, 4, 5]:  # Name, Course code, Topics
        extraction_prompt = f"{QUESTIONS[question_idx]['extraction_prompt']} '{cleaned_input}'"
        response = co.generate(
            model='command-r-plus',
            max_tokens=30,
            temperature=0.1,
            prompt=extraction_prompt
        )
        return response.generations[0].text.strip()
    
    elif question_idx == 1:  # Contact
        digits = ''.join(filter(str.isdigit, cleaned_input))
        if len(digits) == 10:
            return digits
    
    elif question_idx == 7:  # Group size
        try:
            size = int(re.search(r'\d+', cleaned_input).group())
            if 2 <= size <= 9:
                return str(size)
        except (ValueError, AttributeError):
            pass
    
    return cleaned_input

def display_sidebar():
    """Enhanced sidebar display with more information"""
    if st.session_state.login_details:
        st.sidebar.title("📚 Student Profile")
        fields = ["Name", "Contact", "Year", "Gender", "Course", "Topics", "Knowledge Level", "Group Size"]
        for field, value in zip(fields, st.session_state.login_details):
            st.sidebar.markdown(f"**{field}:** {value}")
        
        if st.sidebar.button("🔍 Find Study Partners"):
            st.sidebar.markdown("### Looking for study partners...")
            matching_prompt = f"""
            Find suitable study partners for a student with:
            Course: {st.session_state.student_data.get('course_code')}
            Topics: {st.session_state.student_data.get('topics')}
            Knowledge Level: {st.session_state.student_data.get('knowledge_level')}
            Preferred Group Size: {st.session_state.student_data.get('group_size')}
            """
            response = co.generate(
                model='command-r-plus',
                max_tokens=150,
                temperature=0.7,
                prompt=matching_prompt
            )
            st.sidebar.success("Matching complete!")
            st.sidebar.info(response.generations[0].text.strip())
        
        st.sidebar.divider()
        st.sidebar.caption("*AI-powered matching in progress")

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
                        next_message = "Great! Registration complete. Check the sidebar for your profile and to find study partners!"
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
        st.write("### Study Group Status")
        contact = st.text_input("Enter your contact number to check status")
        if contact:
            if contact == st.session_state.student_data.get("contact"):
                st.info("Your study partner matching is in progress. Check back soon!")
                if st.session_state.student_data:
                    st.write("### Your Profile")
                    for key, value in st.session_state.student_data.items():
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.error("No matching record found.")

    display_sidebar()

if __name__ == "__main__":
    main()