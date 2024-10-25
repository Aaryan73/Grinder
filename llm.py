
import cohere
import re
 
co = cohere.Client('WL7FBHcRECYhJqH5USMsO64KsHyfsObKbUxlS4C2')
 
def get_response(prompt):
    response = co.generate(
        model='command-r-plus',  
        max_tokens=50,
        temperature=0.7,
        prompt=prompt + " Collect only the main relevant data: name, contact, year of study (as a number), gender, course code, sub-topic, and group size."
    )
    return response.generations[0].text.strip()
 
def parse_summary_to_dict(summary): 
    summary_dict = {}
    for line in summary.split('\n'):
        key_value = line.split(': ')
        if len(key_value) == 2:
            key = key_value[0].strip()
            value = key_value[1].strip()
            summary_dict[key] = value
    return summary_dict

def data_collection_chatbot():
    
    student_data = {}

    prompt = "What is your name?"
    student_data["name"] = input(prompt + "\n> ")
    
    prompt = "Enter your contact details?"
    student_data["contact"] = input(prompt + "\n> ")

    prompt = "What year are you currently in?(1/2/3/4)"
    student_data["year"] = input(prompt + "\n> ")
    
    prompt = "What is your Gender?"
    student_data["gender"] = input(prompt + "\n> ")

    prompt = "In which course do you feel you are weak in, Please enter course code."
    student_data["course_code"] = input(prompt + "\n> ")

    prompt = "In which sub-topic do you feel you need more help?"
    student_data["sub_topic"] = input(prompt + "\n> ")

    prompt = "What's your prefered group size?"
    student_data["group-size"] = input(prompt + "\n> ")
    

    summary_prompt = (
        f"Summarize student info: Name - {student_data['name']}, "
        f"Contact - {student_data['contact']}, "
        f"Year of study - {student_data['year']}, "
        f"Gender - {student_data['gender']}, "
        f"Course Code - {student_data['course_code']}, "
        f"Sub-topic - {student_data['sub_topic']}, "
        f"Group size - {student_data['group-size']}."
    )

    summary_response = get_response(summary_prompt)

    student_summary = parse_summary_to_dict(summary_response)

    return student_summary

student_info = data_collection_chatbot()

print(student_info)


