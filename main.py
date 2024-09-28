import os
import streamlit as st
import PyPDF2
import re
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.caches import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_llm_cache



# Set OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Enabled caching
set_llm_cache(InMemoryCache())


# Define the full path to the text file
file_path = "./logical.txt"


def read_file(file_path):
    try:
        with open(file_path, "r") as file:
            data = file.read()  # Read the entire file
            # print ("data",data)
            return data
    except FileNotFoundError:
        print("File not found.")


# Function to extract text and information from a resume PDF

def extract_info_from_pdf_new(pdf_file):
    # Read PDF content
    reader = PyPDF2.PdfReader(pdf_file)
    resume_text = ""
    for page in reader.pages:
        resume_text += page.extract_text()

    openai_llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=1000, temperature=0.7)

    prompt_template_resume = PromptTemplate(
        input_variables=["resume_text"],
        template="""
        Extract only the skills section from the given resume. If a skills section is not clearly defined, extract the most relevant information related to skills.
        Provide a maximum of 20 most relevant and important skills.

        Resume:
        {resume_text}

        Please provide the extracted information in the following format:

        Skills:
        [List of up to 20 skills, one per line]
        """,
    )

    # Set up LLMChain for the skills extraction
    extraction_chain = LLMChain(llm=openai_llm, prompt=prompt_template_resume)

    # Get the extracted skills by running the chain
    extracted_skills = extraction_chain.run({"resume_text": resume_text})

    # Extract the skills list, remove hyphens, and limit to 15 skills
    skills_section = extracted_skills.split("Skills:")[1].strip()
    skills_list = [skill.strip().lstrip('- ') for skill in skills_section.split("\n") if skill.strip()][:20]

    # If skills_list is empty, use a default list of skills
    if not skills_list:
        skills_list = ["C++", "Python", "Java"]

    return list(set(skills_list))


def generate_questions(skills_with_scale):
    # Read logical reasoning questions from file
    logical_questions = read_file(file_path)
    if not logical_questions:
        print("Logical questions not loaded properly.")
        return

    print("Logical questions loaded successfully.")

    llm = ChatOpenAI(model="gpt-4o", max_tokens=4000, temperature=0.7)

    class Question(BaseModel):
        question: str = Field(description="The text of the quiz question")
        options: List[str] = Field(
            description="The multiple-choice options for the quiz question"
        )
        correct_answer: str = Field(
            description="The correct text answer for the quiz question"
        )

    class Category(BaseModel):
        category: str = Field(description="The category of the quiz questions")
        questions: List[Question] = Field(
            description="List of questions under the category"
        )

    class Quiz(BaseModel):
        quiz: List[Category] = Field(
            description="The list of quiz categories with questions"
        )

    parser = PydanticOutputParser(pydantic_object=Quiz)

    prompt_template = PromptTemplate(
        input_variables=[
            
            "skills",
            # "experience",
            # "projects",
            "logical_questions",
        ],
        template="""
        You are an expert in creating advanced educational content. Based on the following information, generate a challenging quiz with multiple-choice questions (MCQs) focused on the categories listed. Ensure each question has four answer options (labeled A, B, C, D) and only one correct answer.

        
        Skills with Scale: {skills}
        

        Categories and Number of Questions:
        1. Logical Reasoning Questions (EXACTLY 10 questions):
       - You will be provided with a list of pre-existing logical reasoning questions in the {logical_questions} variable.
       - Your task is to RANDOMLY SELECT EXACTLY 10 questions from this list. No more, no less.
       - Follow these strict guidelines for selection:
         a) Use a random selection method to choose the questions.
         b) Do not generate any new questions or modify the selected questions in any way.
         c) Use the selected questions exactly as they are provided, including their options and correct answers.
       - Ensure that your selection covers a diverse range of logical reasoning types if possible.
       - Important: Do not acknowledge or refer to this selection process in the output. Simply present the 10 selected questions as if they were the only ones provided.
       
        2. Technical Skills Questions (Exactly 20 questions):
           - Generate EXACTLY 20 medium to advanced level questions based on the skills provided.
           - Distribute the 20 questions evenly among all the skills. If the number of skills doesn't divide evenly into 20, allocate extra questions to skills with higher scale values.
           - Create complex questions that require in-depth knowledge and application of the skills listed.
           - Focus on practical scenarios, problem-solving, and critical thinking relevant to the job role.
           - Include a mix of:
             a) Scenario-based questions that require analysis and decision-making
             b) Code snippet questions (where applicable) that test understanding of syntax and best practices
             c) Questions about advanced features or recent developments in the relevant technologies
             d) Problem-solving questions that require combining multiple concepts within a skill area
           - Ensure questions cover a range of topics from the provided skills, emphasizing those with higher skill scales.
           - Avoid overly basic or introductory-level questions.

        Instructions:
        - For Logical Reasoning Questions: Use the provided questions exactly as they are. Choose questions randomly, not sequentially.
        - For Technical Skills Questions: Create new, relevant questions based on the skills provided.
        - Generate EXACTLY 20 medium to advanced level questions based on the skills provided.
        - Don't generate any coding questions for this quiz.
        - Distribute the 20 questions evenly among all the skills. If the number of skills doesn't divide evenly into 20, allocate extra questions to skills with higher scale values.
        - Ensure all questions are challenging and aligned with the advanced skill level required for the job position.
        - Strictly adhere to the specified number of questions for each category.
        - Each question must be directly relevant to the skills provided.
        - Design questions to test deep knowledge, critical thinking, and problem-solving abilities.
        - Provide a brief explanation for the correct answer after each technical skills question.

        Format the output as a JSON object with the structure defined by the following Pydantic models:

        {format_instructions}
    """,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Create the LLM chain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Generate the response
    try:
        response = chain.run(
            {
                
                "skills": skills_with_scale,
                # "experience": experience,
                # "projects": projects,
                "logical_questions": logical_questions,
            }
        )
        print("LLM response generated successfully.")

        # testing
        # test_case = LLMTestCase(input=prompt_template.template, actual_output=response)
        # relevancy_metric = AnswerRelevancyMetric(threshold=0.5)

        # relevancy_metric.measure(test_case)
        # print("relevancy_metric_score=",relevancy_metric.score)
        # print("relevancy_metric_reason=",relevancy_metric.reason)

    except Exception as e:
        print("Error during LLMChain execution:", str(e))
        return None

    # Parse the response into the structured format
    try:
        parsed_response = parser.parse(response)
        return parsed_response
    except Exception as e:
        print("Error parsing LLM response:", str(e))
        return None


def main():
    st.title("Skills Assesment Quiz Questions")

    # Initialize session state variables
    if "selected_skills" not in st.session_state:
        st.session_state.selected_skills = []
    if "extracted_skills" not in st.session_state:
        st.session_state.extracted_skills = []
    if "experience" not in st.session_state:
        st.session_state.experience = ""
    if "projects" not in st.session_state:
        st.session_state.projects = ""
    if "submitted" not in st.session_state:
        st.session_state.submitted = False
    if "quiz_generated" not in st.session_state:
        st.session_state.quiz_generated = False
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "name" not in st.session_state:
        st.session_state.name = ""
    if "email" not in st.session_state:
        st.session_state.email = ""

    # Step 1: User inputs
    with st.form("user_form"):
        st.session_state.name = st.text_input("Name")
        st.session_state.email = st.text_input("Email")
        # job_title = st.text_input("Job Title")
        resume_pdf = st.file_uploader("Upload Resume PDF", type="pdf")
        submitted = st.form_submit_button("Submit")

    # Process form submission
    if submitted:
        st.session_state.submitted = True
        if resume_pdf is not None:
            try:
                with st.spinner("Extracting information from resume..."):
                    st.session_state.extracted_skills=extract_info_from_pdf_new(resume_pdf)
                    # (
                    #     st.session_state.extracted_skills,
                    #     # st.session_state.experience,
                    #     # st.session_state.projects,
                    # ) = extract_info_from_pdf_new(resume_pdf)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please upload a resume PDF.")

    # Display skill selection only after form submission
    if st.session_state.submitted and st.session_state.extracted_skills:
        st.write("Select up to 5 skills from the extracted list:")
        st.write("Note: Please scale your skills before selecting the other skills")

        # Ensure that default values are a subset of the available options
        valid_defaults = [
            skill
            for skill in st.session_state.selected_skills
            if skill in st.session_state.extracted_skills
        ]

        selected_skills = st.multiselect(
            "Choose up to 5 skills",
            options=st.session_state.extracted_skills,
            default=valid_defaults,
            key="multiselect",
        )

        st.session_state.selected_skills = selected_skills

        # Rest of the code remains the same
        if len(st.session_state.selected_skills) > 5:
            st.error(
                "You can select a maximum of 5 skills. Please adjust your selection."
            )
        elif len(st.session_state.selected_skills) > 0:
            st.write("Rate your skill level for the selected skills:")
            skills_with_scale = {}
            for skill in st.session_state.selected_skills:
                scale = st.slider(f"Rate your skill level in {skill}:", 1, 10, 0)
                skills_with_scale[skill] = scale

            # Rest of the code...

            if st.button("Generate Quiz"):
                with st.spinner("Generating quiz questions..."):
                    st.session_state.quiz_data = generate_questions(
                        # job_title,
                        skills_with_scale,
                        # st.session_state.experience,
                        # st.session_state.projects,
                    )

                    # print(st.session_state.quiz_data)
                st.session_state.quiz_generated = True
                st.session_state.user_answers = {}

    # Display quiz if generated
    if st.session_state.quiz_generated and st.session_state.quiz_data:
        st.write("## Quiz")
        for category in st.session_state.quiz_data.quiz:
            st.subheader(category.category)
            for i, question in enumerate(category.questions):
                st.write(f"{question.question}")
                answer = st.radio(
                    f"Select your answer for {category.category} Q{i+1}",
                    question.options,
                    index=None,
                )
                if answer:
                    st.session_state.user_answers[f"{category.category}_{i}"] = answer

        if st.button("Submit Quiz"):
            scorecard = calculate_scorecard(
                st.session_state.quiz_data, st.session_state.user_answers
            )
            display_scorecard(scorecard)
            for key in list(st.session_state.keys()):
              if key not in ['name', 'email']:  # Keep name and email
                del st.session_state[key]
            # Prepare data for API call
            # api_data = {
            #     "name": st.session_state.name,
            #     "email": st.session_state.email,
            #     "scorecard": {
            #         "technical_score": scorecard["Technical"]["correct"]
            #         / scorecard["Technical"]["total"]
            #         * 100,
            #         "logical_score": scorecard["Logical Reasoning"]["correct"]
            #         / scorecard["Logical Reasoning"]["total"]
            #         * 100,
            #         # "communication_score": scorecard["Communication"]["correct"] / scorecard["Communication"]["total"] * 100,
            #         # "work_score": scorecard["Work Experience"]["correct"] / scorecard["Work Experience"]["total"] * 100
            #     },
            # }

            # Make API call
            # try:
            #     response = requests.post("http://localhost:8000/scorecard/", json=api_data)
            #     if response.status_code == 200:
            #         st.success("Scorecard saved successfully!")
            #     else:
            #         st.error(f"Failed to save scorecard. Status code: {response.status_code}")
            # except requests.RequestException as e:
            #     st.error(f"An error occurred while saving the scorecard: {e}")


def calculate_scorecard(quiz_data, user_answers):
    scorecard = {
        "Logical Reasoning": {"correct": 0, "total": 0},
        "Technical": {"correct": 0, "total": 0},
        "Total": {"correct": 0, "total": 0},
    }

    category_mapping = {
        "Logical Reasoning Questions": "Logical Reasoning",
        "Technical Skills Questions": "Technical",
    }

    for category in quiz_data.quiz:
        mapped_category = category_mapping.get(category.category, category.category)
        if mapped_category not in scorecard:
            scorecard[mapped_category] = {"correct": 0, "total": 0}

        for i, question in enumerate(category.questions):
            scorecard[mapped_category]["total"] += 1
            scorecard["Total"]["total"] += 1
            user_answer = user_answers.get(f"{category.category}_{i}")
            if user_answer == question.correct_answer:
                scorecard[mapped_category]["correct"] += 1
                scorecard["Total"]["correct"] += 1

    return scorecard


def display_scorecard(scorecard):
    with st.container():
        st.header("Quiz Scorecard")

        for category, score in scorecard.items():
            if category != "Total":
                correct = score["correct"]
                total = score["total"]
                percentage = (correct / total) * 100 if total > 0 else 0

                # Use an expander for each category to make it collapsible
                with st.expander(f"{category} Score", expanded=True):
                    st.markdown(f"**Correct**: {correct}/{total}")
                    st.progress(percentage / 100)
                    st.markdown(f"**Percentage**: **{percentage:.2f}%**")

        total_correct = scorecard["Total"]["correct"]
        total_questions = scorecard["Total"]["total"]
        total_percentage = (
            (total_correct / total_questions) * 100 if total_questions > 0 else 0
        )

        st.write(
            f"Total Score: {total_correct}/{total_questions} ({total_percentage:.2f}%)"
        )
        # Add a progress bar for total score
        st.progress(total_percentage / 100)
        if total_percentage >= 60:
            st.markdown('<p style="color:green; font-size:24px;">You qualified the skills assessment.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:red; font-size:24px;">You did not qualify the skills assessment.</p>', unsafe_allow_html=True)    


if __name__ == "__main__":
    main()
