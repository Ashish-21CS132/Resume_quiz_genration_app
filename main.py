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
import langchain
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_llm_cache


# Set OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

#Enabled caching
set_llm_cache(InMemoryCache())

# Function to extract text and information from a resume PDF
def extract_info_from_pdf_new(pdf_file):
    # Read PDF content
    reader = PyPDF2.PdfReader(pdf_file)
    resume_text = ""
    for page in reader.pages:
        resume_text += page.extract_text()

    openai_llm = ChatOpenAI(model="gpt-4o-mini",max_tokens=1000, temperature=0.7)

    prompt_template_resume = PromptTemplate(
        input_variables=["resume_text"],
        template="""
        Extract the following sections from the given resume: skills, experience, and projects. If any section is not clearly defined, extract the most relevant information related to that section.

        Resume:
        {resume_text}

        Please provide the extracted information in the following format:

        Skills:
        [Extracted skills information]

        Experience:
        [Extracted experience information]

        Projects:
        [Extracted projects information]
        """,
    )

    # Set up LLMChain for the combined extraction
    extraction_chain = LLMChain(llm=openai_llm, prompt=prompt_template_resume)

    # Get the extracted information by running the chain
    extracted_info = extraction_chain.run({"resume_text": resume_text})

    # Split the extracted information into sections
    sections = re.split(r"\n\s*\n", extracted_info)

    skills_section = next(
        (s for s in sections if s.lower().strip().startswith("skills:")), ""
    )
    experience_section = next(
        (s for s in sections if s.lower().strip().startswith("experience:")), ""
    )
    projects_section = next(
        (s for s in sections if s.lower().strip().startswith("projects:")), ""
    )

    # Extract specific skills
    skills_list = re.findall(
        r"\b(Java|Python|C\+\+|JavaScript|SQL|HTML|CSS|Django|React|NodeJS|ExpressJS|Docker|Langchain|MongoDB|Machine Learning)\b",
        skills_section,
        re.IGNORECASE,
    )

    # If skills_list is empty, use all words in the skills section as skills
    if not skills_list:
        # skills_list = re.findall(r"\b\w+\b", skills_section.replace("Skills:", ""))
        skills_list=["C++","Python", "Java"]

    return (
        list(set(skills_list)),
        experience_section.replace("Experience:", "").strip(),
        projects_section.replace("Projects:", "").strip(),
    )
    


def generate_questions(position, skills_with_scale, experience, projects):
    llm = ChatOpenAI(model="chatgpt-4o-latest",max_tokens=4000, temperature=0.5)
    
    class Question(BaseModel):
        question: str = Field(description="The text of the quiz question")
        options: List[str] = Field(description="The multiple-choice options for the quiz question")
        correct_answer: str = Field(description="The correct text answer for the quiz question")

    class Category(BaseModel):
        category: str = Field(description="The category of the quiz questions")
        questions: List[Question] = Field(description="List of questions under the category")

    class Quiz(BaseModel):
        quiz: List[Category] = Field(description="The list of quiz categories with questions")
        
    parser = PydanticOutputParser(pydantic_object=Quiz)
    
    prompt_template = PromptTemplate(
    input_variables=["position", "skills", "experience", "projects"],
    template="""
        You are an expert in creating advanced educational content. Based on the following information, generate a challenging quiz with multiple-choice questions (MCQs) focused on the categories listed. Ensure each question has four answer options (labeled A, B, C, D) and only one correct answer.

        Job Position: {position}
        Skills with Scale: {skills}
        Work Experience: {experience}
        Projects: {projects}

        Categories and Number of Questions:
        - Technical Questions (8 questions): Create questions that are complex and require in-depth knowledge and understanding of the skills listed. Focus on advanced technical scenarios, problem-solving, and critical thinking skills relevant to the job role.
        - Logical Reasoning Questions (8 questions): Develop questions that are mathematically challenging and involve multiple steps or advanced reasoning. These should test high-level analytical and problem-solving abilities and should be compulsory questions.
        - Communication Questions (7 questions): Based on real placement interviews, focusing on assessing effective communication skills in various job-related scenarios.
        - Work Experience Questions (7 questions): Based on working in a company on real-time projects, focusing on practical challenges and decision-making in a professional environment. Avoid irrelevant questions.Please ask questions from the all sections of workexperience sections.

        Instructions:
        - Clearly indicate the correct answer for each question.
        - Ensure all questions are challenging and align with the advanced skill level required for the job position.
        - Strictly adhere to the number of questions for each category as specified above.
        - Each question must be relevant to the Job Position and Skills provided.
        - Each question should test the deep knowledge and abilities necessary for the job role.
        - Ensure that the Logical Reasoning Questions are based on advanced mathematics and are challenging yet appropriate for the job role.
        - Format the output as a JSON object with the structure defined by the following Pydantic models:

        {format_instructions}

        Generate the quiz data according to these specifications, ensuring a high level of difficulty for the Technical and Logical Reasoning sections.
    """,
    partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    
    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run(
        {
            "position": position,
            "skills": skills_with_scale,
            "experience": experience,
            "projects": projects,
        }
    )
    
    return parser.parse(response)

def main():
    st.title("Automated Interview Quiz Generator")

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

    # Step 1: User inputs
    with st.form("user_form"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        job_title = st.text_input("Job Title")
        resume_pdf = st.file_uploader("Upload Resume PDF", type="pdf")
        submitted = st.form_submit_button("Submit")

    # Process form submission
    if submitted:
        st.session_state.submitted = True
        if resume_pdf is not None:
            try:
                with st.spinner("Extracting information from resume..."):
                    (
                        st.session_state.extracted_skills,
                        st.session_state.experience,
                        st.session_state.projects,
                    ) = extract_info_from_pdf_new(resume_pdf)
            except Exception as e:
                st.error(f"An error occurred: {e}")        
        else:
            st.error("Please upload a resume PDF.")

    # Display skill selection only after form submission
    if st.session_state.submitted and st.session_state.extracted_skills:
        st.write("Select up to 5 skills from the extracted list:")
        st.write("Note: Please scale your skills before selecting the other skills")
        
        # Ensure that default values are a subset of the available options
        valid_defaults = [skill for skill in st.session_state.selected_skills if skill in st.session_state.extracted_skills]
        
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
                        job_title,
                        skills_with_scale,
                        st.session_state.experience,
                        st.session_state.projects,
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
            scorecard = calculate_scorecard(st.session_state.quiz_data, st.session_state.user_answers)
            display_scorecard(scorecard)

def calculate_scorecard(quiz_data, user_answers):
    scorecard = {
        "Technical": {"correct": 0, "total": 0},
        "Logical Reasoning": {"correct": 0, "total": 0},
        "Communication": {"correct": 0, "total": 0},
        "Work Experience": {"correct": 0, "total": 0},
        "Total": {"correct": 0, "total": 0}
    }

    category_mapping = {
        "Technical Questions": "Technical",
        "Logical Reasoning Questions": "Logical Reasoning",
        "Communication Questions": "Communication",
        "Work Experience Questions": "Work Experience"
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
    st.header("Quiz Scorecard")
    
    for category, score in scorecard.items():
        if category != "Total":
            correct = score['correct']
            total = score['total']
            percentage = (correct / total) * 100 if total > 0 else 0
            st.markdown(f"{category} Score:- **{percentage:.2f}%**")
    
    total_correct = scorecard['Total']['correct']
    total_questions = scorecard['Total']['total']
    total_percentage = (total_correct / total_questions) * 100 if total_questions > 0 else 0
    
    st.write(f"Total Score: {total_correct}/{total_questions} ({total_percentage:.2f}%)")

if __name__ == "__main__":
    main()

