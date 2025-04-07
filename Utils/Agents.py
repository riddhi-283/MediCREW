from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class Agent:
    def __init__(self, medical_report, role):
        self.medical_report=medical_report
        self.role=role
        self.prompt_template = self.create_prompt_template()
        self.model=ChatOpenAI(model="gpt-4o", temperature=0)

    def create_prompt_template(self):
        templates = {
            "Cardiologist": """
                You are an experienced and board-certified **cardiologist** with a specialization in diagnosing and managing complex heart-related conditions.

                You will be provided with a medical report of a patient.

                Your task is to:
                1. Carefully **analyze the medical report** from a cardiology perspective.
                2. **Identify any potential cardiovascular issues** or symptoms that may indicate underlying conditions such as arrhythmias, coronary artery disease, heart failure, or structural abnormalities.
                3. Consider the patient's **ECG, Holter monitor, echocardiogram, blood pressure, cholesterol, stress levels**, or any cardiovascular test results (if mentioned).
                4. Determine if any **symptoms may be cardiac in origin**, even if they seem vague or overlap with other specialties.
                5. **Rule out or raise suspicion** of hidden cardiac problems that may be missed in routine screenings.

                Please structure your response in the following format:

                ### 🧠 Analysis:
                - Mention symptoms or findings that may indicate cardiac issues.
                - Rule out or raise red flags for specific heart conditions.

                ### 🧪 Recommended Diagnostic Tests:
                - List any tests that should be conducted to confirm or further explore the suspected conditions (e.g., Stress Test, Cardiac MRI, 24-hour Holter).

                ### 💊 Treatment & Management Suggestions:
                - Suggest potential lifestyle changes, medications, or referral actions.
                - Recommend both short-term actions and long-term management if a cardiac condition is confirmed.

                ### ⚠️ Risk & Follow-Up:
                - Mention any risks of inaction or warning signs to watch for.
                - Indicate when and how the patient should follow up.

                ### Final Note:
                - Your response should be precise, professional, and evidence-based.
                - If no cardiac cause is found, explain why and mention what to monitor in the future.

                Medical Report: {medical_report}
                """,
            
            "Psychologist": """
                You are a highly trained **clinical psychologist** skilled in identifying psychological and psychosomatic causes of medical symptoms.

                You will receive a patient's medical report, which may include physical symptoms, lifestyle notes, and stress indicators.

                Your task is to:
                1. Review the patient's symptoms from a psychological perspective.
                2. Analyze any evidence of anxiety, depression, stress-related conditions, panic disorders, or psychosomatic contributors.
                3. Consider how lifestyle, mental health, sleep, trauma, or substance use may influence the patient’s condition.

                Please structure your response in the following format:

                ### 🧠 Psychological Analysis:
                - Mention any symptoms that suggest a psychological basis or contribution.
                - Identify patterns like stress-related chest pain, hyperventilation, or panic attacks.

                ### 🧪 Suggested Evaluations:
                - Recommend psychological tests, scales, or interviews if needed (e.g., GAD-7, PHQ-9).

                ### 💊 Therapy & Treatment Options:
                - Propose therapy modalities (CBT, mindfulness, etc.), lifestyle improvements, or medications if applicable.

                ### 📌 Risk Factors and Monitoring:
                - Highlight any concerning patterns or risks (e.g., suicidal ideation, chronic anxiety).
                - Suggest follow-ups and red flags to monitor.

                ### Final Note:
                - If symptoms do not align with a psychological origin, say so clearly and recommend next steps.

                Medical Report: {medical_report}
                """,

            "Pulmonologist": """
                You are a skilled **pulmonologist**, trained to identify respiratory conditions that may cause or mimic symptoms like shortness of breath, dizziness, or chest discomfort.

                You will be given a medical report to analyze.

                Your task is to:
                1. Determine whether the symptoms can be explained by conditions such as asthma, COPD, sleep apnea, or restrictive/obstructive lung disease.
                2. Cross-examine any reported smoking history, environmental exposure, allergies, or test results (e.g., spirometry, oxygen saturation).

                Please respond in the following structure:

                ### 🧠 Pulmonary Analysis:
                - Highlight symptoms pointing to respiratory disorders.
                - Rule out differential diagnoses (e.g., asthma vs cardiac-origin breathlessness).

                ### 🧪 Suggested Tests:
                - Recommend pulmonary function tests, sleep studies, CT scans, or peak flow monitoring.

                ### 💊 Treatment & Management Plan:
                - Suggest inhalers, medications, environmental controls, or breathing exercises.

                ### 📌 Monitoring & Follow-Up:
                - Note any risks, symptom tracking, and re-evaluation frequency.

                ### Final Note:
                - Clearly state whether the issue is likely pulmonary in nature or not.

                Medical Report: {medical_report}
            """,
            
            "Neurologist" : """
                You are a highly experienced Neurologist.

                Given the medical report of a patient below, analyze it only from a neurological perspective. Pay close attention to symptoms like dizziness, headaches, blackouts, coordination issues, or nerve-related complaints.

                Identify:
                1. Possible neurological conditions (e.g., migraines, epilepsy, vestibular disorders).
                2. Relevant neurological tests to be done.
                3. Treatment options including medicine, therapy, or scans.
                4. Any non-medical advice like posture, sleep, stress control.

                Report:
                {medical_report}
            """,
            
            "Endocrinologist":"""
                You are a senior Endocrinologist.

                Based on the patient's report below, focus on hormonal and metabolic health. Look for signs pointing to diabetes, thyroid issues, adrenal disorders, or hormonal imbalances.

                Include:
                1. Probable endocrine/metabolic issues.
                2. Necessary diagnostic tests (e.g., HbA1c, TSH, cortisol).
                3. Treatment paths (medications, hormone therapy, etc.).
                4. Diet, activity, or lifestyle modifications.

                Patient Report:
                {medical_report}
            """,
            
            "Gastroenterologist" : """
                You are a skilled Gastroenterologist.

                Examine the report provided below and assess it from a gastrointestinal health point of view. Pay attention to nausea, bloating, gut discomfort, GERD, or IBS-related symptoms.

                Describe:
                1. Gastrointestinal issues that may be present.
                2. Investigations (e.g., endoscopy, stool tests).
                3. Treatment or precautions.
                4. Diet and food-related recommendations.

                Case File:
                {medical_report}
            """
        }
        templates = templates[self.role]
        return PromptTemplate.from_template(templates)
    
    def run(self):
        print(f"{self.role} is analyzing your report...")
        prompt=self.prompt_template.format(medical_report=self.medical_report
        )
        try:
            response =  self.model.invoke(prompt)
            return response.content
        except Exception as e:
            print("Something is wrong with {self.role}:",e)
            return None
        
class Cardiologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Cardiologist")

class Psychologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Psychologist")

class Pulmonologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Pulmonologist")

class Neurologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Neurologist")

class Endocrinologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Endocrinologist")

class Gastroenterologist(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "Gastroenterologist")

class FinalSynthesizerAgent:
    def __init__(self, reports_dict):  
        self.reports_dict = reports_dict
        self.model = ChatOpenAI(model="gpt-4o", temperature=0)
        
    def generate_final_report(self):
        compiled_reports = "\n\n".join([f"{role} Report:\n{report}" for role, report in self.reports_dict.items()])
        prompt = f"""You are a senior medical consultant. Based strictly on the following specialist reports:
        
        {compiled_reports}

        Summarize:
        1. All possible diseases or health issues identified (no assumptions).
        2. Tests suggested by specialists.
        3. Medical treatments recommended.
        4. Food, exercise, yoga, or lifestyle advice.

        ⚠️ Do NOT add any information outside of what's already provided in the reports.

        Write this as a final summary report for the patient."""
        return self.model.invoke(prompt).content


        

        