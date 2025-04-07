from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.Agents import Cardiologist, Pulmonologist, Psychologist, Neurologist, Endocrinologist, Gastroenterologist, FinalSynthesizerAgent
from Utils.FileHandler import save_report
import json, os

load_dotenv()

with open("sample_report_1.txt","r") as file:
    medical_report = file.read()

agents = {
    "Cardiologist":Cardiologist(medical_report), ## agent_name:agent
    "Pulmonologist":Pulmonologist(medical_report),
    "Psychologist":Psychologist(medical_report),
    "Neurologist": Neurologist(medical_report),
    "Endocrinologist": Endocrinologist(medical_report),
    "Gastroenterologist": Gastroenterologist(medical_report)
}

# Ensure that output directory exists
os.makedirs("Reports", exist_ok=True)

# Helper function to get agent response
def get_response(agent_name, agent):
    response = agent.run()
    return agent_name, response

# Run all agents parallely
responses = {}
with ThreadPoolExecutor() as executer:
    futures = {executer.submit(get_response, name, agent): name for name, agent in agents.items()}

    for future in as_completed(futures):
        agent_name, response = future.result()
        responses[agent_name] = response
        save_report(agent_name, response)
    


final_agent = FinalSynthesizerAgent(responses)
final_report = final_agent.generate_final_report()
save_report("Final_summary_report",final_report)