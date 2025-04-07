from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.Agents import Cardiologist, Pulmonologist, Psychologist
import json, os

load_dotenv()

with open("sample_report_1.txt","r") as file:
    medical_report = file.read()

agents = {
    "Cardiologist":Cardiologist(medical_report), ## agent_name:agent
    "Pulmonologist":Pulmonologist(medical_report),
    "Psychologist":Psychologist(medical_report)
}

def get_response(agent_name, agent):
    response = agent.run()
    return agent_name, response

responses = {}
with ThreadPoolExecutor() as executer:
    futures = {executer.submit(get_response, name, agent): name for name, agent in agents.items()}

    for future in as_completed(futures):
        agent_name, response = future.result()
        responses[agent_name] = response
    
for agent_name, report in responses.items():
    print("\n" + "="*60)
    print(f"🩺 {agent_name} Report")
    print("="*60)
    print(report)

