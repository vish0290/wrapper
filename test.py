import ollama
from langchain_ollama import ChatOllama 

cot = '\nExplain the solution for the above question clearly step by step.'
self_refl = '\n Understand the solution provided with respect to the qusetion and give a score from 1 to 10 how relavant it is with respect to the question .'
tool_pro = '\n Understand the user query and give a boolean value whether the tool is required or not. If yes, suggest the tool that can be used to solve the problem. If no, explain why the tool is not required.'
available_tools = ['Google Search','Code_sandbox']


def cot_resp(user_query,cot):
    llm = ChatOllama(model="llama3.2", temperature=0.6, top_p=0.9, top_k=30, num_predict=350)
    final_prompt = user_query + cot
    response = llm.invoke(final_prompt)
    return response.content

def self_reflection(response,user_query,self_refl):
    llm = ChatOllama(model="llama3.2", temperature=0.3, top_p=0.45, top_k=40, num_predict=150)
    final_prompt = f"""
    The user query is {user_query}
    The reponse generated {response}
    Your task {self_refl}
    """
    response = llm.invoke(final_prompt)
    return response.content

def summarize_response(response):
    llm = ChatOllama(model="llama3.2", temperature=0.3, top_p=0.45, top_k=30, num_predict=150)
    final_prompt = f"Summarize the response: {response}"
    response = llm.invoke(final_prompt)
    return response.content

def regenerate_response(cot_response,user_query,self_response):
    llm = ChatOllama(model="llama3.2", temperature=0.45, top_p=0.5, top_k=30, num_predict=100)
    final_prompt = f"""
    The user query is {user_query}\n\n
    The reponse generated {cot_response}\n\n
    Here is the feedback from the self reflection {self_response}\n\n
    Your task is to regenerate the response based on the feedback provided.
    """
    response = llm.invoke(final_prompt)
    return response.content

def tool_suggestion(user_query,tool_suggestion):
    llm = ChatOllama(model="llama3.2", temperature=0.6, top_p=0.9, top_k=30, num_predict=100)
    final_prompt = user_query + tool_pro + f"Here are the available tools: {available_tools}"
    response = llm.invoke(final_prompt)
    return response.content

user_query = input("User query: ")
cot_response = cot_resp(user_query,cot)
print("Chain of thought response: ",cot_response)
self_reflection_response = self_reflection(cot_response,user_query,self_refl)
regen_response = regenerate_response(cot_response,user_query,self_reflection_response)
print("Self reflection response: ",self_reflection_response)
final_response = summarize_response(regen_response)
print("Final response: ",final_response)
tool_suggestion_response = tool_suggestion(user_query,tool_suggestion)
print("Tool suggestion response: ",tool_suggestion_response)
