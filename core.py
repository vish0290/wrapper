from typing_extensions import TypedDict
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import requests
from models import *
from prompt import *

load_dotenv()

class ModelHandler:
    def __init__(self,model_name: str):
        self.model_name = model_name
    def _load_model(self):
        """
        Load the model based on the model name.
        """
        response = requests.post("http://localhost:11434/api/chat",
                                 json={
                "model": self.model_name,
                "messages": [],         
            }
        )
        if response.status_code == 200 and response.json().get("done_reason")=="load" and response.json().get("done")==True:
            print(f"Model {self.model_name} loaded successfully.")
        else:
            print(f"Failed to load model {self.model_name}.")
            raise Exception("Model loading failed.")
    
    def _unload_model(self):
        """
        Unload the model based on the model name.
        """
        response = requests.post("http://localhost:11434/api/chat",
                                 json={
                "model": self.model_name,
                "messages": [],
                "keep_alive": 0,  # Unload the model   
            }
        )
        if response.status_code == 200 and response.json().get("done_reason")=="unload" and response.json().get("done")==True:
            print(f"Model {self.model_name} unloaded successfully.")
        else:
            print(f"Failed to unload model {self.model_name}.")
            raise Exception("Model unloading failed.")
        
    def llm_call(self,prompt:str, temperature: float = 0.7, max_length: int = 100, top_p: float = 0.9, top_k: int= 25) -> str:
        
        llm = ChatOllama(model=self.model_name, temperature=temperature, top_p=top_p, top_k=top_k, num_predict=max_length)
        response = llm.invoke(prompt)
        return response.content

class ReasoningNodes:
    def __init__(self):
        self.thought_parser = PydanticOutputParser(pydantic_object=ThoughtOutput)
        self.thought_prompt = PromptTemplate(template=thought_prompt, input_variables=["query"],partial_variables={"format_instructions": self.thought_parser.get_format_instructions()})
        self.answer_parser = PydanticOutputParser(pydantic_object=ScoredAnswerOutput)
        self.answer_prompt = PromptTemplate(template=answer_prompt, input_variables=["cot", "tools", "history"],partial_variables={"format_instructions": self.answer_parser.get_format_instructions()})
        self.reflection_parser = PydanticOutputParser(pydantic_object=ReflectionOutput)
        self.reflection_prompt = PromptTemplate(template=reflection_prompt, input_variables=["response"],partial_variables={"format_instructions": self.reflection_parser.get_format_instructions()})
        self.tool_parser = PydanticOutputParser(pydantic_object=ToolSuggestionOutput)
        self.tool_suggestion_prompt = PromptTemplate(template=tool_suggestion_prompt, input_variables=["query", "available_tools"],partial_variables={"format_instructions": self.tool_parser.get_format_instructions()})
        self.filter_prompt = PromptTemplate(template=filter_prompt, input_variables=["answer"])
        self.regenerate_prompt = PromptTemplate(template=reanswer_prompt, input_variables=["user_query","filtered_answer","llm_feedback","best_answer_section"])
        
    def CoT_generate(self, query: str,model_handler: ModelHandler) -> ThoughtOutput:
        """
        Generate a chain of thought and modular breakdown for the given query.
        """
        prompt = self.thought_prompt.format(query=query)
        response = model_handler.llm_call(prompt,temperature=0.7, top_p=0.9, top_k=50, max_length=300)
        return self.thought_parser.parse(response)
    
    def generate_scored_answer(self, cot: str, tools: List[str], history: str,model_handler: ModelHandler) -> ScoredAnswerOutput:
        """
        Generate two potential answers with scores based on the chain of thought and tools.
        """
        prompt = self.answer_prompt.format(cot=cot, tools=tools, history=history)
        response = model_handler.llm_call(prompt,temperature=0.7, top_p=0.9, top_k=40, max_length=400)
        return self.answer_parser.parse(response)
    
    def generate_filtered_answer(self, answer: str,model_handler: ModelHandler) -> str:
        """
        Filter the generated answer to a concise summary or choice.
        """
        prompt = self.filter_prompt.format(answer=answer)
        response = model_handler.llm_call(prompt,temperature=0.6, top_p=0.8, top_k=30, max_length=100)
        return response
    
    def regenerate_answer(self, user_query: str, filtered_answer: str, llm_feedback: str, best_answer_section: str ,model_handler: ModelHandler) -> str:
        """
        Filter the generated answer to a concise summary or choice.
        """
        prompt = self.regenerate_prompt.format(user_query=user_query, filtered_answer=filtered_answer, llm_feedback=llm_feedback, best_answer_section=best_answer_section)
        response = model_handler.llm_call(prompt,temperature=0.6, top_p=0.8, top_k=30, max_length=100)
        return response
    
    def generate_reflection(self, response: str,model_handler: ModelHandler) -> ReflectionOutput:
        """
        Reflect on the generated answer and provide a quality score and feedback.
        """
        prompt = self.reflection_prompt.format(response=response)
        response = model_handler.llm_call(prompt,temperature=0.6, top_p=0.9, top_k=40, max_length=150)
        return self.reflection_parser.parse(response)
    
    def generate_tool_suggestion(self, query: str, available_tools: List[str],model_handler: ModelHandler) -> ToolSuggestionOutput:
        """
        Suggest relevant tools needed for the task based on the query.
        """
        prompt = self.tool_suggestion_prompt.format(query=query, available_tools=available_tools)
        response = model_handler.llm_call(prompt,temperature=0.7, top_p=0.9, top_k=45, max_length=150)
        return self.tool_parser.parse(response)
    
    