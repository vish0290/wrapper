from typing import List, Dict, Any
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Define Pydantic models for structured outputs
class ThoughtOutput(BaseModel):
    chain_of_thought: str = Field(description="Step-by-step reasoning process")
    modular_breakdown: List[str] = Field(description="List of modular parts to solve the problem")

class ScoredAnswer(BaseModel):
    answer: str = Field(description="Potential answer to the query")
    score: int = Field(description="Score from 1-10 for helpfulness and accuracy")

class ScoredAnswerOutput(BaseModel):
    answer_1: ScoredAnswer = Field(description="First potential answer with score")
    answer_2: ScoredAnswer = Field(description="Second potential answer with score")

class ReflectionOutput(BaseModel):
    score: int = Field(description="Quality score from 1-10")
    feedback: str = Field(description="Brief improvement suggestion")

class ToolSuggestionOutput(BaseModel):
    tools_needed: List[str] = Field(description="List of relevant tools needed", default_factory=list)

class ReasoningPipeline:
    def __init__(self, model_id="meta-llama/Meta-Llama-3.2-8B-Instruct", device="cuda"):
        """
        Initialize the reasoning pipeline with a specified model
        
        Args:
            model_id (str): HuggingFace model ID
            device (str): Device to load model on ('cuda', 'cpu', etc.)
        """
        self.model_id = model_id
        self.device = device
        
        # Create parsers for structured outputs
        self.thought_parser = PydanticOutputParser(pydantic_object=ThoughtOutput)
        self.scored_answers_parser = PydanticOutputParser(pydantic_object=ScoredAnswerOutput)
        self.reflection_parser = PydanticOutputParser(pydantic_object=ReflectionOutput)
        self.tool_suggestion_parser = PydanticOutputParser(pydantic_object=ToolSuggestionOutput)

        # Create prompt templates
        self.thought_prompt = PromptTemplate(
            template="""You are an expert reasoning assistant.
User has asked: "{query}"
First, think through the question step by step (chain of thought).
Then, break the problem into smaller modular parts to solve.

{format_instructions}
""",
            input_variables=["query"],
            partial_variables={"format_instructions": self.thought_parser.get_format_instructions()}
        )

        self.answer_prompt = PromptTemplate(
            template="""You are an evaluator LLM.
Context:
Chain of Thought: {cot}
Tools: {tools}
Chat History: {history}
Based on the context, generate two possible answers and score them from 1 to 10 for helpfulness and accuracy.

{format_instructions}
""",
            input_variables=["cot", "tools", "history"],
            partial_variables={"format_instructions": self.scored_answers_parser.get_format_instructions()}
        )

        self.reflection_prompt = PromptTemplate(
            template="""You are a self-evaluating LLM.
You just generated this answer:
"{response}"
Reflect on the quality of this response and give a score from 1 to 10, along with a brief improvement suggestion if needed.

{format_instructions}
""",
            input_variables=["response"],
            partial_variables={"format_instructions": self.reflection_parser.get_format_instructions()}
        )

        self.tool_suggestion_prompt = PromptTemplate(
            template="""You are a smart planner assistant.
User query:
"{query}"
Here is the list of available tools:
{available_tools}
Determine whether this task requires any of these tools.
If yes, pick relevant tools from the list.
If not, make tools_needed an empty list.

{format_instructions}
""",
            input_variables=["query", "available_tools"],
            partial_variables={"format_instructions": self.tool_suggestion_parser.get_format_instructions()}
        )

        self.filter_prompt = PromptTemplate(
            template="""You are a summarizer assistant.
Here is a detailed answer:
"{answer}"
Filter this to:
- A single-line summary or choice if possible
- Otherwise, reduce to 1-3 lines maximum
Keep it concise and decisive.
""",
            input_variables=["answer"]
        )
        
        # Load model and tokenizer once
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer once"""
        print(f"Loading model {self.model_id}...")
        
        # Check if CUDA is available when device is set to cuda
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Load model with appropriate settings
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        print("Model loaded successfully!")
    
    def _create_pipeline(self, max_new_tokens, temperature, top_p, do_sample=True):
        """Create a text generation pipeline with specified parameters"""
        text_gen_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            do_sample=do_sample
        )
        
        return HuggingFacePipeline(pipeline=text_gen_pipeline)
    
    def generate_thought_and_problem_modularization(self, query: str) -> ThoughtOutput:
        """Generate thought process with structured output"""
        # Using parameters from original function: max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True
        llm = self._create_pipeline(max_new_tokens=300, temperature=0.7, top_p=0.9, do_sample=True)
        chain = LLMChain(llm=llm, prompt=self.thought_prompt)
        result = chain.run(query=query)
        return self.thought_parser.parse(result)
    
    def generate_scored_answers(self, cot: str, tools: str, history: str = "") -> ScoredAnswerOutput:
        """Generate and score answers with structured output"""
        # Using parameters from original function: max_new_tokens=400, temperature=0.7, top_p=0.9, do_sample=True
        llm = self._create_pipeline(max_new_tokens=400, temperature=0.7, top_p=0.9, do_sample=True)
        chain = LLMChain(llm=llm, prompt=self.answer_prompt)
        result = chain.run(cot=cot, tools=tools, history=history)
        return self.scored_answers_parser.parse(result)
    
    def generate_filtered_answer(self, answer: str) -> str:
        """Generate filtered answer (simple text output)"""
        # Using parameters from original function: max_new_tokens=100, temperature=0.6, top_p=0.8, do_sample=False
        llm = self._create_pipeline(max_new_tokens=100, temperature=0.6, top_p=0.8, do_sample=False)
        chain = LLMChain(llm=llm, prompt=self.filter_prompt)
        result = chain.run(answer=answer)
        return result
    
    def generate_self_reflection(self, response: str) -> ReflectionOutput:
        """Generate self-reflection with structured output"""
        # Using parameters from original function: max_new_tokens=150, temperature=0.6, top_p=0.9, do_sample=True
        llm = self._create_pipeline(max_new_tokens=150, temperature=0.6, top_p=0.9, do_sample=True)
        chain = LLMChain(llm=llm, prompt=self.reflection_prompt)
        result = chain.run(response=response)
        return self.reflection_parser.parse(result)
    
    def generate_tool_suggestions(self, query: str, available_tools: List[str]) -> ToolSuggestionOutput:
        """Generate tool suggestions with structured output"""
        # Using parameters from original function: max_new_tokens=150, temperature=0.7, top_p=0.9, do_sample=True
        llm = self._create_pipeline(max_new_tokens=150, temperature=0.7, top_p=0.9, do_sample=True)
        tools_str = "\n".join(f"- {tool}" for tool in available_tools)
        chain = LLMChain(llm=llm, prompt=self.tool_suggestion_prompt)
        result = chain.run(query=query, available_tools=tools_str)
        return self.tool_suggestion_parser.parse(result)
    
    def process_query(self, query: str, available_tools: List[str] = None, history: str = "") -> Dict[str, Any]:
        """
        Process a query through the full pipeline
        
        Args:
            query (str): The user's query
            available_tools (List[str], optional): List of available tools
            history (str, optional): Chat history
            
        Returns:
            Dict[str, Any]: The structured results
        """
        if available_tools is None:
            available_tools = []
        
        # Step 1: Generate thought process
        thought_output = self.generate_thought_and_problem_modularization(query)
        
        # Step 2: Check if tools are needed
        tool_suggestions = self.generate_tool_suggestions(query, available_tools)
        
        # Step 3: Generate scored answers
        scored_answers = self.generate_scored_answers(
            thought_output.chain_of_thought,
            str(tool_suggestions.tools_needed),
            history
        )
        
        # Step 4: Pick the answer with the highest score
        best_answer = scored_answers.answer_1
        if scored_answers.answer_2.score > scored_answers.answer_1.score:
            best_answer = scored_answers.answer_2
        
        # Step 5: Filter the best answer
        filtered_answer = self.generate_filtered_answer(best_answer.answer)
        
        # Step 6: Generate self-reflection
        reflection = self.generate_self_reflection(filtered_answer)
        
        # Return structured results
        return {
            "thought_process": thought_output.dict(),
            "tool_suggestions": tool_suggestions.dict(),
            "scored_answers": scored_answers.dict(),
            "filtered_answer": filtered_answer,
            "reflection": reflection.dict()
        }
    
    def unload(self):
        """Unload model from memory"""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Model unloaded from memory.")

# Example usage
if __name__ == "__main__":
    # Initialize the pipeline once
    pipeline = ReasoningPipeline(
        model_id="meta-llama/Meta-Llama-3.2-8B-Instruct",  # Use smaller model (8B) if VRAM is limited
        device="cuda"  # Use "cpu" if no GPU is available
    )
    
    # Example queries
    queries = [
        "What are the key factors to consider when implementing a machine learning pipeline?",
        "How can I improve my Python coding skills?",
        "What are the benefits and drawbacks of microservices architecture?"
    ]
    
    # Available tools
    available_tools = ["DataAnalyzer", "ModelSelector", "HyperparameterTuner", "DataVisualizer"]
    
    # Process multiple queries without reloading the model
    for i, query in enumerate(queries):
        print(f"\n\nProcessing Query {i+1}: {query}")
        result = pipeline.process_query(query, available_tools)
        print(f"Filtered Answer: {result['filtered_answer']}")
        print(f"Reflection Score: {result['reflection']['score']}/10")
        print(f"Reflection Feedback: {result['reflection']['feedback']}")
    
    # Unload model when done (optional)
    pipeline.unload()