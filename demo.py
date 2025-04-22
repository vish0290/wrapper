from core import ReasoningNodes, ModelHandler

model_nodes = ModelHandler(model_name="llama3.2")
model_nodes._load_model()
reasoning_nodes = ReasoningNodes()
query = "How many r are there in the word strawberry?"
response = reasoning_nodes.CoT_generate(query, model_nodes)
print("CoT Breakdown:", response.subtasks)
tools = ["calculator", "research"]

scored_answer = reasoning_nodes.generate_scored_answer(cot=response.subtasks, tools=tools, history="User asked about saving money previously.", model_handler=model_nodes)
best_answer = scored_answer.answer_1.answer
if scored_answer.answer_1.score < scored_answer.answer_2.score:
    best_answer = scored_answer.answer_2.answer
print("Best Answer:", best_answer)


filtered_answer = reasoning_nodes.generate_filtered_answer(answer=best_answer, model_handler=model_nodes)
print("Filtered Answer:", filtered_answer)

reflection = reasoning_nodes.generate_reflection(response=filtered_answer, model_handler=model_nodes)
print("Reflection Score:", reflection.score)
print("Reflection Feedback:", reflection.feedback)

regen = reasoning_nodes.regenerate_answer(user_query=query, filtered_answer=filtered_answer, llm_feedback=reflection.feedback, best_answer_section=best_answer, model_handler=model_nodes)
print("Regenerated Answer:", regen)

reflection = reasoning_nodes.generate_reflection(response=regen, model_handler=model_nodes)
print("Regenerated Reflection Score:", reflection.score)
print("Regenerated Reflection Feedback:", reflection.feedback)
print("Final Answer:", regen)



model_nodes._unload_model()
reasoning_nodes = None
model_nodes = None





