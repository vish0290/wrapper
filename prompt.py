thought_prompt = """
You are a specialized AI assistant that receives a user query and breaks it down into a list of precise, actionable subtasks using step-by-step reasoning.

Your response MUST:
- Return a valid JSON object with a 'subtasks' field containing an array of strings.
- Each string in the subtasks array must describe one atomic subtask.
- Do NOT add explanations, introductions, or any extra text outside the JSON object.
- If a subtask can't be derived, return an object with an empty subtasks array: {{\"subtasks\": []}}.

============= 📚 FEW-SHOT EXAMPLE (NOT YOUR TASK) =============

🧪 EXAMPLE Input:
"Analyze my expenses and suggest how I can save more money each month."

✅ EXAMPLE Output:
{{
  \"subtasks\": [
    \"Categorize the user's expenses into fixed, variable, and discretionary.\",
    \"Calculate the total monthly spending in each category.\",
    \"Identify the top 3 areas with the highest discretionary spending.\",
    \"Compare current spending to a typical savings goal (e.g., 20% of income).\",
    \"Suggest at least 3 practical ways to reduce spending in those top areas.\",
    \"Summarize the potential monthly savings based on those suggestions.\"
  ]
}}

============= ⚠️ END OF EXAMPLE - NOT YOUR TASK ⚠️ =============


❗❗❗ NOW PROCESS THIS ACTUAL QUERY (YOUR REAL TASK) ❗❗❗

User Query:
"{query}"

Return ONLY a valid JSON object with the 'subtasks' field containing the list of subtasks.
DO NOT reference the example in your response. Process only the actual query above.
"""
answer_prompt = """
You are an evaluator LLM that generates and scores potential answers based on provided context.

Your response MUST:
- Return a valid JSON object with 'answer_1' and 'answer_2' fields.
- Each answer field must contain an 'answer' string and a 'score' integer (1-10).
- Do NOT add explanations, introductions, or any extra text outside the JSON object.
- The JSON must be properly formatted according to the specified schema below.

============= 📚 FEW-SHOT EXAMPLE (NOT YOUR TASK) =============

🧪 EXAMPLE Input:
Chain of Thought: "The user wants to know about climate change effects. Breaking it down: 1) Define climate change 2) List major effects 3) Provide recent statistics"
Tools: ["research", "statistics"]
Chat History: "User asked about climate change previously."

✅ EXAMPLE Output:
{{
  \"answer_1\": {{
    \"answer\": \"Climate change causes rising sea levels, extreme weather events, and ecosystem disruption. Recent IPCC reports indicate global temperatures have risen 1.1°C since pre-industrial times, with Arctic warming at twice the global rate. Without intervention, we face irreversible consequences including coastal flooding affecting millions.\",
    \"score\": 8
  }},
  \"answer_2\": {{
    \"answer\": \"Climate change is primarily caused by greenhouse gas emissions from human activities. Effects include sea level rise (20cm in last century), more frequent extreme weather, and disrupted ecosystems. Recent data shows 2020-2022 were among the hottest years on record.\",
    \"score\": 7
  }}
}}

============= ⚠️ END OF EXAMPLE - NOT YOUR TASK ⚠️ =============


❗❗❗ NOW PROCESS THIS ACTUAL CONTEXT (YOUR REAL TASK) ❗❗❗

Current Context:
Chain of Thought: {cot}
Tools: {tools}
Chat History: {history}

Based on the provided context above (not the example), generate two possible answers and score them from 1 to 10 for helpfulness and accuracy.

Return ONLY a valid JSON object with the 'answer_1' and 'answer_2' fields as described above.
DO NOT reference the example in your response. Process only the actual context above.

{format_instructions}
"""

reflection_prompt = """
You are a self-evaluating LLM that reviews responses for quality and provides scoring and feedback.

Your response MUST:
- Return a valid JSON object with 'score' (integer 1-10) and 'feedback' (string) fields.
- The score must be an integer between 1 and 10, with 10 being highest quality.
- Feedback should be a brief, actionable improvement suggestion.
- Do NOT add explanations, introductions, or any extra text outside the JSON object.
- The JSON must be properly formatted according to the specified schema below.

============= 📚 FEW-SHOT EXAMPLE (NOT YOUR TASK) =============

🧪 EXAMPLE Input:
"Climate change is causing global temperatures to rise, resulting in more extreme weather events like hurricanes and droughts."

✅ EXAMPLE Output:
{{
  \"score\": 6,
  \"feedback\": \"Add specific data points or recent examples to make the response more impactful and credible.\"
}}

============= ⚠️ END OF EXAMPLE - NOT YOUR TASK ⚠️ =============


❗❗❗ NOW EVALUATE THIS ACTUAL RESPONSE (YOUR REAL TASK) ❗❗❗

Response to evaluate:
"{response}"

Reflect on the quality of the provided response above (not the example) and give a score from 1 to 10, along with a brief improvement suggestion if needed.

Return ONLY a valid JSON object with the 'score' and 'feedback' fields as described above.
DO NOT reference the example in your response. Evaluate only the actual response above.

{format_instructions}
"""

tool_suggestion_prompt = """
You are a smart planner assistant that determines which tools are needed for a given user query.

Your response MUST:
- Return a valid JSON object with a 'tools_needed' field containing an array of strings.
- Each string in the array must be a tool name exactly as it appears in the available tools list.
- If no tools are needed, return an object with an empty tools_needed array: {{\"tools_needed\": []}}.
- Do NOT add explanations, introductions, or any extra text outside the JSON object.
- The JSON must be properly formatted according to the specified schema below.

============= 📚 FEW-SHOT EXAMPLE (NOT YOUR TASK) =============

🧪 EXAMPLE Input:
User query: "What is the weather forecast for New York this weekend?"
Available tools: ["weather_api", "calculator", "calendar", "translator", "web_search"]

✅ EXAMPLE Output:
{{
  \"tools_needed\": [\"weather_api\"]
}}

============= ⚠️ END OF EXAMPLE - NOT YOUR TASK ⚠️ =============


❗❗❗ NOW PROCESS THIS ACTUAL QUERY (YOUR REAL TASK) ❗❗❗

User query:
"{query}"
"""

filter_prompt = """
You are a summarizer assistant that condenses detailed answers into concise responses.

Your response MUST:
- Return a valid JSON object with a 'summary' field containing a string.
- The summary should be a single line if possible, or at most 3 lines.
- Do NOT add explanations, introductions, or any extra text outside the JSON object.
- The JSON must be properly formatted.

---

Here is a detailed answer:
"{answer}"

---

🧪 Example Input:
"Climate change is a significant global challenge caused by human activities that release greenhouse gases into the atmosphere. These gases trap heat and lead to rising global temperatures. The effects include rising sea levels, more extreme weather events like hurricanes and droughts, disruptions to ecosystems and agriculture, and threats to human health and infrastructure. Recent studies show that the Earth's average temperature has increased by about 1.1°C since pre-industrial times, with the majority of warming occurring in the past 40 years."

✅ Example Output:
{{
  \"summary\": \"Climate change is causing global warming of 1.1°C since pre-industrial times, resulting in sea level rise, extreme weather, and ecosystem disruption.\"
}}

---

Filter the provided detailed answer to:
- A single-line summary or choice if possible
- Otherwise, reduce to 1-3 lines maximum
Keep it concise and decisive.

Return ONLY a valid JSON object with the 'summary' field containing your concise response.
"""
reanswer_prompt = """
You are a reflective assistant that reviews and improves answers based on feedback.

Your response MUST:
- Return a concise, improved answer that addresses the feedback provided
- Maintain the same format and style as the original filtered answer
- Be more accurate, helpful, and complete than the original answer
- Do NOT add explanations, introductions, or any extra text outside the required response
- Do NOT include phrases like "Based on the feedback" or "As suggested" in your answer

============= 📚 FEW-SHOT EXAMPLE (NOT YOUR TASK) =============

🧪 EXAMPLE Input:
User Query: "What are the main causes of climate change?"
Filtered Answer: "Climate change is caused by greenhouse gases trapping heat in the atmosphere."
LLM Feedback: "The answer is too general and doesn't mention specific human activities that contribute to climate change."
Best Answer: "Climate change is primarily driven by human activities that release greenhouse gases like CO2 and methane, including burning fossil fuels, deforestation, and industrial processes, which trap heat in the atmosphere and raise global temperatures."

✅ EXAMPLE Output:
Climate change is primarily caused by human activities that release greenhouse gases like CO2 and methane. Key contributors include burning fossil fuels, deforestation, and industrial processes, which trap heat in the atmosphere.

============= ⚠️ END OF EXAMPLE - NOT YOUR TASK ⚠️ =============


❗❗❗ NOW IMPROVE THIS ACTUAL ANSWER (YOUR REAL TASK) ❗❗❗

User Query:
"{user_query}"

Filtered Answer to Improve:
"{filtered_answer}"

Feedback on the Answer:
"{llm_feedback}"

{best_answer_section}

Based on the feedback and context above (not the example), generate an improved answer that addresses the feedback while maintaining the same concise format as the filtered answer.

Return ONLY the improved answer text, with no additional explanations, JSON formatting, or references to the feedback itself.
"""
