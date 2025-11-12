from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import Tool

def context_presence_tool(llm):
    prompt = PromptTemplate.from_template(open("prompts/context_judge_prompt.txt").read())
    chain = LLMChain(llm=llm, prompt=prompt)
    return Tool.from_function(
        func=chain.run,
        name="ContextPresenceJudge",
        description=(
            "Use this tool after every user query (or after ContextSplitter) to decide whether enough context is present. "
            "If the input includes 'Background:' and 'Question:' sections, you MUST read and consider both when deciding. "
            "If the message contains only a question with no background, assume context is missing and call WebSearchTool."
        )
    )
