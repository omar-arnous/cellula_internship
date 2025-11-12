# tools/context_splitter.py
from langchain.tools import StructuredTool
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

class ContextSplitInput(BaseModel):
    text: str = Field(..., description="The user's full message or context to analyze.")

def build_context_splitter_tool(llm):
    """
    Build a tool that splits user input into background context and actual question.
    """

    def split_context(text: str) -> str:
        prompt_text = PromptTemplate.from_template(open("prompts/context_splitter_prompt.txt").read())
        formatted = prompt_text.format(text=text)
        response = llm.invoke(formatted)

        # Return plain text directly
        return response.content if hasattr(response, "content") else str(response)

    return StructuredTool.from_function(
        func=split_context,
        name="ContextSplitter",
        description=(
            "You should run this tool before checking for context presence or performing a web search, "
            "It separates the input into two labeled parts: 'Background:' and 'Question:'. "
            "After using this tool, you MUST combine both parts into one message in the format:\n\n"
            "'Background: <background text>\\nQuestion: <question text>'\n\n"
            "and pass that combined text as the input to the next tool, such as ContextPresenceJudge."
        ),
        args_schema=ContextSplitInput,
    )