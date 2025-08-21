from typing import Optional, Callable, Dict, Any
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from src.modules.analysis.ragService import answer_with_rag
from src.agents.tools.scout_tool import scout_lc_tool
from src.agents.tools.reader_tool import reader_lc_tool
from src.agents.tools.processor_tool import processor_lc_tool

TOOLS = [t for t in [scout_lc_tool, reader_lc_tool, processor_lc_tool] if t is not None]

class AgentEventsHandler(BaseCallbackHandler):
    def __init__(self, emit: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        self.emit = emit

    def on_tool_start(self, serialized, input_str, **kwargs):
        if self.emit:
            self.emit("agent.tool.start", {"name": serialized.get("name"), "input": input_str})

    def on_tool_end(self, output, **kwargs):
        if self.emit:
            self.emit("agent.tool.end", {"output_preview": str(output)[:300]})

    def on_llm_start(self, serialized, prompts, **kwargs):
        if self.emit:
            self.emit("agent.llm.start", {"prompts": prompts})

    def on_llm_end(self, response, **kwargs):
        if self.emit:
            txt = ""
            try:
                txt = response.generations[0][0].text
            except Exception:
                pass
            self.emit("agent.llm.end", {"output_preview": txt[:300]})

def run_answering_agent(
    question: str,
    namespace: str,
    threshold: float = 0.70,
    emit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> dict:
    rag = answer_with_rag(question, namespace, threshold=threshold, emit=emit)
    if rag.get("ok"):
        return {"source": "rag", **rag}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=500)
    handler = AgentEventsHandler(emit)
    agent = initialize_agent(
        tools=TOOLS,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=False,
    )
    plan = (
        f"You may use tools to search arXiv, read PDFs, and process text into Pinecone namespace '{namespace}'. "
        f"Gather enough relevant context to answer. Once indexing is done, reply 'DONE'.\n\n"
        f"Question: {question}"
    )
    agent.invoke({"input": plan}, callbacks=[handler])

    return {"source": "agent+rAG", **answer_with_rag(question, namespace, threshold=threshold, emit=emit)}
