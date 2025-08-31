from typing import Optional, Callable, Dict, Any, AsyncGenerator
from datetime import datetime
import asyncio
import time

from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.schema import AgentAction, AgentFinish

from src.modules.analysis.ragService import answer_with_rag
from src.agents.tools.scout_tool import scout_tool, scout_lc_tool
from src.agents.tools.reader_tool import reader_tool, reader_lc_tool
from src.agents.tools.processor_tool import processor_tool, processor_lc_tool

TOOLS = [t for t in [scout_lc_tool, reader_lc_tool, processor_lc_tool] if t is not None]

class AgentEventsHandler(BaseCallbackHandler):
    def __init__(self, emit: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        self.emit = emit

    def _ts(self) -> str:
        return datetime.now().isoformat()

    def on_tool_start(self, serialized, input_str, **kwargs):
        if self.emit:
            self.emit("agent.tool.start", {"tool": serialized.get("name"), "input": input_str, "ts": self._ts()})

    def on_tool_end(self, output, **kwargs):
        if self.emit:
            self.emit("agent.tool.end", {"output_preview": str(output)[:300], "ts": self._ts()})

    def on_llm_start(self, serialized, prompts, **kwargs):
        if self.emit:
            preview = str(prompts[0])[:200] if prompts else ""
            self.emit("agent.llm.start", {"prompt_preview": preview, "ts": self._ts()})

    def on_llm_end(self, response, **kwargs):
        if self.emit:
            try:
                txt = response.generations[0][0].text
            except Exception:
                txt = "No text"
            self.emit("agent.llm.end", {"output_preview": txt[:300], "ts": self._ts()})

    def on_agent_action(self, action: AgentAction, **kwargs):
        if self.emit:
            self.emit("agent.decision", {
                "action": "use_tool",
                "tool": action.tool,
                "input": action.tool_input,
                "reasoning": action.log,
                "ts": self._ts()
            })

    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        if self.emit:
            self.emit("agent.decision", {
                "action": "finish",
                "output": finish.return_values.get("output", ""),
                "reasoning": finish.log,
                "ts": self._ts()
            })


class IntelligentAnsweringAgent:
    """RAG-first answering agent with tool fallback."""

    def __init__(self, namespace: str, emit: Optional[Callable[[str, Dict[str, Any]], None]] = None):
        self.namespace = namespace
        self.emit = emit
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=300)
        self.handler = AgentEventsHandler(emit)
        self.agent = initialize_agent(
            tools=TOOLS,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
        )

    async def answer_question(self, question: str, threshold: float = 0.5) -> AsyncGenerator[Dict[str, Any], None]:
        rag = answer_with_rag(question, self.namespace, threshold, emit=self.emit)
        if rag.get("ok"):
            yield {"type": "answer", "source": "rag", **rag}
            return

        try:
            search_results = scout_tool(question, max_results=3)
            if not search_results:
                yield {"type": "error", "message": "No papers found"}
                return

            best_paper = search_results[0]
            pdf_result = reader_tool(best_paper["url"], emit=self.emit)
            
            if not pdf_result.get("ok"):
                yield {"type": "error", "message": f"Failed to read PDF: {pdf_result.get('error')}"}
                return

            process_result = processor_tool(
                pdf_result["text"], 
                namespace=self.namespace, 
                meta={"title": best_paper["title"], "url": best_paper["url"]}
            )
            
            if not process_result.get("ok"):
                yield {"type": "error", "message": f"Failed to process PDF: {process_result.get('error')}"}
                return

            if self.emit:
                self.emit("agent.waiting_for_indexing", {"delay_seconds": 3})
            time.sleep(3)
            
            final_rag = answer_with_rag(question, self.namespace, threshold=0.3, emit=self.emit)
            if final_rag.get("ok"):
                yield {"type": "answer", "source": "agent+rag", **final_rag}
            else:
                yield {"type": "error", "message": "Still no relevant content found after processing"}

        except Exception as e:
            yield {"type": "error", "message": f"Processing failed: {str(e)}"}


async def run_answering_agent_stream(
    question: str,
    namespace: str,
    threshold: float = 0.50,
    emit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    agent = IntelligentAnsweringAgent(namespace, emit)
    async for result in agent.answer_question(question, threshold):
        yield result


def run_answering_agent(
    question: str,
    namespace: str,
    threshold: float = 0.50,
    emit: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> dict:
    results = []

    async def collect():
        async for r in run_answering_agent_stream(question, namespace, threshold, emit):
            results.append(r)

    asyncio.run(collect())
    if results:
        last = results[-1]
        if last.get("type") == "answer":
            return {"ok": True, "source": last.get("source"), "answer": last.get("answer"), "matches": last.get("matches", [])}
    return {"ok": False, "source": "agent", "answer": None, "matches": []}
