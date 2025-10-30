# app/rag_system/rag_agent.py

import os
from crewai import Agent, Task, Crew, Process,LLM
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from core.config import settings
from phoenix.otel import register
from phoenix.client import Client
import json
from rag_system.tools import search_collection
from core.config import settings
import logging
logger = logging.getLogger(__name__)


phoenix_client = Client(base_url=settings.PHOENIX_COLLECTOR_ENDPOINT)

def get_prompts(prmot_identifier: str):
    prompt_data = phoenix_client.prompts.get(prompt_identifier=prmot_identifier).format()
    prompt_data  = json.loads(prompt_data.messages[0]['content'])
    return prompt_data




# configure the Phoenix tracer
tracer_provider = register(
  project_name="rag-app", # Default is 'default'
  auto_instrument=True ,# Auto-instrument your app based on installed OI dependencies
  endpoint=  settings.PHOENIX_COLLECTOR_ENDPOINT+"/v1/traces", # Phoenix Collector endpoint
)


LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
CrewAIInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)



class RAGAgent:
    """
    Manages RAG operations using a CrewAI setup within a class structure.
    Initializes agents and LLM, then creates tasks and crew dynamically per query.
    """
    def __init__(self, model_name: str = None, base_url: str = None):
        """
        Initializes the LLM and the agents for the RAG process.
        """
        # --- LLM Configuration ---
        resolved_model_name = model_name or os.getenv("OLLAMA_MODEL", "llama2")
        resolved_base_url = base_url or settings.OLLAMA_URL
        self.document_researcher_prompt = get_prompts("expert-search-query-formulator")
        self.insight_synthesizer_prompt = get_prompts("insight-synthesizer-analyst")
        
        logger.info(f"Initializing RAGAgent with LLM: model='{resolved_model_name}', base_url='{resolved_base_url}'")
        try:
            self.llm = LLM(
                model=resolved_model_name,
                base_url=resolved_base_url,
                temperature=0.2, # Adjusted for more balanced responses
                timeout=300,
                verbose=True,  # Enable verbose logging for debugging
                max_tokens=40000,  # Use maximum context length available
                num_ctx=40000,     # Context window size
                )
            """self.llm = LLM(
            model="gemini/gemini-2.5-flash-lite", # A capable model for complex planning tasks
            temperature=0.01, # Moderate creativity for planning
            max_retries=2,
            api_key=os.getenv("GOOGLE_API_KEY"),
            verbose=True,)"""


        except Exception as e:
            logger.exception(f"Error initializing Ollama LLM: {e}")
            # Decide how to handle LLM failure: raise error or set self.llm to None
            self.llm = None
            # Or raise RuntimeError(f"Failed to initialize LLM: {e}")

        if not self.llm:
             # Prevent agent initialization if LLM failed
             self.document_researcher = None
             self.insight_synthesizer = None
             logger.error("Warning: LLM initialization failed. Agents not created.")
             return

        # --- Agent Definitions ---
        # These are now instance attributes, initialized once per RAGAgent instance.

        # Agent 1: Document Researcher
        self.document_researcher = Agent(
            role=self.document_researcher_prompt['role'],
            goal=self.document_researcher_prompt['goal'],
            backstory=self.document_researcher_prompt['backstory'],
            tools=[search_collection],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

        # Agent 2: Insight Synthesizer
        self.insight_synthesizer = Agent(
            role=self.insight_synthesizer_prompt['role'],
            goal=self.insight_synthesizer_prompt['goal'],
            backstory=self.insight_synthesizer_prompt['backstory'],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            tools=[] # No external tools needed for synthesis.
        )

    def _create_tasks(self, query: str):
        """
        Creates the research and synthesis tasks for a given query.
        Internal helper method.
        """
        if not self.document_researcher or not self.insight_synthesizer:
            raise RuntimeError("Agents not initialized properly (LLM might have failed).")

        # Task for the Document Researcher
        research_task = Task(
            description=self.document_researcher_prompt["task"]['description'].format(query=query),
            expected_output=self.document_researcher_prompt["task"]['expected_output'],
            agent=self.document_researcher,
        )

        # Task for the Insight Synthesizer
        synthesis_task = Task(
            description=self.insight_synthesizer_prompt["task"]['description'].format(query=query),
            expected_output=self.insight_synthesizer_prompt["task"]['expected_output'],
            agent=self.insight_synthesizer,
            context=[research_task], # Explicit dependency
        )

        return [research_task, synthesis_task]

    async def run(self, query: str):
        """
        Takes a user query, creates tasks and a Crew, and kicks off the process.

        Args:
            query (str): The user's question.

        Returns:
            str: The final synthesized answer from the crew, or an error message.
        """
        if not self.llm:
            return "Error: LLM not initialized. Cannot process query."
        if not self.document_researcher or not self.insight_synthesizer:
             return "Error: Agents not initialized properly. Cannot process query."

        try:
            # 1. Create tasks specific to this query
            tasks = self._create_tasks(query)

            # 2. Create the crew for this specific run
            rag_crew = Crew(
                agents=[self.document_researcher, self.insight_synthesizer],
                tasks=tasks,
                process=Process.sequential,
                verbose=True,  # Changed from verbose=2 to verbose=True
                #memory=True # Optional: Enable memory if needed across runs
            )

            # 3. Kick off the crew execution
            logger.info(f"Kicking off RAG crew for query: '{query}'")
            result = await rag_crew.kickoff_async()
            return result

        except Exception as e:
            logger.error(f"An error occurred during crew execution for query '{query}': {e}")
           
            return f"Sorry, an error occurred while processing your request: {e}"


    # If you want the class instance to be callable directly like a function:
    async def __call__(self, query: str):
        """Allows calling the instance like `rag_agent_instance(query)`."""
        
        return await self.run(query)
