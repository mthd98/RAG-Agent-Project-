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

google_models = ["gemini/gemini-2.5-flash-lite","gemini/gemini-2.5-flash"]

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
        resolved_model_name = model_name or os.getenv("LLM_MODEL", "ollama/qwen3:1.7b")
        resolved_base_url = base_url or settings.OLLAMA_URL
        
        logger.info(f"Initializing RAGAgent with LLM: model='{resolved_model_name}', base_url='{resolved_base_url}'")
        try:
            if resolved_model_name not in google_models:  
                self.llm = LLM(
                    model=resolved_model_name,
                    base_url=resolved_base_url,
                    temperature=0.2, # Adjusted for more balanced responses
                    timeout=300,
                    verbose=True,  # Enable verbose logging for debugging
                    max_tokens=40000,  # Use maximum context length available
                    num_ctx=40000,     # Context window size
                    )
                
            else: 
                self.llm = LLM(
                model=resolved_model_name, # A capable model for complex planning tasks
                temperature=0.01, # Moderate creativity for planning
                max_retries=2,
                api_key=os.getenv("GOOGLE_API_KEY"),
                verbose=True,)


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
    def create_agents(self):
        # Get the Prompts From Prompt db
        self.document_researcher_prompt = get_prompts("expert-search-query-formulator")
        self.insight_synthesizer_prompt = get_prompts("insight-synthesizer-analyst")
        self.memory_prompt = get_prompts("memory")
        self.final_answer_prompt = get_prompts("final_answer")
        
        # --- Agent Definitions ---
        # Agent 1 : Chat Memory Agent 
        self.memory_agent = Agent(
            role=self.memory_prompt['role'],
            goal=self.memory_prompt['goal'],
            backstory=self.memory_prompt['backstory'],
            tools=[],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )
    
        # Agent 2: Document Researcher
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

        self.final_answer = Agent(
            role=self.final_answer_prompt['role'],
            goal=self.final_answer_prompt['goal'],
            backstory=self.final_answer_prompt['backstory'],
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
            tools=[] # No external tools needed for synthesis.
        )

    def _create_tasks(self,messages, query: str):
        """
        Creates the research and synthesis tasks for a given query.
        Internal helper method.
        """
        if not self.document_researcher or not self.insight_synthesizer or not self.memory_agent:
            raise RuntimeError("Agents not initialized properly (LLM might have failed).")
        

        # Task for the memory
        memory_task = Task(
            description=self.memory_prompt["task"]['description'].format(history=messages,query=query),
            expected_output=self.memory_prompt["task"]['expected_output'],
            agent=self.memory_agent,
        )

        # Task for the Document Researcher
        research_task = Task(
            description=self.document_researcher_prompt["task"]['description'].format(query=query),
            expected_output=self.document_researcher_prompt["task"]['expected_output'],
            agent=self.document_researcher,
            context=[memory_task]
        )

        # Task for the Insight Synthesizer
        synthesis_task = Task(
            description=self.insight_synthesizer_prompt["task"]['description'].format(query=query),
            expected_output=self.insight_synthesizer_prompt["task"]['expected_output'],
            agent=self.insight_synthesizer,
            context=[research_task,memory_task], # Explicit dependency
        )

        final_answer_task = Task(
            description=self.final_answer_prompt["task"]['description'],
            expected_output=self.final_answer_prompt["task"]['expected_output'],
            agent=self.insight_synthesizer,
            context=[synthesis_task], # Explicit dependency
        )

        return [memory_task,research_task, synthesis_task,final_answer_task]

    async def run(self,messages, query: str):
        """
        Takes a user query, creates tasks and a Crew, and kicks off the process.

        Args:
            query (str): The user's question.

        Returns:
            str: The final synthesized answer from the crew, or an error message.
        """
        # 1. Create the agents 
        self.create_agents()
        
        
        try : 
        # 2. Create tasks specific to this query
        
            tasks = self._create_tasks(messages,query)

            # 3. Create the crew for this specific run
            rag_crew = Crew(
                agents=[self.memory_agent,self.document_researcher, self.insight_synthesizer,self.final_answer],
                tasks=tasks,
                process=Process.sequential,
                verbose=True,  # Changed from verbose=2 to verbose=True
                #memory=True # Optional: Enable memory if needed across runs
            )

            # 4. Kick off the crew execution
            logger.info(f"Kicking off RAG crew for query: '{query}'")
            inputs = {
                "query":query,
                "history":messages
            }
            result = await rag_crew.kickoff_async(inputs=inputs)
            return result

        except Exception as e:
            logger.error(f"An error occurred during crew execution for query '{query}': {e}")
           
            return f"Sorry, an error occurred while processing your request: {e}"


    async def __call__(self, messages: list,query:str):
        """Allows calling the instance like `rag_agent_instance(query)`."""
        
        return await self.run(messages,query)
