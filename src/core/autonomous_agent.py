"""
Autonomous Agent Framework

Provides LLM-based autonomous agents with:
- Planning and reasoning capabilities
- Tool use and function calling
- Long-term task decomposition
- Memory and context management
- Multi-step goal pursuit
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from uuid import uuid4

import structlog

from src.core.llm_service import LLMService, get_llm_service
from src.core.vector_memory import get_vector_memory
from src.core.graph_memory import get_graph_memory, RelationshipType

logger = structlog.get_logger()


class TaskStatus(str, Enum):
    """Status of a task in the agent's plan."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class ReasoningType(str, Enum):
    """Types of reasoning the agent can perform."""
    ANALYSIS = "analysis"
    PLANNING = "planning"
    DECISION = "decision"
    REFLECTION = "reflection"
    SYNTHESIS = "synthesis"


@dataclass
class Tool:
    """A tool that an agent can use."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    requires_confirmation: bool = False
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


@dataclass
class Task:
    """A task in the agent's execution plan."""
    task_id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    
    # Task hierarchy
    parent_task_id: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    
    # Execution
    tool_to_use: Optional[str] = None
    tool_arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Reasoning
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "parent_task_id": self.parent_task_id,
            "subtasks": self.subtasks,
            "depends_on": self.depends_on,
            "tool_to_use": self.tool_to_use,
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "reasoning": self.reasoning,
        }


@dataclass
class ReasoningStep:
    """A step in the agent's reasoning process."""
    step_id: str = field(default_factory=lambda: str(uuid4()))
    reasoning_type: ReasoningType = ReasoningType.ANALYSIS
    thought: str = ""
    conclusion: str = ""
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentMemory:
    """Long-term and working memory for the agent with vector and graph integration."""
    
    # Working memory (current context)
    current_goal: str = ""
    current_context: Dict[str, Any] = field(default_factory=dict)
    
    # Task memory
    tasks: Dict[str, Task] = field(default_factory=dict)
    task_order: List[str] = field(default_factory=list)
    
    # Reasoning trace
    reasoning_history: List[ReasoningStep] = field(default_factory=list)
    
    # Episodic memory (past interactions)
    past_interactions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Semantic memory (learned facts)
    learned_facts: Dict[str, Any] = field(default_factory=dict)
    
    # Performance tracking
    successful_actions: int = 0
    failed_actions: int = 0
    
    # Vector and graph memory references
    agent_id: str = ""
    _vector_store = None
    _graph_store = None
    
    def add_task(self, task: Task) -> None:
        """Add a task to memory."""
        self.tasks[task.task_id] = task
        if task.task_id not in self.task_order:
            self.task_order.append(task.task_id)
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks in order."""
        return [
            self.tasks[tid] for tid in self.task_order
            if tid in self.tasks and self.tasks[tid].status == TaskStatus.PENDING
        ]
    
    def get_executable_tasks(self) -> List[Task]:
        """Get tasks that can be executed (dependencies met)."""
        executable = []
        for task in self.get_pending_tasks():
            deps_met = all(
                self.tasks.get(dep_id, Task()).status == TaskStatus.COMPLETED
                for dep_id in task.depends_on
            )
            if deps_met:
                executable.append(task)
        return executable
    
    def add_reasoning(self, reasoning: ReasoningStep) -> None:
        """Add a reasoning step to history."""
        self.reasoning_history.append(reasoning)
        # Keep last 50 reasoning steps
        if len(self.reasoning_history) > 50:
            self.reasoning_history = self.reasoning_history[-50:]
    
    async def get_context_summary(self) -> str:
        """Get a summary of current context for LLM, including vector and graph memories."""
        recent_reasoning = self.reasoning_history[-5:] if self.reasoning_history else []
        pending_tasks = self.get_pending_tasks()[:5]
        
        # Get relevant vector memories (skip if no memories yet to avoid slow embedding generation)
        vector_context = []
        if self.agent_id and self._vector_store and len(self._vector_store._vectors) > 0:
            try:
                # Use timeout to avoid blocking too long
                import asyncio
                vector_results = await asyncio.wait_for(
                    self._vector_store.get_context(
                        agent_id=self.agent_id,
                        current_goal=self.current_goal,
                        limit=3,
                    ),
                    timeout=0.5  # Max 500ms for context retrieval
                )
                vector_context = vector_results
            except (asyncio.TimeoutError, Exception) as e:
                # Silently skip if retrieval is slow or fails
                pass
        
        # Get causal relationships from graph
        graph_insights = []
        if self.agent_id and self._graph_store:
            try:
                # Find similar past outcomes
                if self.current_context.get("task_id"):
                    task_node_id = f"{self.agent_id}_task_{self.current_context['task_id']}"
                    if self._graph_store.node_exists(task_node_id):
                        successors = self._graph_store.get_causal_successors(
                            task_node_id,
                            relationship_type=RelationshipType.LEADS_TO,
                            max_depth=2,
                        )
                        if successors:
                            graph_insights.append(f"Similar tasks led to: {len(successors)} outcomes")
            except Exception as e:
                logger.warning("Failed to retrieve graph context", error=str(e))
        
        context_parts = [
            f"Current Goal: {self.current_goal}",
            "",
            "Recent Reasoning:",
            *[f'- [{r.reasoning_type.value}] {r.thought[:100]}...' for r in recent_reasoning],
            "",
            f"Pending Tasks ({len(pending_tasks)}):",
            *[f'- {t.description}' for t in pending_tasks],
        ]
        
        if vector_context:
            context_parts.extend([
                "",
                "Relevant Past Experiences:",
                *[f'- {ctx[:150]}...' for ctx in vector_context],
            ])
        
        if graph_insights:
            context_parts.extend([
                "",
                "Causal Insights:",
                *[f'- {insight}' for insight in graph_insights],
            ])
        
        context_parts.extend([
            "",
            f"Context: {json.dumps(self.current_context, default=str)[:500]}",
        ])
        
        return "\n".join(context_parts)


T = TypeVar("T")


class AutonomousAgent(ABC, Generic[T]):
    """
    Base class for LLM-based autonomous agents.
    
    Provides:
    - Goal-directed planning
    - Multi-step reasoning (Chain of Thought)
    - Tool use and function calling
    - Task decomposition and execution
    - Memory management
    - Self-reflection and learning
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        system_prompt: str,
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        
        # LLM service
        self._llm: LLMService = get_llm_service()
        
        # Tools registry
        self._tools: Dict[str, Tool] = {}
        
        # Memory
        self._memory = AgentMemory(agent_id=agent_id)
        self._memory._vector_store = get_vector_memory()
        self._memory._graph_store = get_graph_memory()
        
        # Logger
        self._logger = logger.bind(agent_id=agent_id, agent_name=name)
        
        # A2A message handlers (for agent-to-agent communication)
        self._message_handlers: Dict[str, callable] = {}
        
        # Register core tools
        self._register_core_tools()
    
    def _register_core_tools(self) -> None:
        """Register tools available to all agents."""
        self.register_tool(Tool(
            name="think",
            description="Record a thought or reasoning step. Use this to think through problems step by step.",
            parameters={
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "The thought or reasoning"},
                    "reasoning_type": {
                        "type": "string",
                        "enum": ["analysis", "planning", "decision", "reflection", "synthesis"],
                        "description": "Type of reasoning (default: analysis)"
                    },
                },
                "required": ["thought"],
            },
            function=self._tool_think,
        ))
        
        self.register_tool(Tool(
            name="create_plan",
            description="Create a plan with multiple tasks to achieve a goal. Break down complex goals into smaller steps.",
            parameters={
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "The goal to achieve"},
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "depends_on": {"type": "array", "items": {"type": "integer"}},
                                "tool": {"type": "string"},
                            },
                            "required": ["description"],
                        },
                        "description": "List of tasks to accomplish the goal"
                    },
                },
                "required": ["goal", "tasks"],
            },
            function=self._tool_create_plan,
        ))
        
        self.register_tool(Tool(
            name="reflect",
            description="Reflect on progress and adjust strategy if needed.",
            parameters={
                "type": "object",
                "properties": {
                    "observation": {"type": "string", "description": "What has been observed"},
                    "assessment": {"type": "string", "description": "Assessment of progress"},
                    "adjustment": {"type": "string", "description": "Any adjustments to make"},
                },
                "required": ["observation", "assessment"],
            },
            function=self._tool_reflect,
        ))
        
        self.register_tool(Tool(
            name="complete_task",
            description="Mark the current task as complete with a result.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "ID of task to complete"},
                    "result": {"type": "string", "description": "Result or output of the task"},
                    "success": {"type": "boolean", "description": "Whether task succeeded"},
                },
                "required": ["task_id", "result", "success"],
            },
            function=self._tool_complete_task,
        ))
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool for the agent to use."""
        self._tools[tool.name] = tool
        self._logger.info("tool_registered", tool_name=tool.name)
    
    async def _tool_think(self, thought: str, reasoning_type: str = "analysis") -> Dict[str, Any]:
        """Record a thought."""
        try:
            rt = ReasoningType(reasoning_type)
        except (ValueError, KeyError):
            rt = ReasoningType.ANALYSIS
        
        step = ReasoningStep(
            reasoning_type=rt,
            thought=thought,
        )
        self._memory.add_reasoning(step)
        
        # Store reasoning in vector memory (non-blocking, only for important reasoning)
        # Skip for frequent/trivial reasoning to avoid performance impact
        if self._memory._vector_store and reasoning_type in ["decision", "synthesis", "reflection"]:
            # Fire and forget - don't wait for embedding generation
            # Use fast hash-based embedding for reasoning (no API call)
            try:
                import asyncio
                asyncio.create_task(
                    self._memory._vector_store.store(
                        agent_id=self.agent_id,
                        content=thought,
                        memory_type="reasoning",
                        metadata={"reasoning_type": reasoning_type},
                        use_fast_embedding=True,  # Fast hash-based for reasoning
                    )
                )
            except Exception as e:
                self._logger.debug("Background memory storage failed", error=str(e))
        
        return {"status": "recorded", "thought": thought}
    
    async def _tool_create_plan(self, goal: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an execution plan."""
        self._memory.current_goal = goal
        created_tasks = []
        task_id_map = {}  # Map index to task_id
        
        for i, task_def in enumerate(tasks):
            task = Task(
                description=task_def["description"],
                tool_to_use=task_def.get("tool"),
            )
            task_id_map[i] = task.task_id
            created_tasks.append(task)
        
        # Set up dependencies
        for i, task_def in enumerate(tasks):
            deps = task_def.get("depends_on", [])
            created_tasks[i].depends_on = [task_id_map[d] for d in deps if d in task_id_map]
        
        # Add to memory
        for task in created_tasks:
            self._memory.add_task(task)
        
        return {
            "status": "plan_created",
            "goal": goal,
            "task_count": len(created_tasks),
            "tasks": [t.to_dict() for t in created_tasks],
        }
    
    async def _tool_reflect(
        self,
        observation: str,
        assessment: str,
        adjustment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reflect on progress."""
        step = ReasoningStep(
            reasoning_type=ReasoningType.REFLECTION,
            thought=f"Observation: {observation}\nAssessment: {assessment}",
            conclusion=adjustment or "No adjustment needed",
        )
        self._memory.add_reasoning(step)
        
        return {
            "status": "reflected",
            "observation": observation,
            "assessment": assessment,
            "adjustment": adjustment,
        }
    
    async def _tool_complete_task(
        self,
        task_id: str,
        result: str,
        success: bool
    ) -> Dict[str, Any]:
        """Mark a task as complete."""
        if task_id not in self._memory.tasks:
            return {"status": "error", "message": f"Task {task_id} not found"}
        
        task = self._memory.tasks[task_id]
        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        task.result = result
        task.completed_at = datetime.utcnow()
        
        if success:
            self._memory.successful_actions += 1
        else:
            self._memory.failed_actions += 1
        
        return {
            "status": "completed" if success else "failed",
            "task_id": task_id,
            "result": result,
        }
    
    @abstractmethod
    def _get_domain_tools(self) -> List[Tool]:
        """Get domain-specific tools for this agent. Override in subclasses."""
        pass
    
    @abstractmethod
    def _get_domain_system_prompt(self) -> str:
        """Get domain-specific system prompt additions. Override in subclasses."""
        pass
    
    def _build_system_prompt(self) -> str:
        """Build the complete system prompt."""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self._tools.items()
        ])
        
        return f"""{self.system_prompt}

{self._get_domain_system_prompt()}

## Your Capabilities

You are an autonomous agent that can:
1. **Plan**: Break down complex goals into manageable tasks
2. **Reason**: Think step-by-step using chain-of-thought reasoning
3. **Use Tools**: Execute actions using available tools
4. **Learn**: Reflect on outcomes and adjust strategies
5. **Remember**: Maintain context across interactions

## Available Tools

{tools_desc}

## Operating Principles

1. Always think before acting - use the 'think' tool to reason through problems
2. Create plans for complex tasks - break them into smaller steps
3. Reflect on progress - assess what's working and adjust
4. Be thorough but efficient - complete tasks fully before moving on
5. Handle errors gracefully - if something fails, reason about why and try alternatives

## Response Format

When given a task, you should:
1. First, think about the task and what's needed
2. Create a plan if the task is complex
3. Execute the plan step by step using tools
4. Reflect on the results
5. Provide a final response

Always explain your reasoning and be transparent about your thought process.
"""
    
    async def reason(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Perform reasoning about a query.
        
        Uses Chain-of-Thought prompting to think through the problem.
        """
        self._memory.current_context = context or {}
        
        context_summary = await self._memory.get_context_summary()
        
        prompt = f"""Query: {query}

Context:
{context_summary}

Think through this step by step:
1. What is being asked?
2. What information do I have?
3. What do I need to figure out?
4. What's my reasoning?
5. What's my conclusion?

Provide your reasoning in a structured way."""

        response = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self._build_system_prompt(),
            temperature=0.3,
        )
        
        # Record the reasoning
        self._memory.add_reasoning(ReasoningStep(
            reasoning_type=ReasoningType.ANALYSIS,
            thought=f"Query: {query}",
            conclusion=response[:500],
        ))
        
        return response
    
    async def plan(self, goal: str, context: Optional[Dict[str, Any]] = None) -> List[Task]:
        """
        Create a plan to achieve a goal.
        
        Decomposes the goal into subtasks with dependencies.
        """
        self._memory.current_goal = goal
        self._memory.current_context = context or {}
        
        # Store goal in vector memory (non-blocking background task)
        if self._memory._vector_store:
            try:
                import asyncio
                asyncio.create_task(
                    self._memory._vector_store.store(
                        agent_id=self.agent_id,
                        content=f"Goal: {goal}\nContext: {json.dumps(context or {}, default=str)[:500]}",
                        memory_type="episodic",
                        metadata={"goal": goal, **(context or {})},
                    )
                )
            except Exception as e:
                self._logger.debug("Background goal storage failed", error=str(e))
        
        tools_list = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self._tools.items()
        ])
        
        # Build context summary (avoid sending full data)
        context_summary = {}
        if context:
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    context_summary[key] = value
                elif isinstance(value, list):
                    context_summary[key] = f"List with {len(value)} items"
                    if value and isinstance(value[0], dict):
                        # Show sample structure
                        context_summary[f"{key}_sample"] = str(value[0])[:200]
                elif isinstance(value, dict):
                    context_summary[key] = f"Dict with keys: {list(value.keys())[:5]}"
                else:
                    context_summary[key] = str(type(value).__name__)
        
        prompt = f"""Goal: {goal}

Available Tools:
{tools_list}

Context Summary:
{json.dumps(context_summary, indent=2, default=str)[:800]}

Create a detailed step-by-step plan to achieve this goal. 

IMPORTANT: Return ONLY valid JSON, no other text. Use this exact format:
[
  {{"description": "Analyze donation history to extract metrics", "tool": "analyze_donation_history", "depends_on": []}},
  {{"description": "Identify cause affinities from donations", "tool": "identify_cause_affinities", "depends_on": [0]}},
  ...
]

Rules:
- Use tool names exactly as listed above
- Use "null" if no tool is needed for a step
- Dependencies are indices (0-based) of previous tasks
- First task should have empty depends_on: []
- Think about what data each tool needs and order accordingly"""

        response = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self._build_system_prompt(),
            temperature=0.2,
        )
        
        # Parse the plan with robust error handling
        try:
            import re
            
            # Strategy 1: Try to find JSON array in response (most common)
            json_match = re.search(r'\[[\s\S]*?\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                
                # Clean up common JSON issues
                # Remove trailing commas before closing brackets/braces
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                # Remove comments
                json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
                json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                # Fix single quotes to double quotes (basic)
                json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
                json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
                # Remove extra whitespace but preserve structure
                json_str = re.sub(r'\s+', ' ', json_str)
                
                try:
                    tasks_data = json.loads(json_str)
                except json.JSONDecodeError as json_err:
                    # Strategy 2: Try to extract and fix individual task objects
                    self._logger.warning("initial_json_parse_failed", error=str(json_err), json_preview=json_str[:300])
                    
                    # Try to extract task objects individually
                    task_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    task_matches = re.findall(task_pattern, json_str, re.DOTALL)
                    
                    if task_matches:
                        tasks_data = []
                        for match in task_matches:
                            try:
                                # Clean the individual task
                                cleaned = match.strip()
                                cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
                                cleaned = re.sub(r'//.*?$', '', cleaned, flags=re.MULTILINE)
                                task_obj = json.loads(cleaned)
                                if isinstance(task_obj, dict):
                                    tasks_data.append(task_obj)
                            except json.JSONDecodeError:
                                continue
                        
                        if not tasks_data:
                            raise ValueError("Could not parse any valid tasks from response")
                    else:
                        raise ValueError(f"Could not extract tasks: {str(json_err)}")
                
                # Validate tasks_data
                if isinstance(tasks_data, list) and len(tasks_data) > 0:
                    # Ensure all tasks have required fields
                    validated_tasks = []
                    for i, task in enumerate(tasks_data):
                        if isinstance(task, dict):
                            validated_tasks.append({
                                "description": task.get("description", f"Task {i+1}"),
                                "tool": task.get("tool"),
                                "depends_on": task.get("depends_on", []),
                            })
                    
                    if validated_tasks:
                        await self._tool_create_plan(goal, validated_tasks)
                        self._logger.info("plan_created", task_count=len(validated_tasks))
                    else:
                        raise ValueError("No valid tasks in plan")
                else:
                    raise ValueError("Plan is not a valid list")
            else:
                # Strategy 3: Try to find JSON object with "tasks" or "plan" key
                obj_match = re.search(r'\{[\s\S]*?"(?:tasks|plan)"\s*:\s*\[[\s\S]*?\]', response, re.DOTALL)
                if obj_match:
                    obj_str = obj_match.group()
                    obj_str = re.sub(r',(\s*[}\]])', r'\1', obj_str)
                    obj_data = json.loads(obj_str)
                    tasks_data = obj_data.get("tasks") or obj_data.get("plan", [])
                    if isinstance(tasks_data, list) and len(tasks_data) > 0:
                        validated_tasks = []
                        for i, task in enumerate(tasks_data):
                            if isinstance(task, dict):
                                validated_tasks.append({
                                    "description": task.get("description", f"Task {i+1}"),
                                    "tool": task.get("tool"),
                                    "depends_on": task.get("depends_on", []),
                                })
                        if validated_tasks:
                            await self._tool_create_plan(goal, validated_tasks)
                            self._logger.info("plan_created", task_count=len(validated_tasks))
                        else:
                            raise ValueError("No valid tasks in plan")
                    else:
                        raise ValueError("No JSON array found in response")
                else:
                    raise ValueError("No JSON array or object found in response")
        except (json.JSONDecodeError, AttributeError, ValueError) as e:
            self._logger.warning("failed_to_parse_plan", error=str(e), response_preview=response[:200])
            # Create a fallback plan based on goal keywords
            fallback_tasks = self._create_fallback_plan(goal)
            if fallback_tasks:
                await self._tool_create_plan(goal, fallback_tasks)
            else:
                # Last resort: single task
                task = Task(description=goal)
                self._memory.add_task(task)
        
        return list(self._memory.tasks.values())
    
    def _create_fallback_plan(self, goal: str) -> List[Dict[str, Any]]:
        """Create a fallback plan based on goal keywords when LLM plan parsing fails."""
        goal_lower = goal.lower()
        fallback_tasks = []
        
        # Try to infer tasks from goal keywords and available tools
        if "profile" in goal_lower or "donor" in goal_lower:
            if "analyze_donation_history" in self._tools:
                fallback_tasks.append({"description": "Analyze donation history", "tool": "analyze_donation_history", "depends_on": []})
            if "identify_cause_affinities" in self._tools:
                fallback_tasks.append({"description": "Identify cause affinities", "tool": "identify_cause_affinities", "depends_on": [0] if fallback_tasks else []})
            if "detect_giving_motivators" in self._tools:
                dep_idx = len(fallback_tasks) - 1 if fallback_tasks else 0
                fallback_tasks.append({"description": "Detect giving motivators", "tool": "detect_giving_motivators", "depends_on": [dep_idx]})
            if "generate_donor_insights" in self._tools:
                dep_idx = len(fallback_tasks) - 1 if fallback_tasks else 0
                fallback_tasks.append({"description": "Generate donor insights", "tool": "generate_donor_insights", "depends_on": [dep_idx]})
            if "save_profile" in self._tools:
                dep_idx = len(fallback_tasks) - 1 if fallback_tasks else 0
                fallback_tasks.append({"description": "Save donor profile", "tool": "save_profile", "depends_on": [dep_idx]})
        
        elif "analyze" in goal_lower and "campaign" in goal_lower:
            if "analyze_campaign_semantics" in self._tools:
                fallback_tasks.append({"description": "Analyze campaign semantics", "tool": "analyze_campaign_semantics", "depends_on": []})
            if "build_taxonomy" in self._tools:
                fallback_tasks.append({"description": "Build campaign taxonomy", "tool": "build_taxonomy", "depends_on": [0] if fallback_tasks else []})
            if "assess_urgency" in self._tools:
                dep_idx = len(fallback_tasks) - 1 if fallback_tasks else 0
                fallback_tasks.append({"description": "Assess campaign urgency", "tool": "assess_urgency", "depends_on": [dep_idx]})
            if "evaluate_legitimacy" in self._tools:
                dep_idx = len(fallback_tasks) - 1 if fallback_tasks else 0
                fallback_tasks.append({"description": "Evaluate campaign legitimacy", "tool": "evaluate_legitimacy", "depends_on": [dep_idx]})
            if "save_analysis" in self._tools:
                dep_idx = len(fallback_tasks) - 1 if fallback_tasks else 0
                fallback_tasks.append({"description": "Save campaign analysis", "tool": "save_analysis", "depends_on": [dep_idx]})
        
        elif "curate" in goal_lower or "recurring" in goal_lower:
            if "assess_recurring_suitability" in self._tools:
                fallback_tasks.append({"description": "Assess recurring suitability", "tool": "assess_recurring_suitability", "depends_on": []})
            if "calculate_recommended_amount" in self._tools:
                fallback_tasks.append({"description": "Calculate recommended amount", "tool": "calculate_recommended_amount", "depends_on": [0] if fallback_tasks else []})
            if "generate_recurring_pitch" in self._tools:
                dep_idx = len(fallback_tasks) - 1 if fallback_tasks else 0
                fallback_tasks.append({"description": "Generate recurring pitch", "tool": "generate_recurring_pitch", "depends_on": [dep_idx]})
        
        return fallback_tasks
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with given arguments."""
        if tool_name not in self._tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        tool = self._tools[tool_name]
        
        self._logger.info(
            "executing_tool",
            tool_name=tool_name,
            arguments=str(arguments)[:200],
        )
        
        try:
            if asyncio.iscoroutinefunction(tool.function):
                result = await tool.function(**arguments)
            else:
                result = tool.function(**arguments)
            
            self._memory.successful_actions += 1
            return result
            
        except Exception as e:
            self._logger.error("tool_execution_failed", tool=tool_name, error=str(e))
            self._memory.failed_actions += 1
            return {"error": str(e)}
    
    async def execute_plan(self) -> Dict[str, Any]:
        """Execute the current plan step by step."""
        results = []
        max_iterations = 50  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get executable tasks
            executable = self._memory.get_executable_tasks()
            
            if not executable:
                # Check if we're done or blocked
                pending = self._memory.get_pending_tasks()
                if not pending:
                    break  # All done
                else:
                    # Tasks are blocked
                    self._logger.warning("tasks_blocked", pending_count=len(pending))
                    break
            
            # Execute the first executable task
            task = executable[0]
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.utcnow()
            
            self._logger.info("executing_task", task_id=task.task_id, description=task.description)
            
            if task.tool_to_use and task.tool_to_use in self._tools:
                # Execute the specified tool
                result = await self.execute_tool(task.tool_to_use, task.tool_arguments)
                task.result = result
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.utcnow()
            else:
                # Use LLM to determine how to complete the task
                result = await self._execute_task_with_reasoning(task)
                task.result = result
            
            results.append({
                "task_id": task.task_id,
                "description": task.description,
                "result": task.result,
                "status": task.status.value,
            })
        
        return {
            "goal": self._memory.current_goal,
            "tasks_completed": len([t for t in self._memory.tasks.values() if t.status == TaskStatus.COMPLETED]),
            "tasks_failed": len([t for t in self._memory.tasks.values() if t.status == TaskStatus.FAILED]),
            "results": results,
        }
    
    async def _execute_task_with_reasoning(self, task: Task) -> Any:
        """Execute a task using LLM reasoning to determine actions."""
        tools_schema = [tool.to_schema() for tool in self._tools.values()]
        
        context_summary = await self._memory.get_context_summary()
        
        prompt = f"""Task to complete: {task.description}

Current Context:
{context_summary}

Available tools: {[t.name for t in self._tools.values()]}

Decide how to complete this task. You can:
1. Use a tool by responding with: {{"tool": "tool_name", "arguments": {{...}}}}
2. Provide a direct answer if no tool is needed: {{"answer": "your answer"}}
3. Break into subtasks: {{"subtasks": ["task1", "task2", ...]}}

Think step by step about what's needed."""

        response = await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self._build_system_prompt(),
            temperature=0.2,
        )
        
        # Try to parse as JSON action
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                action = json.loads(json_match.group())
                
                if "tool" in action:
                    result = await self.execute_tool(action["tool"], action.get("arguments", {}))
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.utcnow()
                    return result
                elif "answer" in action:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.utcnow()
                    return action["answer"]
                elif "subtasks" in action:
                    # Create subtasks
                    for subtask_desc in action["subtasks"]:
                        subtask = Task(
                            description=subtask_desc,
                            parent_task_id=task.task_id,
                        )
                        task.subtasks.append(subtask.task_id)
                        self._memory.add_task(subtask)
                    return {"subtasks_created": len(action["subtasks"])}
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: treat response as the result
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        return response
    
    async def run(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 20,
    ) -> Dict[str, Any]:
        """
        Run the agent autonomously to achieve a goal.
        
        This is the main entry point for autonomous operation.
        """
        self._logger.info("agent_run_started", goal=goal)
        
        # Step 1: Reason about the goal
        reasoning = await self.reason(goal, context)
        
        # Step 2: Create a plan
        await self.plan(goal, context)
        
        # Step 3: Execute the plan
        execution_result = await self.execute_plan()
        
        # Step 4: Reflect on the outcome
        reflection = await self._tool_reflect(
            observation=f"Completed {execution_result['tasks_completed']} tasks",
            assessment=f"Goal: {goal}. Success rate: {execution_result['tasks_completed']}/{execution_result['tasks_completed'] + execution_result['tasks_failed']}",
            adjustment=None,
        )
        
        # Step 5: Synthesize final response
        final_response = await self._synthesize_response(goal, execution_result)
        
        return {
            "goal": goal,
            "reasoning": reasoning,
            "plan": [t.to_dict() for t in self._memory.tasks.values()],
            "execution": execution_result,
            "reflection": reflection,
            "response": final_response,
        }
    
    async def _synthesize_response(self, goal: str, execution_result: Dict[str, Any]) -> str:
        """Synthesize a final response from execution results."""
        prompt = f"""Goal: {goal}

Execution Results:
{json.dumps(execution_result, indent=2, default=str)[:2000]}

Reasoning History:
{chr(10).join(f'- {r.thought[:100]}' for r in self._memory.reasoning_history[-5:])}

Synthesize a clear, comprehensive response that:
1. Summarizes what was accomplished
2. Provides the key findings or results
3. Notes any issues or limitations
4. Suggests next steps if applicable"""

        return await self._llm._provider.complete(
            prompt=prompt,
            system_prompt=self._build_system_prompt(),
            temperature=0.5,
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "current_goal": self._memory.current_goal,
            "tasks": [t.to_dict() for t in self._memory.tasks.values()],
            "reasoning_steps": len(self._memory.reasoning_history),
            "successful_actions": self._memory.successful_actions,
            "failed_actions": self._memory.failed_actions,
        }
    
    def register_handler(self, action: str, handler: callable) -> None:
        """Register a handler for A2A message actions."""
        self._message_handlers[action] = handler
        self._logger.info("a2a_handler_registered", action=action)
    
    async def process_message(self, message) -> Any:
        """
        Process an incoming A2A message.
        
        This allows AutonomousAgent to work with the A2A protocol.
        Returns an AgentMessage with the response.
        """
        from src.core.base_agent import AgentMessage
        
        action = message.action if hasattr(message, "action") else message.get("action", "")
        payload = message.payload if hasattr(message, "payload") else message.get("payload", {})
        
        # Check if we have a handler for this action
        handler = self._message_handlers.get(action)
        
        if handler:
            try:
                result = await handler(message)
                # Wrap result in AgentMessage if not already
                if isinstance(result, AgentMessage):
                    return result
                else:
                    # Create response message
                    return AgentMessage(
                        sender=self.agent_id,
                        recipient=message.sender if hasattr(message, "sender") else message.get("sender", ""),
                        message_type="response",
                        action=action,
                        payload=result if isinstance(result, dict) else {"result": result},
                        correlation_id=message.id if hasattr(message, "id") else message.get("id"),
                    )
            except Exception as e:
                self._logger.error("Handler error", action=action, error=str(e))
                return AgentMessage(
                    sender=self.agent_id,
                    recipient=message.sender if hasattr(message, "sender") else message.get("sender", ""),
                    message_type="error",
                    action=action,
                    payload={"error": str(e)},
                    correlation_id=message.id if hasattr(message, "id") else message.get("id"),
                )
        else:
            # Fallback: return error
            self._logger.warning("no_handler_for_action", action=action)
            return AgentMessage(
                sender=self.agent_id,
                recipient=message.sender if hasattr(message, "sender") else message.get("sender", ""),
                message_type="error",
                action=action,
                payload={"error": f"No handler registered for action: {action}"},
                correlation_id=message.id if hasattr(message, "id") else message.get("id"),
            )


# Import asyncio for the execute_tool method
import asyncio

