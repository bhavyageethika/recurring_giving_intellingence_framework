"""
Chat Orchestrator Agent

Routes user messages to appropriate agents using dynamic tool calling and A2A protocol.
Integrates with MCP for external service access.
"""

from typing import Dict, Any, List, Optional
import asyncio
import inspect
import structlog

from src.core.autonomous_agent import AutonomousAgent, Tool
from src.core.a2a_protocol import get_a2a_protocol
from src.core.mcp_server import get_mcp_server
from src.core.llm_service import get_llm_service

logger = structlog.get_logger()


class ChatOrchestrator(AutonomousAgent[Dict[str, Any]]):
    """
    Chat orchestrator that routes messages to appropriate agents using A2A protocol.
    Uses dynamic tool calling to determine which agent to use.
    """
    
    def __init__(self):
        super().__init__(
            agent_id="chat_orchestrator",
            name="Chat Orchestrator",
            description="Orchestrates chat interactions by routing messages to specialized agents and MCP tools.",
            system_prompt="You are a helpful chat orchestrator that routes user requests to the most appropriate agents or tools. Always prioritize using tools when a clear intent is present.",
        )
        try:
            self._a2a = get_a2a_protocol()
            self._mcp = get_mcp_server()
            self._logger = logger.bind(agent="chat_orchestrator")
            
            # Register domain-specific tools (required by AutonomousAgent)
            for tool in self._get_domain_tools():
                self.register_tool(tool)
            
            self._logger.info("ChatOrchestrator initialized successfully", tools_count=len(self._tools))
        except Exception as e:
            self._logger.error("Failed to initialize ChatOrchestrator", error=str(e), exc_info=True)
            raise
    
    def _get_domain_system_prompt(self) -> str:
        """Get domain-specific system prompt additions."""
        return """
## Domain Expertise: Chat Orchestration & Routing

You specialize in:
1. **Intent Recognition**: Understanding user requests and determining the best agent or tool to handle them
2. **Dynamic Tool Calling**: Using LLM reasoning to select appropriate tools based on context
3. **Agent Coordination**: Routing requests to specialized agents via A2A protocol
4. **MCP Integration**: Accessing external services (email, contacts, calendar) through MCP tools
5. **Context Management**: Maintaining session context (donations, donor info) across interactions

You are the central orchestrator that ensures user requests are handled by the most appropriate specialized agent or tool.
Always prioritize using tools when a clear intent is present, and route complex requests to specialized agents.
"""
    
    def _get_domain_tools(self) -> List[Tool]:
        """Get domain-specific tools for routing to agents and MCP services."""
        return self._register_routing_tools()
    
    def _register_routing_tools(self) -> List[Tool]:
        """Get tools that route to different agents via A2A."""
        
        # Agent routing tools
        agent_tools = [
            Tool(
                name="analyze_campaign",
                description="Analyze a GoFundMe campaign URL for quality, success potential, tone, and messaging. Use this when user provides a campaign URL or asks about campaign analysis.",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "GoFundMe campaign URL"},
                        "user_message": {"type": "string", "description": "Original user message for context"},
                    },
                    "required": ["url"],
                },
                function=self._route_to_campaign_intelligence,
            ),
            Tool(
                name="build_donor_profile",
                description="Build a donor profile from donation history. Use this when user shares donations or asks to build their profile. Can parse donations from user_message if donations array is empty.",
                parameters={
                    "type": "object",
                    "properties": {
                        "donations": {"type": "array", "items": {"type": "object"}, "description": "List of donations (can be empty if parsing from user_message)"},
                        "donor_info": {"type": "object", "description": "Donor information (name, email, location)"},
                        "user_message": {"type": "string", "description": "Original user message (will be parsed for donations if donations array is empty)"},
                    },
                    "required": ["user_message"],
                },
                function=self._route_to_donor_profiler,
            ),
            Tool(
                name="get_campaign_recommendations",
                description="Get personalized campaign recommendations for a donor. Use this when user asks for recommendations, matches, or suggestions.",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_profile": {"type": "object", "description": "Donor profile data"},
                        "limit": {"type": "integer", "description": "Number of recommendations", "default": 5},
                        "user_message": {"type": "string", "description": "Original user message"},
                    },
                    "required": ["donor_profile"],
                },
                function=self._route_to_campaign_matcher,
            ),
            Tool(
                name="analyze_tone",
                description="Analyze the tone of campaign messaging. Use this when user asks about tone, messaging style, or communication approach.",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_data": {"type": "object", "description": "Campaign data to analyze"},
                        "user_message": {"type": "string", "description": "Original user message"},
                    },
                    "required": ["campaign_data"],
                },
                function=self._route_to_tone_checker,
            ),
            Tool(
                name="create_ab_test",
                description="Create A/B testing plan for campaign messaging. Use this when user asks about testing, variants, or optimization.",
                parameters={
                    "type": "object",
                    "properties": {
                        "campaign_data": {"type": "object", "description": "Campaign data"},
                        "channels": {"type": "array", "items": {"type": "string"}, "description": "Channels to test"},
                        "user_message": {"type": "string", "description": "Original user message"},
                    },
                    "required": ["campaign_data"],
                },
                function=self._route_to_ab_testing,
            ),
            Tool(
                name="discover_communities",
                description="Discover giving communities and circles. Use this when user asks about communities, groups, or social connections.",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_profile": {"type": "object", "description": "Donor profile"},
                        "location": {"type": "string", "description": "Location for community discovery"},
                        "user_message": {"type": "string", "description": "Original user message"},
                    },
                    "required": ["donor_profile"],
                },
                function=self._route_to_community_discovery,
            ),
            Tool(
                name="find_recurring_opportunities",
                description="Find recurring giving opportunities. Use this when user asks about recurring donations, subscriptions, or ongoing giving.",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_profile": {"type": "object", "description": "Donor profile"},
                        "user_message": {"type": "string", "description": "Original user message"},
                    },
                    "required": ["donor_profile"],
                },
                function=self._route_to_recurring_curator,
            ),
            Tool(
                name="suggest_giving_circles",
                description="Suggest giving circles for the donor. Use this when user asks about giving circles, group giving, or collaborative giving.",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_profile": {"type": "object", "description": "Donor profile"},
                        "community_context": {"type": "object", "description": "Community context"},
                        "user_message": {"type": "string", "description": "Original user message"},
                    },
                    "required": ["donor_profile"],
                },
                function=self._route_to_giving_circle,
            ),
            Tool(
                name="create_engagement_plan",
                description="Create an engagement plan for a donor. Use this when user asks about engagement, outreach, or communication strategies.",
                parameters={
                    "type": "object",
                    "properties": {
                        "donor_profile": {"type": "object", "description": "Donor profile"},
                        "user_message": {"type": "string", "description": "Original user message"},
                    },
                    "required": ["donor_profile"],
                },
                function=self._route_to_engagement_agent,
            ),
        ]
        
        # MCP tools for external services
        mcp_tools = [
            Tool(
                name="get_contacts",
                description="Get contacts from address book. Use this when user asks about contacts, people, or sending messages to specific people.",
                parameters={
                    "type": "object",
                    "properties": {
                        "filter": {"type": "object", "description": "Filter by name, email, or tags"},
                        "limit": {"type": "integer", "description": "Max contacts to return", "default": 100},
                    },
                },
                function=self._call_mcp_get_contacts,
            ),
            Tool(
                name="send_email",
                description="Send an email. Use this when user wants to send an email or message.",
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient email"},
                        "subject": {"type": "string", "description": "Email subject"},
                        "body": {"type": "string", "description": "Email body"},
                    },
                    "required": ["to", "subject", "body"],
                },
                function=self._call_mcp_send_email,
            ),
        ]
        
        # Return all tools (they will be registered by the base class)
        return agent_tools + mcp_tools
    
    async def _route_to_campaign_intelligence(self, url: str, user_message: str = "", **kwargs) -> Dict[str, Any]:
        """Route to campaign intelligence agent via A2A."""
        try:
            import asyncio
            # Very aggressive timeout - return quick response if it takes too long
            response = await asyncio.wait_for(
                self._a2a.send_message(
                    sender_id=self.agent_id,
                    recipient_id="campaign_intelligence_agent",
                    action="analyze_campaign",
                    payload={"url": url, "user_message": user_message},
                    timeout=15.0,  # Very aggressive - 15 seconds max
                ),
                timeout=18.0  # Overall timeout
            )
            # Handle response - might be AgentMessage or dict
            if hasattr(response, 'payload'):
                return response.payload
            elif isinstance(response, dict):
                return response
            else:
                return {"error": "Unexpected response format"}
        except asyncio.TimeoutError:
            self._logger.warning("Campaign intelligence request timed out, returning quick analysis", url=url[:50])
            # Return a quick basic analysis instead of error
            return {
                "campaign_id": url.split('/')[-1].split('?')[0] if url else "unknown",
                "quality_score": {
                    "overall_score": 0.6,
                    "message": "Quick analysis - full analysis timed out"
                },
                "success_prediction": {
                    "probability": 0.5,
                    "message": "Analysis incomplete - request timed out"
                },
                "similar_campaigns": [],
                "messaging_variants": {},
                "note": "Full analysis timed out. This is a quick response. Please try again for complete analysis."
            }
        except Exception as e:
            self._logger.error("Failed to route to campaign intelligence", error=str(e))
            return {"error": str(e)}
    
    async def _route_to_donor_profiler(self, donations: List[Dict], donor_info: Dict = None, user_message: str = "", **kwargs) -> Dict[str, Any]:
        """Route to donor profiler via A2A."""
        try:
            import asyncio
            response = await asyncio.wait_for(
                self._a2a.send_message(
                    sender_id=self.agent_id,
                    recipient_id="donor_affinity_profiler",
                    action="build_profile",
                    payload={"donations": donations, "donor_info": donor_info or {}, "user_message": user_message},
                    timeout=30.0,  # Reduced from 60
                ),
                timeout=35.0  # Overall timeout
            )
            # Handle response - might be AgentMessage or dict
            if hasattr(response, 'payload'):
                return response.payload
            elif isinstance(response, dict):
                return response
            else:
                return {"error": "Unexpected response format"}
        except asyncio.TimeoutError:
            self._logger.error("Donor profiler request timed out")
            return {"error": "Request timed out. The profile building is taking too long."}
        except Exception as e:
            self._logger.error("Failed to route to donor profiler", error=str(e))
            return {"error": str(e)}
    
    async def _route_to_campaign_matcher(self, donor_profile: Dict, limit: int = 5, user_message: str = "", **kwargs) -> Dict[str, Any]:
        """Route to campaign matcher via A2A."""
        try:
            response = await self._a2a.send_message(
                sender_id=self.agent_id,
                recipient_id="campaign_matching_engine",
                action="match_to_donor",
                payload={"donor_profile": donor_profile, "limit": limit, "user_message": user_message},
                timeout=60.0,
            )
            return response.payload
        except Exception as e:
            self._logger.error("Failed to route to campaign matcher", error=str(e))
            return {"error": str(e)}
    
    async def _route_to_tone_checker(self, campaign_data: Dict, user_message: str = "", **kwargs) -> Dict[str, Any]:
        """Route to tone checker via A2A."""
        try:
            response = await self._a2a.send_message(
                sender_id=self.agent_id,
                recipient_id="tone_checker_agent",
                action="analyze_tone",
                payload={"campaign_data": campaign_data, "user_message": user_message},
                timeout=30.0,
            )
            return response.payload
        except Exception as e:
            self._logger.error("Failed to route to tone checker", error=str(e))
            return {"error": str(e)}
    
    async def _route_to_ab_testing(self, campaign_data: Dict, channels: List[str] = None, user_message: str = "", **kwargs) -> Dict[str, Any]:
        """Route to A/B testing agent via A2A."""
        try:
            response = await self._a2a.send_message(
                sender_id=self.agent_id,
                recipient_id="ab_testing_agent",
                action="generate_test_plan",
                payload={"campaign_data": campaign_data, "channels": channels or ["Email"], "user_message": user_message},
                timeout=30.0,
            )
            return response.payload
        except Exception as e:
            self._logger.error("Failed to route to A/B testing", error=str(e))
            return {"error": str(e)}
    
    async def _route_to_community_discovery(self, donor_profile: Dict, location: str = "", user_message: str = "", **kwargs) -> Dict[str, Any]:
        """Route to community discovery via A2A."""
        try:
            response = await self._a2a.send_message(
                sender_id=self.agent_id,
                recipient_id="community_discovery",
                action="discover_communities",
                payload={"donor_profile": donor_profile, "location": location, "user_message": user_message},
                timeout=60.0,
            )
            return response.payload
        except Exception as e:
            self._logger.error("Failed to route to community discovery", error=str(e))
            return {"error": str(e)}
    
    async def _route_to_recurring_curator(self, donor_profile: Dict, user_message: str = "", **kwargs) -> Dict[str, Any]:
        """Route to recurring curator via A2A."""
        try:
            response = await self._a2a.send_message(
                sender_id=self.agent_id,
                recipient_id="recurring_curator",
                action="curate_opportunities",
                payload={"donor_profile": donor_profile, "user_message": user_message},
                timeout=60.0,
            )
            return response.payload
        except Exception as e:
            self._logger.error("Failed to route to recurring curator", error=str(e))
            return {"error": str(e)}
    
    async def _route_to_giving_circle(self, donor_profile: Dict, community_context: Dict = None, user_message: str = "", **kwargs) -> Dict[str, Any]:
        """Route to giving circle orchestrator via A2A."""
        try:
            response = await self._a2a.send_message(
                sender_id=self.agent_id,
                recipient_id="giving_circle_orchestrator",
                action="orchestrate_circle",
                payload={"donor_profile": donor_profile, "community_context": community_context or {}, "user_message": user_message},
                timeout=60.0,
            )
            return response.payload
        except Exception as e:
            self._logger.error("Failed to route to giving circle", error=str(e))
            return {"error": str(e)}
    
    async def _route_to_engagement_agent(self, donor_profile: Dict, user_message: str = "", **kwargs) -> Dict[str, Any]:
        """Route to engagement agent via A2A."""
        try:
            response = await self._a2a.send_message(
                sender_id=self.agent_id,
                recipient_id="engagement_agent",
                action="create_engagement_plan",
                payload={"donor_profile": donor_profile, "user_message": user_message},
                timeout=60.0,
            )
            return response.payload
        except Exception as e:
            self._logger.error("Failed to route to engagement agent", error=str(e))
            return {"error": str(e)}
    
    async def _call_mcp_get_contacts(self, filter: Dict = None, limit: int = 100, **kwargs) -> Dict[str, Any]:
        """Call MCP get_contacts tool."""
        try:
            result = await self._mcp.call_tool("get_contacts", {"filter": filter, "limit": limit})
            return result
        except Exception as e:
            self._logger.error("Failed to call MCP get_contacts", error=str(e))
            return {"error": str(e)}
    
    async def _call_mcp_send_email(self, to: str, subject: str, body: str, **kwargs) -> Dict[str, Any]:
        """Call MCP send_email tool."""
        try:
            result = await self._mcp.call_tool("send_email", {"to": to, "subject": subject, "body": body, **kwargs})
            return result
        except Exception as e:
            self._logger.error("Failed to call MCP send_email", error=str(e))
            return {"error": str(e)}
    
    async def process_chat_message(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a chat message using LLM-powered conversation with tool calling.
        
        Args:
            message: User's message
            context: Additional context (donations, session data, etc.)
            
        Returns:
            Response with agent results
        """
        context = context or {}
        
        # Ensure context has required keys
        if "donations" not in context:
            context["donations"] = []
        if "donor_info" not in context:
            context["donor_info"] = {}
        
        # Check if tools are registered
        if not self._tools:
            self._logger.error("No tools registered in chat orchestrator")
            return {
                "response": "I'm not properly configured. Please check server logs.",
                "error": "No tools registered",
                "tools_used": [],
            }
        
        try:
            # Build conversation context
            context_summary = self._build_context_summary(context)
            
            # Build system prompt for conversational chat
            tools_description = self._build_tools_description()
            system_prompt = f"""You are a helpful AI assistant for GoFundMe's recurring giving intelligence platform. You help donors discover campaigns, build profiles, and make giving decisions.

{tools_description}

## Context
{context_summary}

## Instructions
- Have a natural conversation with the user
- When you need to use a tool, respond with: TOOL_CALL: tool_name(arg1="value1", arg2="value2")
- After tool execution, you'll receive the results to incorporate into your response
- Be conversational, helpful, and friendly
- If the user shares donation history, use build_donor_profile
- If they want to analyze a campaign, use analyze_campaign
- If they ask for recommendations, use get_campaign_recommendations (if profile exists)

Remember: You can respond directly to questions, or use tools when needed. Be natural and helpful."""

            # First, check if we should use tools based on simple heuristics (faster than LLM)
            import re
            tools_to_call = []
            
            # Quick heuristic-based tool detection
            message_lower = message.lower()
            
            # Check for campaign URLs (more robust pattern)
            url_pattern = r'https?://[^\s\)"\']+'  # Match URLs even if followed by closing paren or quotes
            url_match = re.search(url_pattern, message)
            if url_match:
                url = url_match.group(0).rstrip('.,;!?)"\'')  # Remove trailing punctuation
                tools_to_call.append({
                    "name": "analyze_campaign",
                    "arguments": {"url": url, "user_message": message}
                })
                self._logger.info("Detected campaign URL via heuristic", url=url[:50])
            
            # Check for donation mentions - parse from message if user is sharing donations
            if any(kw in message_lower for kw in ["donated", "donation", "gave", "$", "i gave", "i donated"]):
                # Use existing donations from context, or let the tool parse from message
                donations_to_use = context.get("donations", [])
                # If no donations in context, the build_donor_profile tool will parse from user_message
                tools_to_call.append({
                    "name": "build_donor_profile",
                    "arguments": {
                        "donations": donations_to_use,
                        "donor_info": context.get("donor_info", {}),
                        "user_message": message
                    }
                })
            
            # Check for recommendation requests
            if any(kw in message_lower for kw in ["recommend", "suggest", "match", "show me"]) and context.get("donor_profile"):
                tools_to_call.append({
                    "name": "get_campaign_recommendations",
                    "arguments": {
                        "donor_profile": context.get("donor_profile"),
                        "limit": 5,
                        "user_message": message
                    }
                })
            
            # Execute tools if detected via heuristics (fast path)
            tools_used = []
            tool_results = {}
            
            if tools_to_call:
                self._logger.info("Executing tools from heuristics", tools=[tc["name"] for tc in tools_to_call])
                for tool_call in tools_to_call:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["arguments"]
                    
                    if tool_name in self._tools:
                        try:
                            tool = self._tools[tool_name]
                            filtered_args = self._filter_tool_arguments(tool, tool_args)
                            
                            self._logger.info("Executing tool", tool=tool_name, args=list(filtered_args.keys()))
                            
                            # Execute tool with timeout
                            import asyncio
                            result = await asyncio.wait_for(
                                tool.function(**filtered_args),
                                timeout=25.0  # Tool execution timeout
                            )
                            tool_results[tool_name] = result
                            tools_used.append(tool_name)
                            
                        except asyncio.TimeoutError:
                            self._logger.error("Tool execution timed out", tool=tool_name)
                            tool_results[tool_name] = {"error": "Tool execution timed out"}
                        except Exception as e:
                            self._logger.error("Tool execution failed", tool=tool_name, error=str(e))
                            tool_results[tool_name] = {"error": str(e)}
                
                # If tools were executed via heuristics, synthesize response with LLM (but skip function calling)
                if tool_results:
                    self._logger.info("Tools executed via heuristics, synthesizing response with LLM")
                    results_summary = self._format_tool_results_for_llm(tool_results)
                    synthesis_prompt = f"""User: {message}

Tool execution results:
{results_summary}

Provide a helpful, natural response incorporating the tool results. Be conversational and helpful."""
                    
                    try:
                        import asyncio
                        final_response = await asyncio.wait_for(
                            self._llm._provider.complete(
                                prompt=synthesis_prompt,
                                system_prompt=system_prompt,
                                temperature=0.7,
                                max_tokens=300,
                            ),
                            timeout=15.0
                        )
                        return {
                            "response": final_response.strip() if final_response else self._format_response(tool_results, message),
                            "results": tool_results,
                            "tools_used": tools_used,
                        }
                    except asyncio.TimeoutError:
                        self._logger.warning("Synthesis timed out, using formatted response")
                        return {
                            "response": self._format_response(tool_results, message),
                            "results": tool_results,
                            "tools_used": tools_used,
                        }
                    except Exception as e:
                        self._logger.error("Synthesis failed", error=str(e))
                        return {
                            "response": self._format_response(tool_results, message),
                            "results": tool_results,
                            "tools_used": tools_used,
                        }
            
            # For simple messages without tools, use direct LLM call (faster)
            if not tools_to_call and not tool_results:
                self._logger.info("Simple message, using direct LLM call", message_preview=message[:100])
                try:
                    import asyncio
                    # Check if provider is available
                    if not self._llm._provider:
                        self._logger.error("LLM provider not available")
                        return {
                            "response": "I'm currently unavailable. Please check your LLM API configuration.",
                            "results": {},
                            "tools_used": [],
                        }
                    
                    simple_response = await asyncio.wait_for(
                        self._llm._provider.complete(
                            prompt=message,
                            system_prompt=system_prompt,
                            temperature=0.7,
                            max_tokens=200,  # Reduced for faster response
                        ),
                        timeout=8.0  # Very aggressive timeout for simple messages
                    )
                    
                    if not simple_response or not simple_response.strip():
                        # Empty response - use fallback
                        return {
                            "response": "I'm here to help! You can ask me about campaign analysis, donor profiling, or recommendations. What would you like to know?",
                            "results": {},
                            "tools_used": [],
                        }
                    
                    return {
                        "response": simple_response.strip(),
                        "results": {},
                        "tools_used": [],
                    }
                except asyncio.TimeoutError:
                    self._logger.error("Simple LLM call timed out - LLM provider may be slow or unavailable")
                    return {
                        "response": "I'm experiencing delays connecting to the AI service. This might be due to network issues or API configuration. Please check your LLM API key settings.",
                        "results": {},
                        "tools_used": [],
                    }
                except Exception as e:
                    self._logger.error("Simple LLM call failed", error=str(e), exc_info=True)
                    return {
                        "response": f"I'm having trouble processing your message. Error: {str(e)[:100]}. Please check your LLM API configuration.",
                        "results": {},
                        "tools_used": [],
                    }
            
            # Prepare tools schema for LLM function calling (only if we need tools)
            tools_schema = []
            for tool in self._tools.values():
                # Skip core autonomous agent tools for chat interface
                if tool.name in {"think", "create_plan", "reflect", "complete_task"}:
                    continue
                
                # Convert to appropriate format based on provider
                tool_schema = tool.to_schema()["function"]
                
                # For Anthropic, need to convert to their format
                if hasattr(self._llm._provider, '_get_client') and 'anthropic' in str(type(self._llm._provider)).lower():
                    # Anthropic format: {name, description, input_schema}
                    tools_schema.append({
                        "name": tool_schema["name"],
                        "description": tool_schema["description"],
                        "input_schema": tool_schema["parameters"]
                    })
                else:
                    # OpenAI format: {name, description, parameters}
                    tools_schema.append(tool_schema)
            
            # Use native function calling with LLM
            messages = [{"role": "user", "content": message}]
            
            # Add tool results as assistant messages if any
            if tool_results:
                results_summary = self._format_tool_results_for_llm(tool_results)
                messages.append({
                    "role": "assistant",
                    "content": f"Tool execution results:\n{results_summary}"
                })
            
            self._logger.info("Calling LLM with function calling", message_preview=message[:100], tools_count=len(tools_schema))
            
            # Use asyncio timeout to prevent hanging
            import asyncio
            max_iterations = 2  # Reduced from 3 to prevent long chains
            iteration = 0
            total_timeout = 30.0  # Reduced overall timeout
            start_time = asyncio.get_event_loop().time()
            
            while iteration < max_iterations:
                # Check overall timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > total_timeout:
                    self._logger.warning("Overall timeout reached", elapsed=elapsed)
                    break
                
                iteration += 1
                try:
                    # Check if provider supports complete_with_tools
                    if hasattr(self._llm._provider, 'complete_with_tools') and tools_schema:
                        response = await asyncio.wait_for(
                            self._llm._provider.complete_with_tools(
                                messages=messages,
                                system_prompt=system_prompt,
                                tools=tools_schema,
                                temperature=0.7,
                                max_tokens=200,  # Reduced
                            ),
                            timeout=10.0  # Aggressive timeout
                        )
                        
                        # Handle response format
                        if isinstance(response, dict):
                            llm_content = response.get("content", "")
                            tool_calls = response.get("tool_calls", [])
                        else:
                            llm_content = response
                            tool_calls = []
                    else:
                        # Fallback to regular complete if no tool support
                        llm_content = await asyncio.wait_for(
                            self._llm._provider.complete(
                                prompt=message,
                                system_prompt=system_prompt,
                                temperature=0.7,
                                max_tokens=400,
                            ),
                            timeout=20.0
                        )
                        tool_calls = []
                    
                    # Execute tool calls if any
                    if tool_calls:
                        self._logger.info("LLM requested tool calls", tool_calls=[tc.get("name") for tc in tool_calls])
                        
                        # Add assistant message with tool calls
                        messages.append({
                            "role": "assistant",
                            "content": llm_content,
                            "tool_calls": tool_calls
                        })
                        
                        # Execute each tool call
                        tool_results_batch = {}
                        for tool_call in tool_calls:
                            tool_name = tool_call.get("name")
                            tool_call_id = tool_call.get("id", "")
                            
                            if tool_name in self._tools and tool_name not in tools_used:
                                try:
                                    # Parse arguments (may be string or dict)
                                    tool_args = tool_call.get("arguments", {})
                                    if isinstance(tool_args, str):
                                        import json
                                        tool_args = json.loads(tool_args)
                                    
                                    # Add user_message for routing tools
                                    core_tools = {"think", "create_plan", "reflect", "complete_task"}
                                    if tool_name not in core_tools:
                                        tool_args["user_message"] = message
                                    
                                    tool = self._tools[tool_name]
                                    filtered_args = self._filter_tool_arguments(tool, tool_args)
                                    
                                    self._logger.info("Executing tool from LLM", tool=tool_name, args=list(filtered_args.keys()))
                                    
                                    result = await asyncio.wait_for(
                                        tool.function(**filtered_args),
                                        timeout=25.0  # Tool execution timeout
                                    )
                                    tool_results_batch[tool_name] = result
                                    tool_results[tool_name] = result
                                    tools_used.append(tool_name)
                                    
                                    # Add tool result to messages for next iteration
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": str(result)
                                    })
                                    
                                except asyncio.TimeoutError:
                                    self._logger.error("Tool execution timed out", tool=tool_name)
                                    error_result = {"error": "Tool execution timed out"}
                                    tool_results_batch[tool_name] = error_result
                                    tool_results[tool_name] = error_result
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": str(error_result)
                                    })
                                except Exception as e:
                                    self._logger.error("Tool execution failed", tool=tool_name, error=str(e))
                                    error_result = {"error": str(e)}
                                    tool_results_batch[tool_name] = error_result
                                    tool_results[tool_name] = error_result
                                    messages.append({
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": str(error_result)
                                    })
                        
                        # Continue loop to get LLM response with tool results
                        continue
                    else:
                        # No more tool calls, return final response
                        if not llm_content or not llm_content.strip():
                            if tool_results:
                                llm_content = self._format_response(tool_results, message)
                            else:
                                llm_content = "I'm having trouble processing your message. Please try again."
                        
                        return {
                            "response": llm_content.strip(),
                            "results": tool_results,
                            "tools_used": tools_used,
                        }
                        
                except asyncio.TimeoutError:
                    self._logger.error("LLM call timed out")
                    # Fallback response
                    if tool_results:
                        return {
                            "response": self._format_response(tool_results, message),
                            "results": tool_results,
                            "tools_used": tools_used,
                        }
                    else:
                        return {
                            "response": "I'm taking longer than expected to respond. Please try again with a simpler question.",
                            "results": {},
                            "tools_used": [],
                        }
            
            # Max iterations reached
            if tool_results:
                return {
                    "response": self._format_response(tool_results, message),
                    "results": tool_results,
                    "tools_used": tools_used,
                }
            else:
                return {
                    "response": "I processed your request but reached the maximum number of tool calls. Please try a simpler question.",
                    "results": {},
                    "tools_used": [],
                }
                
        except Exception as e:
            self._logger.error("Chat processing failed", error=str(e), exc_info=True)
            error_msg = str(e)
            if len(error_msg) > 200:
                error_msg = error_msg[:200] + "..."
            return {
                "response": f"I encountered an error: {error_msg}. Please try rephrasing your question.",
                "error": error_msg,
                "results": {},
                "tools_used": [],
            }
    
    def _build_context_summary(self, context: Dict[str, Any]) -> str:
        """Build a summary of the conversation context."""
        parts = []
        
        donations = context.get("donations", [])
        if donations:
            total = sum(d.get("amount", 0) for d in donations)
            parts.append(f"- Donation history: {len(donations)} donations, ${total:,.2f} total")
        
        donor_info = context.get("donor_info", {})
        if donor_info:
            name = donor_info.get("name", "Unknown")
            parts.append(f"- Donor: {name}")
        
        if not parts:
            parts.append("- No donation history or profile yet")
        
        return "\n".join(parts) if parts else "No context available"
    
    def _build_tools_description(self) -> str:
        """Build a description of available tools for the LLM."""
        tool_descriptions = []
        for name, tool in self._tools.items():
            # Skip core autonomous agent tools for chat interface
            if name in {"think", "create_plan", "reflect", "complete_task"}:
                continue
            tool_descriptions.append(f"- {name}: {tool.description}")
        
        return "## Available Tools\n\n" + "\n".join(tool_descriptions) if tool_descriptions else "## Available Tools\n\nNone"
    
    def _parse_tool_args(self, args_str: str) -> Dict[str, Any]:
        """Parse tool arguments from string like 'arg1="value1", arg2="value2"'."""
        import re
        args = {}
        
        # Match key="value" or key='value' patterns
        pattern = r'(\w+)=["\']([^"\']*)["\']'
        matches = re.findall(pattern, args_str)
        
        for key, value in matches:
            # Normalize common parameter name variations
            if key == "campaign_url":
                key = "url"  # Map campaign_url to url
            elif key == "donor_id":
                key = "donor_id"  # Keep as is
            
            # Try to parse as JSON if it looks like JSON
            try:
                import json
                args[key] = json.loads(value)
            except:
                args[key] = value
        
        return args
    
    def _format_tool_results_for_llm(self, results: Dict[str, Any]) -> str:
        """Format tool results for LLM consumption."""
        formatted = []
        for tool_name, result in results.items():
            if "error" in result:
                formatted.append(f"{tool_name}: Error - {result['error']}")
            else:
                # Convert result to readable string
                import json
                result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
                formatted.append(f"{tool_name}:\n{result_str}")
        
        return "\n\n".join(formatted)
    
    def _filter_tool_arguments(self, tool: Tool, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter tool arguments to only include parameters defined in the tool's schema or function signature.
        
        Args:
            tool: The Tool object with its parameter schema
            args: Raw arguments from LLM
            
        Returns:
            Filtered arguments containing only valid parameters
        """
        if not args:
            return {}
        
        # Get allowed parameters from function signature (most reliable)
        try:
            sig = inspect.signature(tool.function)
            func_params = set(sig.parameters.keys())
        except Exception as e:
            self._logger.warning("Could not inspect function signature", tool=tool.name, error=str(e))
            func_params = set()
        
        # Also get allowed parameters from schema (fallback)
        schema_params = set()
        if tool.parameters and isinstance(tool.parameters, dict):
            properties = tool.parameters.get("properties", {})
            if isinstance(properties, dict):
                schema_params = set(properties.keys())
        
        # Use function signature params if available, otherwise use schema params
        if func_params:
            allowed_params = func_params
        elif schema_params:
            allowed_params = schema_params
        else:
            self._logger.warning("Could not determine allowed parameters", tool=tool.name)
            return {}
        
        # Filter to only include allowed parameters
        filtered = {k: v for k, v in args.items() if k in allowed_params}
        
        # Log if any arguments were filtered out
        filtered_out = set(args.keys()) - allowed_params
        if filtered_out:
            self._logger.warning(
                "Filtered out invalid tool arguments",
                tool=tool.name,
                filtered_params=list(filtered_out),
                allowed_params=list(allowed_params),
                original_args=list(args.keys()),
            )
        
        return filtered
    
    def _fallback_routing(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback routing using keyword matching."""
        message_lower = message.lower()
        tools = []
        
        if any(kw in message_lower for kw in ["campaign", "gofundme", "analyze", "url"]):
            # Extract URL
            import re
            url_match = re.search(r'https?://[^\s]+', message)
            if url_match:
                tools.append({
                    "name": "analyze_campaign",
                    "arguments": {"url": url_match.group(0)},
                })
        
        if any(kw in message_lower for kw in ["donated", "donation", "gave", "$"]):
            donations = context.get("donations", [])
            if donations:
                tools.append({
                    "name": "build_donor_profile",
                    "arguments": {"donations": donations, "donor_info": context.get("donor_info", {})},
                })
        
        if any(kw in message_lower for kw in ["recommend", "suggest", "match"]):
            donor_profile = context.get("donor_profile")
            if donor_profile:
                tools.append({
                    "name": "get_campaign_recommendations",
                    "arguments": {"donor_profile": donor_profile},
                })
        
        return {"tools": tools}
    
    def _format_response(self, results: Dict[str, Any], original_message: str) -> str:
        """Format agent results into a user-friendly response."""
        response_parts = []
        
        for tool_name, result in results.items():
            if "error" in result:
                response_parts.append(f"❌ {tool_name} encountered an error: {result['error']}")
            else:
                # Format based on tool type
                if tool_name == "analyze_campaign":
                    response_parts.append("✅ Campaign analysis complete! Check the results above.")
                elif tool_name == "build_donor_profile":
                    total = result.get("total_lifetime_giving", 0)
                    response_parts.append(f"✅ Profile built! Total giving: ${total:,.2f}")
                elif tool_name == "get_campaign_recommendations":
                    recs = result.get("recommendations", [])
                    response_parts.append(f"✅ Found {len(recs)} recommendations!")
                else:
                    response_parts.append(f"✅ {tool_name} completed successfully.")
        
        return "\n".join(response_parts) if response_parts else "Task completed."

