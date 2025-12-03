"""
AG-UI Server for streaming agent thoughts and status updates.

Implements AG-UI protocol for real-time communication between agents and frontend.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import json
import re
import asyncio
from datetime import datetime
from pydantic import BaseModel

from src.core.llm_service import get_llm_service
import structlog

logger = structlog.get_logger()

app = FastAPI(title="AG-UI Server")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections
active_connections: List[WebSocket] = []


# Request models
class CampaignIntelligenceRequest(BaseModel):
    url: str = ""


class DonorJourneyRequest(BaseModel):
    donor_info: Dict = {}
    donations: List[Dict] = []


class ChatRequest(BaseModel):
    message: str


async def broadcast_agent_thought(agent_name: str, thought: str, step: str = "", data: Dict = None):
    """Broadcast agent thought to all connected clients."""
    message = {
        "type": "agent_thought",
        "timestamp": datetime.now().isoformat(),
        "agent": agent_name,
        "step": step,
        "thought": thought,
        "data": data or {},
    }
    
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except:
            disconnected.append(connection)
    
    # Remove disconnected connections
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)


async def broadcast_agent_status(agent_id: str, status: str, current_task: str = "", progress: int = 0):
    """Broadcast agent status update to all connected clients."""
    message = {
        "type": "agent_status",
        "timestamp": datetime.now().isoformat(),
        "agent_id": agent_id,
        "status": status,
        "current_task": current_task,
        "progress": progress,
    }
    
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except:
            disconnected.append(connection)
    
    # Remove disconnected connections
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AG-UI Server is running",
        "endpoints": [
            "/health",
            "/ag-ui/stream",
            "/api/campaign-intelligence",
            "/api/donor-journey",
            "/api/chat"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AG-UI Server"}


@app.get("/api/chat/test")
async def chat_test():
    """Simple test endpoint to verify chat is working."""
    return {
        "response": "Chat endpoint is working! This is a test response.",
        "agent_name": "System",
    }


@app.websocket("/ag-ui/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for AG-UI streaming."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Add timeout to receive to detect dead connections
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                # Handle ping/pong for keepalive
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong", "timestamp": message.get("timestamp")})
                        continue
                except:
                    pass
                # Echo back or process message
                await websocket.send_json({"type": "ack", "message": "received"})
            except asyncio.TimeoutError:
                # Send a ping to check if connection is still alive
                try:
                    await websocket.send_json({"type": "ping", "timestamp": datetime.now().isoformat()})
                except:
                    # Connection is dead, break out
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.post("/api/campaign-intelligence")
async def campaign_intelligence_endpoint(request: CampaignIntelligenceRequest):
    """Endpoint for campaign intelligence analysis."""
    import asyncio
    try:
        # Add overall timeout to prevent hanging requests
        async def run_analysis():
            return await _perform_campaign_analysis(request)
        
        # Run with 4 minute timeout (increased to allow for all operations including autonomous agents)
        result = await asyncio.wait_for(run_analysis(), timeout=240.0)
        return result
    except asyncio.TimeoutError:
        logger.error("Campaign intelligence analysis timed out after 4 minutes")
        await broadcast_agent_thought(
            "System",
            "Analysis timed out after 4 minutes. The request took too long to complete.",
            "error"
        )
        return {
            "error": "Analysis timed out after 4 minutes. Please try again or check server logs.",
            "extracted_campaign_summary": {"title": "Timeout", "description": "Analysis timed out"},
        }
    except Exception as e:
        logger.error("Error in campaign intelligence endpoint", error=str(e), exc_info=True)
        await broadcast_agent_thought(
            "System",
            f"Error: {str(e)[:200]}",
            "error"
        )
        # Return error response instead of raising
        return {
            "error": str(e)[:500],
            "extracted_campaign_summary": {"title": "Error", "description": str(e)[:200]},
        }


async def _perform_campaign_analysis(request: CampaignIntelligenceRequest):
    """Perform the actual campaign analysis."""
    try:
        # Broadcast that we're starting
        await broadcast_agent_thought(
            "Campaign Intelligence Agent",
            "Starting campaign analysis...",
            "start"
        )
        await broadcast_agent_status("campaign_intelligence_agent", "active", "Starting campaign analysis", 0)
        
        from src.agents.campaign_data_agent import CampaignDataAgent
        from src.agents.campaign_intelligence_agent import CampaignIntelligenceAgent
        from src.agents.tone_checker_agent import ToneCheckerAgent
        from src.agents.ab_testing_agent import ABTestingAgent
        
        url = request.url or ""
        
        print(f"Received URL: '{url}'")
        
        await broadcast_agent_thought(
            "Campaign Intelligence Agent",
            f"Received campaign URL: {url if url else '(empty - using demo)'}",
            "processing"
        )
        
        # Process URL and analyze
        data_agent = CampaignDataAgent()
        intelligence_agent = CampaignIntelligenceAgent()
        tone_checker = ToneCheckerAgent()
        ab_testing_agent = ABTestingAgent()
        
        # Process URL - use direct tool call for speed, with autonomous fallback
        # Direct tool call is faster for simple URL extraction
        if url:
            try:
                await broadcast_agent_thought(
                    "Campaign Data Agent",
                    f"Fetching campaign data from URL: {url}",
                    "extracting"
                )
                await broadcast_agent_status("campaign_data_agent", "active", "Fetching campaign data", 20)
                
                # Use direct tool call for speed (autonomous method adds overhead)
                # This is faster for simple URL extraction tasks
                tool_result = await data_agent._tool_process_url(url)
                campaign_data = tool_result.get("campaign", tool_result) if isinstance(tool_result, dict) else {}
                
                # If direct call didn't work, try autonomous method as fallback
                if not campaign_data or not campaign_data.get("title"):
                    await broadcast_agent_thought(
                        "Campaign Data Agent",
                        "Direct extraction incomplete, using autonomous agent method...",
                        "reasoning"
                    )
                    await broadcast_agent_status("campaign_data_agent", "active", "Using autonomous extraction", 40)
                    
                    goal = f"""Extract and enrich campaign data from the GoFundMe URL: {url}

Your task:
1. Use the process_campaign_url tool to fetch and parse the campaign page
2. Validate the extracted data
3. Enrich any missing information using LLM reasoning
4. Return complete, validated campaign data

The URL is: {url}"""
                    
                    context = {
                        "url": url,
                        "source": "url",
                    }
                    
                    result = await data_agent.run(goal, context, max_iterations=3)  # Reduced iterations for speed
                    
                    # Extract from autonomous agent results
                    if isinstance(result, dict) and "execution" in result:
                        execution = result["execution"]
                        if "results" in execution:
                            for task_result in execution["results"]:
                                if isinstance(task_result, dict) and "result" in task_result:
                                    task_data = task_result["result"]
                                    if isinstance(task_data, dict):
                                        if "campaign" in task_data:
                                            campaign_data = task_data["campaign"]
                                            break
                                        elif "enriched_data" in task_data:
                                            campaign_data = task_data["enriched_data"]
                                            break
                                        elif "validated_data" in task_data:
                                            campaign_data = task_data["validated_data"]
                                            break
                    
                    # Also check memory tasks
                    if not campaign_data and hasattr(data_agent, '_memory'):
                        for task in data_agent._memory.tasks.values():
                            if task.status.value == "completed" and task.result:
                                if isinstance(task.result, dict):
                                    if "campaign" in task.result:
                                        campaign_data = task.result["campaign"]
                                        break
                                    elif "enriched_data" in task.result:
                                        campaign_data = task.result["enriched_data"]
                                        break
                
                # Validate we got campaign data
                if not campaign_data or not campaign_data.get("title"):
                    raise ValueError("Failed to extract campaign data from URL")
                
                print(f"Extracted campaign_data: {campaign_data}")
                
                # Show what was extracted
                extracted_title = campaign_data.get('title', 'Unknown')
                goal_amount = campaign_data.get('goal_amount', 0)
                raised_amount = campaign_data.get('raised_amount', 0)
                donors = campaign_data.get('donor_count', 0)
                
                await broadcast_agent_thought(
                    "Campaign Data Agent",
                    f"✓ Successfully extracted campaign: '{extracted_title}' (Goal: ${goal_amount:,.0f}, Raised: ${raised_amount:,.0f}, Donors: {donors})",
                    "complete"
                )
                await broadcast_agent_status("campaign_data_agent", "complete", "Campaign data extracted", 100)
                
            except Exception as e:
                print(f"Error in campaign data acquisition: {e}")
                import traceback
                traceback.print_exc()
                await broadcast_agent_thought(
                    "Campaign Data Agent",
                    f"Error extracting campaign data: {str(e)[:200]}. Using demo campaign...",
                    "error"
                )
                await broadcast_agent_status("campaign_data_agent", "error", "Using demo campaign", 0)
                
                # Final fallback: demo campaign
                campaign_data = {
                    "campaign_id": "demo",
                    "title": "Help Sarah Fight Cancer",
                    "description": "Medical treatment fund",
                    "category": "medical",
                    "goal_amount": 50000.0,
                    "raised_amount": 15000.0,
                    "donor_count": 45,
                }
        else:
            # Demo campaign
            campaign_data = {
                "campaign_id": "demo",
                "title": "Help Sarah Fight Cancer",
                "description": "Medical treatment fund",
                "category": "medical",
                "goal_amount": 50000.0,
                "raised_amount": 15000.0,
                "donor_count": 45,
            }
        
        # Run tone analysis and intelligence in parallel for speed
        await broadcast_agent_thought(
            "System",
            "Starting parallel analysis: Tone checking and campaign intelligence...",
            "analyzing"
        )
        await broadcast_agent_status("tone_checker_agent", "active", "Analyzing campaign tone", 40)
        await broadcast_agent_status("campaign_intelligence_agent", "active", "Analyzing campaign quality and predicting success", 40)
        
        import asyncio
        
        # Run tone and intelligence in parallel
        tone_task = tone_checker.analyze_tone(campaign_data)
        intelligence_task = intelligence_agent.analyze_campaign(campaign_data)
        
        try:
            tone_analysis, intelligence = await asyncio.gather(tone_task, intelligence_task)
            
            await broadcast_agent_thought(
                "Tone Checker Agent",
                f"Tone analysis complete! Overall tone score: {tone_analysis.overall_tone_score:.1%}",
                "complete"
            )
            await broadcast_agent_status("tone_checker_agent", "complete", "Tone analysis complete", 100)
            
            await broadcast_agent_thought(
                "Campaign Intelligence Agent",
                f"Analysis complete! Quality score: {intelligence.quality_score.overall_score:.1%}",
                "complete"
            )
            await broadcast_agent_status("campaign_intelligence_agent", "complete", "Campaign intelligence analysis complete", 100)
        except Exception as e:
            logger.error("Campaign analysis failed", error=str(e), exc_info=True)
            # Handle errors separately for tone and intelligence
            if 'tone_analysis' not in locals():
                # Tone analysis failed, create default
                from src.agents.tone_checker_agent import ToneAnalysis
                tone_analysis = ToneAnalysis(
                    overall_tone_score=0.5,
                    empathy_score=0.5,
                    authenticity_score=0.5,
                    clarity_score=0.5,
                    urgency_appropriateness=0.5,
                    respectful_language_score=0.5,
                    tone_issues=["Tone analysis error occurred"],
                    tone_strengths=[],
                    improvement_suggestions=[],
                    sensitive_phrases=[],
                    recommended_changes=[],
                )
                await broadcast_agent_thought(
                    "Tone Checker Agent",
                    f"Tone analysis encountered an error: {str(e)[:100]}",
                    "error"
                )
                await broadcast_agent_status("tone_checker_agent", "error", f"Error: {str(e)[:50]}", 0)
            
            if 'intelligence' not in locals():
                # Intelligence analysis failed, create default
                from src.agents.campaign_intelligence_agent import CampaignIntelligence, QualityScore, SuccessPrediction
                intelligence = CampaignIntelligence(
                    campaign_id=campaign_data.get("campaign_id", "error"),
                    quality_score=QualityScore(
                        overall_score=0.5,
                        dimension_scores={},
                        strengths=[],
                        weaknesses=["Analysis error occurred"],
                        improvement_suggestions=[],
                        priority_improvements=["Please try again or check the campaign URL"],
                    ),
                    success_prediction=SuccessPrediction(
                        success_probability=0.5,
                        predicted_amount=0,
                        predicted_donors=0,
                        confidence_level="low",
                        key_factors=[],
                        risk_factors=["Analysis error"],
                        reasoning=f"Error during analysis: {str(e)[:200]}",
                    ),
                    similar_campaigns=[],
                    messaging_variants={},
                    timestamp=datetime.now().isoformat(),
                )
                await broadcast_agent_thought(
                    "Campaign Intelligence Agent",
                    f"Analysis encountered an error: {str(e)[:100]}",
                    "error"
                )
                await broadcast_agent_status("campaign_intelligence_agent", "error", f"Error: {str(e)[:50]}", 0)
        
        # Generate A/B testing plan (with very short timeout - skip if it takes too long)
        ab_test_plan = None
        from src.agents.ab_testing_agent import ABTestPlan
        
        # Try A/B testing with very short timeout (8 seconds max)
        # If it fails, we'll skip it entirely to prevent blocking
        try:
            await broadcast_agent_thought(
                "A/B Testing Agent",
                "Generating A/B test variants (this may be skipped if it takes too long)...",
                "analyzing"
            )
            await broadcast_agent_status("ab_testing_agent", "active", "Generating A/B test variants", 60)
            
            # Reasonable timeout - skip if it takes longer than 15 seconds
            # Use asyncio.wait_for with a timeout to prevent hanging
            ab_test_plan = await asyncio.wait_for(
                ab_testing_agent.generate_test_plan(campaign_data, variant_count=2, channels=["Email"]),  # Minimal: 1 channel, 2 variants
                timeout=15.0  # 15 second timeout - reasonable for LLM call
            )
            await broadcast_agent_status("ab_testing_agent", "complete", f"Generated {len(ab_test_plan.variants)} variants", 100)
            await broadcast_agent_thought(
                "A/B Testing Agent",
                f"Successfully generated {len(ab_test_plan.variants)} A/B test variants",
                "complete"
            )
        except asyncio.TimeoutError:
            logger.warning("A/B testing timed out after 15 seconds, skipping to prevent blocking")
            await broadcast_agent_thought(
                "A/B Testing Agent",
                "A/B testing skipped (took too long). Analysis will continue without it.",
                "warning"
            )
            await broadcast_agent_status("ab_testing_agent", "idle", "Skipped (timeout)", 0)
            # Create minimal A/B test plan
            ab_test_plan = ABTestPlan(
                campaign_id=campaign_data.get("campaign_id", "unknown"),
                variants=[],
                recommended_tests=[],
                testing_strategy="A/B testing was skipped to prevent timeout. The rest of the analysis completed successfully.",
                success_metrics=["click-through rate", "conversion rate"],
                sample_size_recommendations={},
                duration_recommendations={},
            )
        except Exception as e:
            logger.error("A/B testing plan generation failed", error=str(e), exc_info=True)
            await broadcast_agent_thought(
                "A/B Testing Agent",
                f"A/B testing skipped due to error: {str(e)[:80]}. Analysis will continue.",
                "error"
            )
            await broadcast_agent_status("ab_testing_agent", "error", "Skipped (error)", 0)
            # Create minimal A/B test plan
            ab_test_plan = ABTestPlan(
                campaign_id=campaign_data.get("campaign_id", "unknown"),
                variants=[],
                recommended_tests=[],
                testing_strategy="A/B testing was skipped due to an error. The rest of the analysis completed successfully.",
                success_metrics=["click-through rate", "conversion rate"],
                sample_size_recommendations={},
                duration_recommendations={},
            )
        
        # Ensure ab_test_plan is always valid
        if ab_test_plan is None:
            from src.agents.ab_testing_agent import ABTestPlan
            ab_test_plan = ABTestPlan(
                campaign_id=campaign_data.get("campaign_id", "unknown"),
                variants=[],
                recommended_tests=[],
                testing_strategy="A/B testing not available",
                success_metrics=["click-through rate", "conversion rate"],
                sample_size_recommendations={},
                duration_recommendations={},
            )
        
        # Status already updated above, no need to update again
        
        # Create summary of extracted campaign data
        extracted_summary = {
            "title": campaign_data.get("title", "Unknown"),
            "description": campaign_data.get("description", "")[:200] + ("..." if len(campaign_data.get("description", "")) > 200 else ""),
            "category": campaign_data.get("category", "Unknown"),
            "goal_amount": campaign_data.get("goal_amount", 0),
            "raised_amount": campaign_data.get("raised_amount", 0),
            "donor_count": campaign_data.get("donor_count", 0),
            "location": campaign_data.get("location", ""),
            "organizer_name": campaign_data.get("organizer_name", ""),
            "url": campaign_data.get("url", url),
            "progress_percentage": (campaign_data.get("raised_amount", 0) / campaign_data.get("goal_amount", 1) * 100) if campaign_data.get("goal_amount", 0) > 0 else 0,
        }
    
        # Build A/B testing response with defensive error handling
        try:
            ab_testing_data = {
                "variants": [
                    {
                        "variant_id": getattr(v, "variant_id", f"variant_{i}"),
                        "variant_name": getattr(v, "variant_name", "Unnamed Variant"),
                        "message": getattr(v, "message", ""),
                        "channel": getattr(v, "channel", "General"),
                        "target_audience": getattr(v, "target_audience", ""),
                        "hypothesis": getattr(v, "hypothesis", ""),
                        "expected_outcome": getattr(v, "expected_outcome", ""),
                        "testing_priority": getattr(v, "testing_priority", "medium"),
                    }
                    for i, v in enumerate(getattr(ab_test_plan, "variants", []) if ab_test_plan else [])
                ],
                "recommended_tests": getattr(ab_test_plan, "recommended_tests", []) if ab_test_plan else [],
                "testing_strategy": getattr(ab_test_plan, "testing_strategy", "Not available") if ab_test_plan else "Not available",
                "success_metrics": getattr(ab_test_plan, "success_metrics", []) if ab_test_plan else [],
                "sample_size_recommendations": getattr(ab_test_plan, "sample_size_recommendations", {}) if ab_test_plan else {},
                "duration_recommendations": getattr(ab_test_plan, "duration_recommendations", {}) if ab_test_plan else {},
            }
        except Exception as e:
            logger.error("Error building A/B testing response", error=str(e), exc_info=True)
            ab_testing_data = {
                "variants": [],
                "recommended_tests": [],
                "testing_strategy": "Error building A/B testing data",
                "success_metrics": [],
                "sample_size_recommendations": {},
                "duration_recommendations": {},
            }
        
        # Mark all agents as complete/idle after successful analysis
        await broadcast_agent_status("campaign_data_agent", "idle", "", 0)
        await broadcast_agent_status("campaign_intelligence_agent", "idle", "", 0)
        await broadcast_agent_status("tone_checker_agent", "idle", "", 0)
        await broadcast_agent_status("ab_testing_agent", "idle", "", 0)
    
        # Build response with comprehensive error handling to prevent blank UI
        try:
            response_data = {
                "extracted_campaign_summary": extracted_summary,
            "tone_analysis": {
                "overall_tone_score": tone_analysis.overall_tone_score,
                "empathy_score": tone_analysis.empathy_score,
                "authenticity_score": tone_analysis.authenticity_score,
                "clarity_score": tone_analysis.clarity_score,
                "urgency_appropriateness": tone_analysis.urgency_appropriateness,
                "respectful_language_score": tone_analysis.respectful_language_score,
                "tone_issues": tone_analysis.tone_issues,
                "tone_strengths": tone_analysis.tone_strengths,
                "improvement_suggestions": tone_analysis.improvement_suggestions,
                "sensitive_phrases": tone_analysis.sensitive_phrases,
                "recommended_changes": tone_analysis.recommended_changes,
            },
            "quality_score": {
                "overall_score": intelligence.quality_score.overall_score,
                "dimension_scores": intelligence.quality_score.dimension_scores,
                "priority_improvements": intelligence.quality_score.priority_improvements,
            },
            "success_prediction": {
                "success_probability": intelligence.success_prediction.success_probability,
                "predicted_amount": intelligence.success_prediction.predicted_amount,
                "predicted_donors": intelligence.success_prediction.predicted_donors,
                "confidence_level": intelligence.success_prediction.confidence_level,
                "key_factors": intelligence.success_prediction.key_factors,
                "risk_factors": intelligence.success_prediction.risk_factors,
            },
            "similar_campaigns": [
                {
                    "title": sc.title,
                    "success_metrics": sc.success_metrics,
                    "what_made_it_work": sc.what_made_it_work,
                    "similarity_score": sc.similarity_score,
                }
                for sc in intelligence.similar_campaigns
            ],
            "messaging_variants": intelligence.messaging_variants,
            "ab_testing": ab_testing_data,
            }
            return response_data
        except Exception as e:
            logger.error("Error building final response", error=str(e), exc_info=True)
            # Return a minimal valid response to prevent blank UI
            await broadcast_agent_thought(
                "System",
                f"Error building response: {str(e)[:100]}. Returning partial results.",
                "error"
            )
            return {
                "extracted_campaign_summary": extracted_summary if 'extracted_summary' in locals() else {"title": "Error", "description": "Error occurred"},
                "tone_analysis": {
                    "overall_tone_score": tone_analysis.overall_tone_score if 'tone_analysis' in locals() else 0.5,
                    "empathy_score": getattr(tone_analysis, "empathy_score", 0.5) if 'tone_analysis' in locals() else 0.5,
                    "authenticity_score": getattr(tone_analysis, "authenticity_score", 0.5) if 'tone_analysis' in locals() else 0.5,
                    "clarity_score": getattr(tone_analysis, "clarity_score", 0.5) if 'tone_analysis' in locals() else 0.5,
                    "urgency_appropriateness": getattr(tone_analysis, "urgency_appropriateness", 0.5) if 'tone_analysis' in locals() else 0.5,
                    "respectful_language_score": getattr(tone_analysis, "respectful_language_score", 0.5) if 'tone_analysis' in locals() else 0.5,
                    "tone_issues": getattr(tone_analysis, "tone_issues", []) if 'tone_analysis' in locals() else [],
                    "tone_strengths": getattr(tone_analysis, "tone_strengths", []) if 'tone_analysis' in locals() else [],
                    "improvement_suggestions": getattr(tone_analysis, "improvement_suggestions", []) if 'tone_analysis' in locals() else [],
                    "sensitive_phrases": getattr(tone_analysis, "sensitive_phrases", []) if 'tone_analysis' in locals() else [],
                    "recommended_changes": getattr(tone_analysis, "recommended_changes", []) if 'tone_analysis' in locals() else [],
                },
                "quality_score": {
                    "overall_score": intelligence.quality_score.overall_score if 'intelligence' in locals() else 0.5,
                    "dimension_scores": getattr(intelligence.quality_score, "dimension_scores", {}) if 'intelligence' in locals() else {},
                    "priority_improvements": getattr(intelligence.quality_score, "priority_improvements", []) if 'intelligence' in locals() else [],
                },
                "success_prediction": {
                    "success_probability": getattr(intelligence.success_prediction, "success_probability", 0.5) if 'intelligence' in locals() else 0.5,
                    "predicted_amount": getattr(intelligence.success_prediction, "predicted_amount", 0) if 'intelligence' in locals() else 0,
                    "predicted_donors": getattr(intelligence.success_prediction, "predicted_donors", 0) if 'intelligence' in locals() else 0,
                    "confidence_level": getattr(intelligence.success_prediction, "confidence_level", "low") if 'intelligence' in locals() else "low",
                    "key_factors": getattr(intelligence.success_prediction, "key_factors", []) if 'intelligence' in locals() else [],
                    "risk_factors": getattr(intelligence.success_prediction, "risk_factors", []) if 'intelligence' in locals() else [],
                },
                "similar_campaigns": getattr(intelligence, "similar_campaigns", []) if 'intelligence' in locals() else [],
                "messaging_variants": getattr(intelligence, "messaging_variants", {}) if 'intelligence' in locals() else {},
                "ab_testing": ab_testing_data if 'ab_testing_data' in locals() else {
                    "variants": [],
                    "recommended_tests": [],
                    "testing_strategy": "A/B testing unavailable due to error",
                    "success_metrics": [],
                    "sample_size_recommendations": {},
                    "duration_recommendations": {},
                },
                "error": f"Partial results returned due to error: {str(e)[:200]}",
            }
    except Exception as e:
        logger.error("Error in _perform_campaign_analysis", error=str(e), exc_info=True)
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in _perform_campaign_analysis: {e}")
        print(error_details)
        # Re-raise to be caught by outer handler
        raise


@app.post("/api/donor-journey")
async def donor_journey_endpoint(request: DonorJourneyRequest):
    """Endpoint for donor journey simulation."""
    try:
        # Broadcast that we're starting
        await broadcast_agent_thought(
            "Donor Affinity Profiler",
            "Starting donor profile building...",
            "start"
        )
        await broadcast_agent_status("donor_affinity_profiler", "active", "Starting donor profile building", 0)
        
        from src.agents.donor_affinity_profiler import DonorAffinityProfiler
        from src.agents.campaign_matching_engine import CampaignMatchingEngine
        from src.agents.community_discovery import CommunityDiscoveryAgent
        from src.agents.recurring_curator import RecurringCuratorAgent
        from src.agents.giving_circle_orchestrator import GivingCircleOrchestrator
        from src.agents.engagement_agent import EngagementAgent
        from datetime import datetime
        
        donor_info = request.donor_info or {}
        donations_raw = request.donations or []
        
        await broadcast_agent_thought(
            "Donor Affinity Profiler",
            f"Processing {len(donations_raw)} donations for donor profile...",
            "processing"
        )
        await broadcast_agent_status("donor_affinity_profiler", "active", f"Processing {len(donations_raw)} donations", 20)
        
        # Normalize donations format - ensure all are dicts with proper fields
        donations = []
        for d in donations_raw:
            if isinstance(d, dict):
                try:
                    # Ensure required fields exist with proper types
                    donation = {
                        "amount": float(d.get("amount", 0)),
                        "category": d.get("category", ""),
                        "campaign_category": d.get("category", ""),  # Also set campaign_category for profiler
                        "campaign_title": d.get("description", d.get("campaign_title", "")),
                        "timestamp": d.get("timestamp", datetime.now().isoformat()),
                    }
                    donations.append(donation)
                except (ValueError, TypeError) as e:
                    # Skip invalid donations
                    print(f"Warning: Skipping invalid donation: {e}")
                    continue
        
        # Build profile
        await broadcast_agent_thought(
            "Donor Affinity Profiler",
            "Analyzing donation patterns and identifying cause affinities...",
            "analyzing"
        )
        await broadcast_agent_status("donor_affinity_profiler", "active", "Analyzing donation patterns", 50)
        
        profiler = DonorAffinityProfiler()
        profile = await profiler.build_profile(
            donor_id=donor_info.get("email", "donor_unknown"),
            donations=donations,
            metadata=donor_info,
        )
        
        profile_dict = profile.to_dict() if hasattr(profile, "to_dict") else profile
        
        await broadcast_agent_thought(
            "Donor Affinity Profiler",
            f"Profile built! Total giving: ${profile_dict.get('total_lifetime_giving', 0):,.2f}",
            "profile_complete"
        )
        await broadcast_agent_status("donor_affinity_profiler", "complete", "Profile built successfully", 100)
        
        # Get recommendations
        await broadcast_agent_thought(
            "Campaign Matching Engine",
            "Finding personalized campaign recommendations...",
            "matching"
        )
        await broadcast_agent_status("campaign_matching_engine", "active", "Finding personalized recommendations", 60)
        
        matcher = CampaignMatchingEngine()
        sample_campaigns = [
            {
                "campaign_id": "campaign_1",
                "title": "Help Sarah Fight Cancer",
                "category": "medical",
            },
            {
                "campaign_id": "campaign_2",
                "title": "Scholarship Fund",
                "category": "education",
            },
        ]
        
        recommendations = []
        for campaign in sample_campaigns:
            match_result = await matcher._tool_match_to_donor(
                campaign=campaign,
                donor_profile=profile_dict,
            )
            recommendations.append({
                "campaign_title": campaign["title"],
                "match_score": match_result.get("match_score", 0),
                "reasons": match_result.get("reasons", []),
            })
        
        await broadcast_agent_thought(
            "Campaign Matching Engine",
            f"Found {len(recommendations)} matching campaigns!",
            "matching_complete"
        )
        await broadcast_agent_status("campaign_matching_engine", "complete", f"Found {len(recommendations)} recommendations", 100)
        
        # Step 3: Community Discovery
        await broadcast_agent_thought(
            "Community Discovery Agent",
            "Discovering local community opportunities...",
            "discovering"
        )
        await broadcast_agent_status("community_discovery", "active", "Discovering community opportunities", 70)
        
        try:
            community_agent = CommunityDiscoveryAgent()
            location = donor_info.get("location", "")
            community_insights = await community_agent.discover_communities(
                donor_profile=profile_dict,
                location=location,
            )
            community_dict = community_insights.to_dict() if hasattr(community_insights, "to_dict") else community_insights
            
            await broadcast_agent_thought(
                "Community Discovery Agent",
                f"Found {len(community_dict.get('local_campaigns', []))} local community opportunities",
                "discovery_complete"
            )
            await broadcast_agent_status("community_discovery", "complete", "Community discovery complete", 100)
        except Exception as e:
            logger.warning("Community discovery failed", error=str(e))
            await broadcast_agent_thought(
                "Community Discovery Agent",
                f"Community discovery encountered an issue: {str(e)[:100]}",
                "error"
            )
            await broadcast_agent_status("community_discovery", "error", "Discovery failed", 0)
            community_dict = {}
        
        # Step 4: Recurring Opportunities
        await broadcast_agent_thought(
            "Recurring Curator Agent",
            "Identifying recurring giving opportunities...",
            "curating"
        )
        await broadcast_agent_status("recurring_curator", "active", "Identifying recurring opportunities", 80)
        
        try:
            curator_agent = RecurringCuratorAgent()
            sample_campaigns_for_recurring = [
                {
                    "campaign_id": "campaign_1",
                    "title": "Help Sarah Fight Cancer",
                    "category": "medical",
                    "description": "Long-term treatment support needed",
                },
            ]
            recurring_opportunities = await curator_agent.curate_opportunities(
                campaigns=sample_campaigns_for_recurring,
            )
            recurring_list = [
                opp.to_dict() if hasattr(opp, "to_dict") else opp
                for opp in recurring_opportunities
            ]
            
            await broadcast_agent_thought(
                "Recurring Curator Agent",
                f"Identified {len(recurring_list)} recurring giving opportunities",
                "curation_complete"
            )
            await broadcast_agent_status("recurring_curator", "complete", f"Found {len(recurring_list)} opportunities", 100)
        except Exception as e:
            logger.warning("Recurring curation failed", error=str(e))
            await broadcast_agent_thought(
                "Recurring Curator Agent",
                f"Recurring curation encountered an issue: {str(e)[:100]}",
                "error"
            )
            await broadcast_agent_status("recurring_curator", "error", "Curation failed", 0)
            recurring_list = []
        
        # Step 5: Giving Circles
        await broadcast_agent_thought(
            "Giving Circle Orchestrator",
            "Checking for giving circle opportunities...",
            "orchestrating"
        )
        await broadcast_agent_status("giving_circle_orchestrator", "active", "Checking giving circles", 85)
        
        try:
            giving_circle_agent = GivingCircleOrchestrator()
            circle_suggestions = await giving_circle_agent.orchestrate_circle(
                donor_profile=profile_dict,
                community_context=community_dict,
            )
            circles_list = [
                circle.to_dict() if hasattr(circle, "to_dict") else circle
                for circle in circle_suggestions
            ]
            
            await broadcast_agent_thought(
                "Giving Circle Orchestrator",
                f"Found {len(circles_list)} giving circle opportunities",
                "orchestration_complete"
            )
            await broadcast_agent_status("giving_circle_orchestrator", "complete", f"Found {len(circles_list)} circles", 100)
        except Exception as e:
            logger.warning("Giving circle orchestration failed", error=str(e))
            await broadcast_agent_thought(
                "Giving Circle Orchestrator",
                f"Giving circle orchestration encountered an issue: {str(e)[:100]}",
                "error"
            )
            await broadcast_agent_status("giving_circle_orchestrator", "error", "Orchestration failed", 0)
            circles_list = []
        
        # Step 6: Engagement Planning
        await broadcast_agent_thought(
            "Engagement Agent",
            "Creating personalized engagement plan...",
            "planning"
        )
        await broadcast_agent_status("engagement_agent", "active", "Creating engagement plan", 90)
        
        try:
            engagement_agent = EngagementAgent()
            engagement_plan = await engagement_agent.create_engagement_plan(
                donor_profile=profile_dict,
                additional_context={},
            )
            engagement_dict = engagement_plan.to_dict() if hasattr(engagement_plan, "to_dict") else engagement_plan
            
            await broadcast_agent_thought(
                "Engagement Agent",
                f"Created engagement plan with {len(engagement_dict.get('recommended_actions', []))} actions",
                "planning_complete"
            )
            await broadcast_agent_status("engagement_agent", "complete", "Engagement plan created", 100)
        except Exception as e:
            logger.warning("Engagement planning failed", error=str(e))
            await broadcast_agent_thought(
                "Engagement Agent",
                f"Engagement planning encountered an issue: {str(e)[:100]}",
                "error"
            )
            await broadcast_agent_status("engagement_agent", "error", "Planning failed", 0)
            engagement_dict = {}
        
        # Mark all agents as idle after completion
        await broadcast_agent_status("donor_affinity_profiler", "idle", "", 0)
        await broadcast_agent_status("campaign_matching_engine", "idle", "", 0)
        await broadcast_agent_status("community_discovery", "idle", "", 0)
        await broadcast_agent_status("recurring_curator", "idle", "", 0)
        await broadcast_agent_status("giving_circle_orchestrator", "idle", "", 0)
        await broadcast_agent_status("engagement_agent", "idle", "", 0)
        
        return {
            "profile": profile_dict,
            "recommendations": recommendations,
            "community_insights": community_dict,
            "recurring_opportunities": recurring_list,
            "giving_circles": circles_list,
            "engagement_plan": engagement_dict,
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in donor_journey_endpoint: {e}")
        print(error_details)
        await broadcast_agent_thought(
            "System",
            f"Error building profile: {str(e)}",
            "error"
        )
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# In-memory storage for chat session data
_chat_sessions = {}  # {session_id: {"donations": [], "donor_info": {}}}
_a2a_protocol = None
_chat_orchestrator = None
_registered_agents = {}


def _initialize_agents():
    """Initialize and register all agents with A2A protocol."""
    global _a2a_protocol, _chat_orchestrator, _registered_agents
    
    if _a2a_protocol is None:
        try:
            from src.core.a2a_protocol import get_a2a_protocol
            from src.agents.chat_orchestrator import ChatOrchestrator
            from src.agents.campaign_data_agent import CampaignDataAgent
            from src.agents.campaign_intelligence_agent import CampaignIntelligenceAgent
            from src.agents.tone_checker_agent import ToneCheckerAgent
            from src.agents.ab_testing_agent import ABTestingAgent
            from src.agents.donor_affinity_profiler import DonorAffinityProfiler
            from src.agents.campaign_matching_engine import CampaignMatchingEngine
            from src.agents.community_discovery import CommunityDiscoveryAgent
            from src.agents.recurring_curator import RecurringCuratorAgent
            from src.agents.giving_circle_orchestrator import GivingCircleOrchestrator
            from src.agents.engagement_agent import EngagementAgent
            
            _a2a_protocol = get_a2a_protocol()
            
            # Create and register all agents
            agents = []
            
            # Create chat orchestrator first (most critical)
            try:
                logger.info("Creating ChatOrchestrator...")
                print("DEBUG: About to create ChatOrchestrator...")
                chat_orch = ChatOrchestrator()
                print(f"DEBUG: ChatOrchestrator created, agent_id={chat_orch.agent_id}")
                agents.append(("chat_orchestrator", chat_orch))
                logger.info("ChatOrchestrator created successfully", agent_id=chat_orch.agent_id)
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.error("Failed to create ChatOrchestrator", error=str(e), traceback=error_trace, exc_info=True)
                print(f"\n{'='*60}")
                print("CRITICAL: ChatOrchestrator creation failed!")
                print(f"Error: {str(e)}")
                print(f"\nFull traceback:\n{error_trace}")
                print(f"{'='*60}\n")
                raise  # Re-raise since chat orchestrator is critical
            
            # Create other agents
            try:
                agents.extend([
                    ("campaign_data_agent", CampaignDataAgent()),
                    ("campaign_intelligence_agent", CampaignIntelligenceAgent()),
                    ("tone_checker_agent", ToneCheckerAgent()),
                    ("ab_testing_agent", ABTestingAgent()),
                    ("donor_affinity_profiler", DonorAffinityProfiler()),
                    ("campaign_matching_engine", CampaignMatchingEngine()),
                    ("community_discovery", CommunityDiscoveryAgent()),
                    ("recurring_curator", RecurringCuratorAgent()),
                    ("giving_circle_orchestrator", GivingCircleOrchestrator()),
                    ("engagement_agent", EngagementAgent()),
                ])
                logger.info("All agents created successfully")
            except Exception as e:
                logger.error("Failed to create some agents", error=str(e), exc_info=True)
                # Continue with agents that were created
            
            for agent_id, agent in agents:
                try:
                    logger.info("Creating agent", agent_id=agent_id)
                    # Agent should already be created, just register it
                    _a2a_protocol.register_agent(agent)
                    _registered_agents[agent_id] = agent
                    logger.info("Agent registered successfully", agent_id=agent_id, name=agent.name)
                except Exception as e:
                    logger.error("Failed to register agent", agent_id=agent_id, error=str(e), exc_info=True)
                    # Continue with other agents even if one fails
                    if agent_id == "chat_orchestrator":
                        # If chat orchestrator fails, log more details
                        logger.error("CRITICAL: Chat orchestrator registration failed", 
                                   agent_id=agent_id,
                                   agent_type=type(agent).__name__,
                                   has_agent_id=hasattr(agent, "agent_id"),
                                   agent_id_value=getattr(agent, "agent_id", "N/A"),
                                   error=str(e),
                                   exc_info=True)
            
            if "chat_orchestrator" in _registered_agents:
                _chat_orchestrator = _registered_agents["chat_orchestrator"]
                logger.info("All agents initialized and registered with A2A protocol")
            else:
                logger.error("Chat orchestrator failed to initialize")
                _chat_orchestrator = None
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error("Failed to initialize agents", error=str(e), traceback=error_trace, exc_info=True)
            print(f"\n{'='*60}")
            print("CRITICAL ERROR: Agent initialization failed!")
            print(f"Error: {str(e)}")
            print(f"\nFull traceback:\n{error_trace}")
            print(f"{'='*60}\n")
            _a2a_protocol = None
            _chat_orchestrator = None
            _registered_agents = {}


@app.post("/api/chat")
async def chat_endpoint(chat_request: ChatRequest, http_request: Request = None):
    """Chat endpoint with dynamic tool calling, A2A, and MCP integration."""
    try:
        # Initialize agents on first request
        if _a2a_protocol is None or _chat_orchestrator is None:
            logger.info("Initializing agents for chat endpoint...")
            _initialize_agents()
            if _chat_orchestrator is None:
                error_msg = "Failed to initialize chat orchestrator. Please check server logs for details."
                logger.error(error_msg)
                return {
                    "response": error_msg,
                    "agent_name": "System",
                    "error": "Initialization failed",
                }
        
        message = chat_request.message.strip() if chat_request.message else ""
        
        if not message:
            return {
                "response": "Please provide a message.",
                "agent_name": "Assistant",
            }
        
        # Get or create session
        session_id = "default"
        if session_id not in _chat_sessions:
            _chat_sessions[session_id] = {"donations": [], "donor_info": {}}
        session = _chat_sessions[session_id]
        
        # Prepare context for orchestrator
        context = {
            "donations": session["donations"],
            "donor_info": session["donor_info"],
            "session_id": session_id,
        }
        
        # Use chat orchestrator with dynamic tool calling - no fallback
        if _chat_orchestrator is None:
            raise Exception("Chat orchestrator not initialized. Please check server logs.")
        
        # Add timeout wrapper to prevent hanging
        try:
            result = await asyncio.wait_for(
                _chat_orchestrator.process_chat_message(message, context),
                timeout=45.0  # 45 second overall timeout
            )
        except asyncio.TimeoutError:
            logger.error("Chat endpoint timed out after 45 seconds")
            return {
                "response": "I'm taking longer than expected to process your request. This might be due to a complex analysis. Please try again with a simpler question or check back in a moment.",
                "agent_name": "System",
                "error": "Request timed out",
            }
        
        # Update session if donations were added (from donation parsing)
        if "donations" in result.get("results", {}).get("build_donor_profile", {}):
            # Donations were processed, session already updated
            pass
        
        # Return orchestrator response
        return {
            "response": result.get("response", "I'm processing your request..."),
            "agent_name": "Chat Orchestrator",
            "tools_used": result.get("tools_used", []),
            "session_data": {
                "donation_count": len(session["donations"]),
                "has_profile": len(session["donations"]) > 0,
            },
        }
        
    except Exception as e:
        logger.error("Error in chat endpoint", error=str(e), exc_info=True)
        return {
            "response": f"I encountered an error: {str(e)[:200]}. Please try rephrasing your question or check if the backend is running properly.",
            "agent_name": "System",
            "error": str(e),
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

