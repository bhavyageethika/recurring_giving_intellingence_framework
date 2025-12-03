"""
Model Context Protocol (MCP) Server

Provides standardized interface for agents to access external tools and services.
Enables integration with email, contacts, calendar, documents, and other services.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import asyncio

import structlog

logger = structlog.get_logger()


@dataclass
class MCPTool:
    """Represents an MCP tool that can be called by agents."""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: callable
    requires_auth: bool = False
    rate_limit: Optional[int] = None  # requests per minute


@dataclass
class MCPResource:
    """Represents an MCP resource (like a contact list or email account)."""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPServer:
    """
    MCP Server that provides tools and resources to agents.
    
    Tools are functions that agents can call (e.g., send_email, get_contacts).
    Resources are data sources agents can access (e.g., contact list, calendar).
    """
    
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._logger = logger.bind(component="mcp_server")
        self._rate_limits: Dict[str, List[datetime]] = {}
        
        # Register built-in tools
        self._register_builtin_tools()
    
    def register_tool(self, tool: MCPTool) -> None:
        """Register a tool that agents can use."""
        self._tools[tool.name] = tool
        self._logger.info("mcp_tool_registered", tool_name=tool.name)
    
    def register_resource(self, resource: MCPResource) -> None:
        """Register a resource that agents can access."""
        self._resources[resource.uri] = resource
        self._logger.info("mcp_resource_registered", uri=resource.uri)
    
    def _register_builtin_tools(self):
        """Register built-in MCP tools."""
        # Contacts tool
        self.register_tool(MCPTool(
            name="get_contacts",
            description="Retrieve contacts from connected address book. Can filter by name, email, or tags.",
            parameters={
                "type": "object",
                "properties": {
                    "filter": {
                        "type": "object",
                        "description": "Optional filters: name, email, tags",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "limit": {"type": "integer", "description": "Max number of contacts to return", "default": 100},
                },
            },
            handler=self._handle_get_contacts,
        ))
        
        # Email tool
        self.register_tool(MCPTool(
            name="send_email",
            description="Send an email. Requires recipient email, subject, and body. Can include attachments.",
            parameters={
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email address"},
                    "cc": {"type": "array", "items": {"type": "string"}, "description": "CC recipients"},
                    "bcc": {"type": "array", "items": {"type": "string"}, "description": "BCC recipients"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body (HTML or plain text)"},
                    "body_type": {"type": "string", "enum": ["html", "text"], "default": "text"},
                    "attachments": {"type": "array", "items": {"type": "string"}, "description": "Attachment file paths"},
                },
                "required": ["to", "subject", "body"],
            },
            handler=self._handle_send_email,
            requires_auth=True,
        ))
        
        # Calendar tool
        self.register_tool(MCPTool(
            name="create_calendar_event",
            description="Create a calendar event. Useful for scheduling follow-ups, reminders, or outreach.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title"},
                    "description": {"type": "string", "description": "Event description"},
                    "start_time": {"type": "string", "description": "ISO format datetime"},
                    "end_time": {"type": "string", "description": "ISO format datetime"},
                    "attendees": {"type": "array", "items": {"type": "string"}, "description": "Attendee emails"},
                    "reminder_minutes": {"type": "integer", "description": "Minutes before event for reminder"},
                },
                "required": ["title", "start_time"],
            },
            handler=self._handle_create_calendar_event,
        ))
        
        # Document tool
        self.register_tool(MCPTool(
            name="save_document",
            description="Save content to a document (Google Docs, local file, etc.).",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Document title"},
                    "content": {"type": "string", "description": "Document content"},
                    "format": {"type": "string", "enum": ["markdown", "html", "text"], "default": "text"},
                    "destination": {"type": "string", "description": "Where to save (local path or cloud service)"},
                },
                "required": ["title", "content"],
            },
            handler=self._handle_save_document,
        ))
    
    async def _handle_get_contacts(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get contacts from address book.
        
        In production, this would connect to:
        - Google Contacts API
        - Outlook/Exchange API
        - Apple Contacts
        - CRM systems
        
        For now, uses a mock contact store that can be populated.
        """
        # Mock contact store (in production, connect to real API)
        contacts = self._get_mock_contacts()
        
        # Apply filters
        if filter:
            if "name" in filter:
                name_filter = filter["name"].lower()
                contacts = [c for c in contacts if name_filter in c.get("name", "").lower()]
            
            if "email" in filter:
                email_filter = filter["email"].lower()
                contacts = [c for c in contacts if email_filter in c.get("email", "").lower()]
            
            if "tags" in filter:
                tag_filter = set(filter["tags"])
                contacts = [
                    c for c in contacts
                    if tag_filter.intersection(set(c.get("tags", [])))
                ]
        
        # Limit results
        contacts = contacts[:limit]
        
        return {
            "contacts": contacts,
            "count": len(contacts),
            "total_available": len(self._get_mock_contacts()),
        }
    
    def _get_mock_contacts(self) -> List[Dict[str, Any]]:
        """Mock contact store. In production, replace with real API calls."""
        # This can be populated from user's actual contacts or a CSV import
        return getattr(self, "_contact_store", [])
    
    def load_contacts(self, contacts: List[Dict[str, Any]]) -> None:
        """Load contacts into the mock store (for demo/testing)."""
        self._contact_store = contacts
        self._logger.info("contacts_loaded", count=len(contacts))
    
    async def _handle_send_email(
        self,
        to: str,
        subject: str,
        body: str,
        cc: Optional[List[str]] = None,
        bcc: Optional[List[str]] = None,
        body_type: str = "text",
        attachments: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Send an email.
        
        In production, this would connect to:
        - Gmail API
        - SendGrid
        - Mailgun
        - SMTP server
        
        For now, logs the email and returns success.
        """
        email_data = {
            "to": to,
            "cc": cc or [],
            "bcc": bcc or [],
            "subject": subject,
            "body": body,
            "body_type": body_type,
            "attachments": attachments or [],
            "sent_at": datetime.utcnow().isoformat(),
        }
        
        # In production, actually send via email service
        self._logger.info("email_sent", to=to, subject=subject[:50])
        
        # Store sent emails for tracking
        if not hasattr(self, "_sent_emails"):
            self._sent_emails = []
        self._sent_emails.append(email_data)
        
        return {
            "success": True,
            "message_id": f"email_{datetime.utcnow().timestamp()}",
            "to": to,
            "subject": subject,
        }
    
    async def _handle_create_calendar_event(
        self,
        title: str,
        start_time: str,
        description: Optional[str] = None,
        end_time: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        reminder_minutes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a calendar event.
        
        In production, this would connect to:
        - Google Calendar API
        - Outlook Calendar API
        - iCal/CalDAV
        
        For now, creates a mock event.
        """
        event = {
            "title": title,
            "description": description or "",
            "start_time": start_time,
            "end_time": end_time or start_time,
            "attendees": attendees or [],
            "reminder_minutes": reminder_minutes,
            "created_at": datetime.utcnow().isoformat(),
        }
        
        self._logger.info("calendar_event_created", title=title)
        
        # Store events for tracking
        if not hasattr(self, "_calendar_events"):
            self._calendar_events = []
        self._calendar_events.append(event)
        
        return {
            "success": True,
            "event_id": f"event_{datetime.utcnow().timestamp()}",
            "title": title,
        }
    
    async def _handle_save_document(
        self,
        title: str,
        content: str,
        format: str = "text",
        destination: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save a document.
        
        In production, this would connect to:
        - Google Docs API
        - Microsoft Word Online
        - Local file system
        - Cloud storage (S3, Dropbox, etc.)
        
        For now, saves to local file system.
        """
        import os
        from pathlib import Path
        
        # Default destination
        if not destination:
            docs_dir = Path("mcp_documents")
            docs_dir.mkdir(exist_ok=True)
            destination = str(docs_dir / f"{title.replace(' ', '_')}.{format}")
        
        # Save file
        with open(destination, "w", encoding="utf-8") as f:
            f.write(content)
        
        self._logger.info("document_saved", title=title, destination=destination)
        
        return {
            "success": True,
            "file_path": destination,
            "title": title,
        }
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool by name with arguments.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool handler
            
        Returns:
            Result from the tool handler
        """
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        tool = self._tools[tool_name]
        
        # Check rate limit
        if tool.rate_limit:
            await self._check_rate_limit(tool_name, tool.rate_limit)
        
        # Call handler
        try:
            result = await tool.handler(**arguments)
            self._logger.info("mcp_tool_called", tool_name=tool_name, success=True)
            return result
        except Exception as e:
            self._logger.error("mcp_tool_failed", tool_name=tool_name, error=str(e))
            raise
    
    async def _check_rate_limit(self, tool_name: str, limit_per_minute: int):
        """Check and enforce rate limits."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        if tool_name not in self._rate_limits:
            self._rate_limits[tool_name] = []
        
        # Remove old entries
        self._rate_limits[tool_name] = [
            ts for ts in self._rate_limits[tool_name]
            if ts > minute_ago
        ]
        
        # Check limit
        if len(self._rate_limits[tool_name]) >= limit_per_minute:
            raise Exception(f"Rate limit exceeded for {tool_name}: {limit_per_minute} requests/minute")
        
        # Add current request
        self._rate_limits[tool_name].append(now)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available MCP tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "requires_auth": tool.requires_auth,
            }
            for tool in self._tools.values()
        ]
    
    def list_resources(self) -> List[Dict[str, Any]]:
        """List all available MCP resources."""
        return [
            {
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mime_type": resource.mime_type,
            }
            for resource in self._resources.values()
        ]


# Singleton instance
_mcp_server: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """Get the singleton MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
    return _mcp_server

