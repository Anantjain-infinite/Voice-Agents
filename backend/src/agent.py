import logging
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


# Lead data storage
class LeadData:
    def __init__(self):
        self.name: Optional[str] = None
        self.company: Optional[str] = None
        self.email: Optional[str] = None
        self.role: Optional[str] = None
        self.use_case: Optional[str] = None
        self.team_size: Optional[str] = None
        self.timeline: Optional[str] = None
        self.conversation_summary: list = []
    
    def to_dict(self):
        return {
            "name": self.name,
            "company": self.company,
            "email": self.email,
            "role": self.role,
            "use_case": self.use_case,
            "team_size": self.team_size,
            "timeline": self.timeline,
            "conversation_summary": self.conversation_summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_to_file(self):
        os.makedirs("leads", exist_ok=True)
        filename = f"leads/lead_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Lead data saved to {filename}")
        return filename


class FAQSearchEngine:
    """Preprocessed FAQ search engine with optimized keyword matching"""
    
    def __init__(self, faq_data: Dict[str, Any]):
        self.faq_data = faq_data
        self.search_index = self._build_search_index()
        logger.info("FAQ search engine initialized with preprocessed data")
    
    def _build_search_index(self) -> Dict[str, str]:
        """Build a keyword-to-response mapping for fast lookups"""
        index = {}
        
        # Pricing responses
        pricing_response = self._build_pricing_response()
        for keyword in ["price", "pricing", "cost", "fee", "expensive", "cheap", "free", "how much"]:
            index[keyword] = pricing_response
        
        # Product overview
        product_response = self._build_product_response()
        for keyword in ["what", "product", "do", "offer", "feature", "capability"]:
            index[keyword] = product_response
        
        # Target audience
        audience_response = self._build_audience_response()
        for keyword in ["who", "for", "audience", "customer", "user", "suitable"]:
            index[keyword] = audience_response
        
        # Free tier
        free_response = self.faq_data["common_questions"]["free_tier"]
        for keyword in ["free", "trial", "no cost"]:
            index[keyword] = free_response
        
        # Integration
        integration_response = self.faq_data["common_questions"]["integration"]
        for keyword in ["integrate", "integration", "connect", "api", "third party"]:
            index[keyword] = integration_response
        
        # Support
        support_response = self.faq_data["common_questions"]["support"]
        for keyword in ["support", "help", "customer service", "assistance"]:
            index[keyword] = support_response
        
        # Security
        security_response = self.faq_data["common_questions"]["data_security"]
        for keyword in ["security", "secure", "safe", "privacy", "gdpr", "compliance"]:
            index[keyword] = security_response
        
        # Migration
        migration_response = self.faq_data["common_questions"]["migration"]
        for keyword in ["migration", "migrate", "import", "transfer", "switch"]:
            index[keyword] = migration_response
        
        # Customization
        customization_response = self.faq_data["common_questions"]["customization"]
        for keyword in ["customize", "customization", "custom", "personalize"]:
            index[keyword] = customization_response
        
        # Scalability
        scalability_response = self.faq_data["common_questions"]["scalability"]
        for keyword in ["scale", "scalability", "grow", "growth", "expand"]:
            index[keyword] = scalability_response
        
        return index
    
    def _build_pricing_response(self) -> str:
        """Build comprehensive pricing response from FAQ data"""
        products = self.faq_data["products"]
        response = "Here's our pricing information:\n\n"
        
        # CRM pricing
        crm = products["zoho_crm"]["pricing"]
        response += f"Zoho CRM: Free for up to 3 users, paid plans from ${crm['standard']['price']} per user per month\n"
        
        # Zoho One pricing
        one = products["zoho_one"]["pricing"]
        response += f"Zoho One (all 45+ apps): ${one['all_employee']['price']} per employee per month (minimum 5 employees)\n"
        
        # Mail pricing
        mail = products["zoho_mail"]["pricing"]
        response += f"Zoho Mail: Free for 5 users, paid plans from ${mail['mail_lite']['price']} per user per month\n"
        
        # Books pricing
        books = products["zoho_books"]["pricing"]
        response += f"Zoho Books: Free up to $50K revenue, then ${books['standard']['price']} per month\n\n"
        
        response += "Many products have generous free tiers to get started!"
        return response
    
    def _build_product_response(self) -> str:
        """Build product overview response from FAQ data"""
        products = self.faq_data["products"]
        response = "Zoho offers a comprehensive suite of business applications including:\n\n"
        
        response += f"Zoho CRM: {products['zoho_crm']['description']}\n\n"
        response += f"Zoho One: {products['zoho_one']['description']}\n\n"
        response += f"Zoho Mail: {products['zoho_mail']['description']}\n\n"
        response += f"Zoho Books: {products['zoho_books']['description']}\n\n"
        response += f"Zoho Projects: {products['zoho_projects']['description']}\n\n"
        response += "And many more apps for marketing, HR, analytics, and collaboration!"
        
        return response
    
    def _build_audience_response(self) -> str:
        """Build target audience response from FAQ data"""
        company = self.faq_data["company"]
        return (
            f"{company['target_audience']}. Whether you're a solo entrepreneur or "
            "a growing team, we have solutions that scale with you."
        )
    
    def search(self, query: str) -> str:
        """Search the FAQ using preprocessed index"""
        query_lower = query.lower()
        
        # Find matching keywords
        for keyword, response in self.search_index.items():
            if keyword in query_lower:
                logger.info(f"FAQ match found for keyword: {keyword}")
                return response
        
        # Default response if no match
        logger.info("Using default FAQ response")
        return (
            f"{self.faq_data['company']['description']} "
            f"We offer both individual apps and Zoho One, which includes access to all 45+ applications. "
            f"{self.faq_data['common_questions']['free_tier']}"
        )


def load_faq_data() -> Dict[str, Any]:
    """Load FAQ data from JSON file"""
    faq_file = "zoho_faq.json"
    
    try:
        with open(faq_file, 'r') as f:
            faq_data = json.load(f)
            logger.info(f"Successfully loaded FAQ data from {faq_file}")
            return faq_data
    except FileNotFoundError:
        logger.error(f"FAQ file {faq_file} not found. Using fallback FAQ data.")
        # Fallback FAQ data if file doesn't exist
        return {
            "company": {
                "name": "Zoho",
                "description": "Zoho is a comprehensive suite of business software applications.",
                "target_audience": "Small to medium-sized businesses and enterprises"
            },
            "products": {
                "zoho_crm": {
                    "name": "Zoho CRM",
                    "description": "Customer relationship management software",
                    "pricing": {
                        "free": {"price": "$0", "users": "Up to 3 users"},
                        "standard": {"price": "$14 per user per month"}
                    }
                },
                "zoho_one": {
                    "name": "Zoho One",
                    "description": "Complete suite of 45+ integrated applications",
                    "pricing": {
                        "all_employee": {"price": "$37 per employee per month"}
                    }
                },
                "zoho_mail": {
                    "name": "Zoho Mail",
                    "description": "Ad-free business email",
                    "pricing": {
                        "free": {"price": "$0", "users": "Up to 5 users"},
                        "mail_lite": {"price": "$1 per user per month"}
                    }
                },
                "zoho_books": {
                    "name": "Zoho Books",
                    "description": "Online accounting software",
                    "pricing": {
                        "free": {"price": "$0", "limit": "Up to $50K revenue"},
                        "standard": {"price": "$15 per organization per month"}
                    }
                },
                "zoho_projects": {
                    "name": "Zoho Projects",
                    "description": "Project management software",
                    "pricing": {
                        "free": {"price": "$0", "users": "Up to 3 users"}
                    }
                }
            },
            "common_questions": {
                "free_tier": "Yes! Zoho offers generous free tiers for many products.",
                "integration": "Zoho integrates with Google Workspace, Microsoft 365, and many more.",
                "support": "24/7 support available through email, phone, and chat.",
                "data_security": "Enterprise-grade security with SOC 2, ISO 27001, GDPR compliance.",
                "migration": "Free migration assistance available from other platforms.",
                "customization": "Highly customizable with no-code tools.",
                "scalability": "Scales from solo entrepreneur to enterprise."
            }
        }


class ZohoSDRAssistant(Agent):
    def __init__(self, lead_data: LeadData, faq_engine: FAQSearchEngine) -> None:
        self.lead_data = lead_data
        self.faq_engine = faq_engine
        
        super().__init__(
            instructions="""You are Emma, a friendly and professional Sales Development Representative for Zoho.

**Your Role:**
- Warmly greet visitors and understand their business needs
- Answer questions about Zoho's products, pricing, and features using the FAQ information
- Naturally collect lead information during the conversation
- Maintain a conversational, helpful tone without being pushy

**Conversation Flow:**
1. Start with a warm greeting and ask what brought them to Zoho today
2. Listen to their needs and ask about their current challenges
3. Answer their questions about Zoho products using the FAQ
4. Naturally collect information: name, company, role, use case, team size, timeline
5. When they indicate they're done (say thanks, goodbye, that's all), provide a brief summary

**Important Guidelines:**
- Keep responses concise and natural (you're speaking, not writing)
- Don't use complex formatting, emojis, or asterisks in your speech
- Ask for information naturally, not like a form - weave it into conversation
- If you don't know something from the FAQ, be honest and offer to connect them with a specialist
- Focus on understanding their needs before pitching solutions
- Use the search_faq tool when users ask about products, pricing, or features
- Use the update_lead_field tool to store information as you learn it
- When the conversation ends, use the end_conversation tool to save everything

**Lead Information to Collect (naturally during conversation):**
- Name
- Company name
- Email address
- Role/title
- What they want to use Zoho for (use case)
- Team size
- Timeline (are they looking to start now, soon, or just exploring)

Remember: You're having a natural conversation, not conducting an interrogation. Be curious, helpful, and friendly!""",
        )
    
    @function_tool
    async def search_faq(self, context: RunContext, question: str):
        """Search the Zoho FAQ database for information about products, pricing, and features.
        
        Use this tool when the user asks about:
        - What Zoho products are available
        - Pricing information
        - Product features
        - Who Zoho is for
        - Free tiers or trials
        - Integrations
        - Security and compliance
        - Customization options
        - Migration assistance
        - Scalability
        
        Args:
            question: The user's question about Zoho products or services
        """
        logger.info(f"Searching FAQ for: {question}")
        answer = self.faq_engine.search(question)
        self.lead_data.conversation_summary.append(f"User asked: {question}")
        return answer
    
    @function_tool
    async def update_lead_field(
        self, 
        context: RunContext, 
        field: str, 
        value: str
    ):
        """Update a specific field in the lead data as you learn information during the conversation.
        
        Use this tool whenever the user shares information about themselves. Store it immediately.
        
        Args:
            field: The field to update. Must be one of: name, company, email, role, use_case, team_size, timeline
            value: The value to store for this field
        """
        valid_fields = ["name", "company", "email", "role", "use_case", "team_size", "timeline"]
        
        if field not in valid_fields:
            return f"Invalid field. Must be one of: {', '.join(valid_fields)}"
        
        setattr(self.lead_data, field, value)
        logger.info(f"Updated lead field {field}: {value}")
        
        return f"Got it! I've noted that your {field.replace('_', ' ')} is {value}."
    
    @function_tool
    async def end_conversation(self, context: RunContext):
        """End the conversation and save the lead data. Use this when the user indicates they're done.
        
        Signs the conversation is ending:
        - User says "thanks", "that's all", "goodbye", "I'm done"
        - User has no more questions
        - Natural end of conversation
        """
        logger.info("Ending conversation and saving lead data")
        
        # Save lead data to file
        filename = self.lead_data.save_to_file()
        
        # Create summary
        summary_parts = []
        if self.lead_data.name:
            summary_parts.append(f"{self.lead_data.name}")
        if self.lead_data.company:
            summary_parts.append(f"from {self.lead_data.company}")
        if self.lead_data.role:
            summary_parts.append(f"who is a {self.lead_data.role}")
        
        summary = " ".join(summary_parts) if summary_parts else "A potential customer"
        
        use_case_info = f" They're interested in using Zoho for {self.lead_data.use_case}." if self.lead_data.use_case else ""
        team_info = f" They have a team of {self.lead_data.team_size}." if self.lead_data.team_size else ""
        timeline_info = f" Their timeline is {self.lead_data.timeline}." if self.lead_data.timeline else ""
        
        full_summary = f"{summary}.{use_case_info}{team_info}{timeline_info}"
        
        return f"""Thank you so much for your time today! Let me quickly recap:

{full_summary}

I've saved all your information, and someone from our team will reach out to you soon to help you get started with Zoho. Have a wonderful day!

[Lead data saved to {filename}]"""


def prewarm(proc: JobProcess):
    """Prewarm function to load models and preprocess data before handling jobs"""
    logger.info("Starting prewarm process...")
    
    # Load VAD model
    logger.info("Loading VAD model...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD model loaded successfully")
    
    # Load and preprocess FAQ data
    logger.info("Loading FAQ data from zoho_faq.json...")
    faq_data = load_faq_data()
    
    # Initialize FAQ search engine with preprocessed data
    logger.info("Building FAQ search index...")
    faq_engine = FAQSearchEngine(faq_data)
    proc.userdata["faq_engine"] = faq_engine
    logger.info("FAQ search engine ready")
    
    logger.info("Prewarming complete - VAD and FAQ ready!")


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    # Initialize lead data for this conversation
    lead_data = LeadData()
    
    # Get preprocessed FAQ engine from prewarm
    faq_engine = ctx.proc.userdata["faq_engine"]
    logger.info("Using prewarmed FAQ search engine")
    
    # Set up voice AI pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    
    # Metrics collection
    usage_collector = metrics.UsageCollector()
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
        logger.info(f"Final lead data: {lead_data.to_dict()}")
    
    ctx.add_shutdown_callback(log_usage)
    
    # Start the session with our SDR agent
    await session.start(
        agent=ZohoSDRAssistant(lead_data, faq_engine),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    
    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))