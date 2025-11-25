import logging
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

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

logger = logging.getLogger("teach-the-tutor")
logger.setLevel(logging.INFO)

load_dotenv(".env.local")

COURSE_CONTENT_PATH = "shared-data/day4_tutor_content.json"


def load_course_content() -> List[Dict[str, Any]]:
    """Load the tutor content JSON. If missing, fall back to default."""
    default_content = [
        {
            "id": "variables",
            "title": "Variables",
            "summary": "Variables store values so you can reuse them later. In most languages, a variable has a name and holds a value in memory. You can read or update this value throughout your program.",
            "sample_question": "What is a variable and why is it useful?"
        },
        {
            "id": "loops",
            "title": "Loops",
            "summary": "Loops let you repeat an action multiple times without writing the same code again. They usually run while a condition is true or over a sequence of items.",
            "sample_question": "Explain the difference between a for loop and a while loop."
        }
    ]

    try:
        with open(COURSE_CONTENT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list) and data:
                logger.info(f"Loaded course content from {COURSE_CONTENT_PATH}")
                return data
            else:
                logger.warning("Course content JSON was empty or invalid, using default.")
                return default_content
    except FileNotFoundError:
        logger.warning(f"{COURSE_CONTENT_PATH} not found, using default content.")
        return default_content
    except Exception as e:
        logger.error(f"Error loading course content: {e}")
        return default_content


@dataclass
class UserData:
    """Shared data across the whole Teach-the-Tutor session."""
    personas: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None
    current_mode: str = "learn"  # "learn", "quiz", "teach_back"
    current_concept_id: str = "variables"
    course_content: List[Dict[str, Any]] = field(default_factory=load_course_content)

    def get_concept(self, concept_id: Optional[str] = None) -> Dict[str, Any]:
        if concept_id is None:
            concept_id = self.current_concept_id

        for c in self.course_content:
            if c.get("id") == concept_id:
                return c

        # fallback to first concept
        return self.course_content[0]

    def summarize(self) -> str:
        titles = ", ".join(
            f"{c.get('id')} ({c.get('title')})" for c in self.course_content
        )
        return (
            "You are part of a Teach-the-Tutor active recall coach. "
            f"Available concepts: {titles}. "
            f"Current mode: {self.current_mode}. "
            f"Current concept id: {self.current_concept_id}."
        )


RunContext_T = RunContext[UserData]


class BaseTutorAgent(Agent):
    """Base agent that handles context sharing + mode switching."""

    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"Entering {agent_name}")

        userdata: UserData = self.session.userdata

        # Copy the chat context for this agent
        chat_ctx = self.chat_ctx.copy()

        # Preserve reasonable amount of history from the previous agent
        if userdata.prev_agent:
            items_copy = self._truncate_chat_ctx(
                userdata.prev_agent.chat_ctx.items, keep_function_call=True
            )
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in items_copy if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # System message describing the multi-agent + mode setup
        chat_ctx.add_message(
            role="system",
            content=(
                f"You are the {agent_name} in a three-mode Teach-the-Tutor system.\n"
                f"{userdata.summarize()}\n\n"
                "COURSE CONTENT (JSON):\n"
                f"{json.dumps(userdata.course_content)}\n\n"
                "Rules:\n"
                "- Always stay focused on the currently selected concept.\n"
                "- In learn mode: explain the concept using its 'summary', give simple examples, then ask a quick check question.\n"
                "- In quiz mode: ask short, targeted questions, primarily based on 'sample_question' and small variations. Wait for the user's answer.\n"
                "- In teach_back mode: ask the user to explain the concept back in their own words, then give brief qualitative feedback comparing their answer to the 'summary'.\n"
                "- The user can say things like 'switch to quiz mode', 'I want to teach it back', or 'go to learn mode'. When they do, call the appropriate mode-switch function tool.\n"
                "- If the user mentions another concept id (like 'variables' or 'loops'), update the current concept and then continue in the current mode.\n"
                "- Keep responses short, conversational, and clear, without formatting symbols or emojis."
            ),
        )

        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply()

    def _truncate_chat_ctx(
        self,
        items: list,
        keep_last_n_messages: int = 6,
        keep_system_message: bool = False,
        keep_function_call: bool = False,
    ) -> list:
        """Truncate the chat context to keep the last n messages."""
        def _valid_item(item) -> bool:
            if not keep_system_message and item.type == "message" and item.role == "system":
                return False
            if not keep_function_call and item.type in ["function_call", "function_call_output"]:
                return False
            return True

        new_items = []
        for item in reversed(items):
            if _valid_item(item):
                new_items.append(item)
            if len(new_items) >= keep_last_n_messages:
                break
        new_items = new_items[::-1]

        while new_items and new_items[0].type in ["function_call", "function_call_output"]:
            new_items.pop(0)

        return new_items

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> Agent:
        """Transfer to another mode-agent while preserving context."""
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.personas[name]

        userdata.prev_agent = current_agent
        userdata.current_mode = name

        return next_agent

    # ========= Shared function tools across all modes =========

    @function_tool
    async def switch_to_learn_mode(self, context: RunContext_T) -> Agent:
        """Switch to Learn mode (Matthew voice) for concept explanations."""
        await self.session.say("Switching to learn mode so I can explain the concept step by step.")
        return await self._transfer_to_agent("learn", context)

    @function_tool
    async def switch_to_quiz_mode(self, context: RunContext_T) -> Agent:
        """Switch to Quiz mode (Alicia voice) for active recall questions."""
        await self.session.say("Switching to quiz mode so I can ask you questions.")
        return await self._transfer_to_agent("quiz", context)

    @function_tool
    async def switch_to_teach_back_mode(self, context: RunContext_T) -> Agent:
        """Switch to Teach-back mode (Ken voice) to have the user explain the concept."""
        await self.session.say("Switching to teach-back mode so you can explain the concept to me.")
        return await self._transfer_to_agent("teach_back", context)

    @function_tool
    async def set_concept(self, context: RunContext_T, concept_id: str) -> str:
        """Set the current concept by its id (e.g., 'variables', 'loops')."""
        userdata = context.userdata
        concepts = [c.get("id") for c in userdata.course_content]
        if concept_id not in concepts:
            return (
                f"I couldn't find the concept '{concept_id}'. "
                f"Available concepts are: {', '.join(concepts)}."
            )

        userdata.current_concept_id = concept_id
        concept = userdata.get_concept(concept_id)
        return (
            f"Got it! We'll focus on {concept.get('title')} now. "
            f"You can keep using the current mode, or ask to switch modes at any time."
        )


# ======================= Mode Agents =======================

class LearnAgent(BaseTutorAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the Learn-mode tutor. Your job is to clearly explain the current concept, "
                "using the summary from the course content, with gentle step-by-step language and simple examples. "
                "After explaining, ask the learner one short check question based on the concept."
            ),
            stt=deepgram.STT(model="nova-3"),
            llm=google.LLM(model="gemini-2.5-flash"),
            tts=murf.TTS(
                voice="en-US-matthew",  # Murf Falcon Voice - Matthew
                style="Conversation",
            ),
            vad=silero.VAD.load(),
        )


class QuizAgent(BaseTutorAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the Quiz-mode tutor. Your job is to ask active recall questions about the current concept. "
                "Base your questions on the 'sample_question' and small variations. "
                "Ask one question at a time and wait for the learner's answer. Do not immediately explain the full concept."
            ),
            stt=deepgram.STT(model="nova-3"),
            llm=google.LLM(model="gemini-2.5-flash"),
            tts=murf.TTS(
                voice="en-US-alicia",  # Murf Falcon Voice - Alicia
                style="Conversation",
            ),
            vad=silero.VAD.load(),
        )


class TeachBackAgent(BaseTutorAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the Teach-back-mode tutor. Your job is to ask the learner to explain the current concept "
                "in their own words. Listen to their explanation, compare it to the summary from the course content, "
                "and then provide brief qualitative feedback: what they got right, what was missing, and a quick correction."
            ),
            stt=deepgram.STT(model="nova-3"),
            llm=google.LLM(model="gemini-2.5-flash"),
            tts=murf.TTS(
                voice="en-US-ken",  # Murf Falcon Voice - Ken
                style="Conversation",
            ),
            vad=silero.VAD.load(),
        )


# ======================= Entrypoint =======================

async def entrypoint(ctx: JobContext):
    """
    Entry for the Day 4 Teach-the-Tutor system.

    Flow:
    - Start in Learn mode (Matthew), greet the user, and ask which mode they want.
    - The user can say 'learn', 'quiz', or 'teach back', and the LLM will call the
      appropriate switch_* tool to hand off to the right agent.
    - Mode switching keeps conversation context using the shared UserData + chat history.
    """

    userdata = UserData()

    learn_agent = LearnAgent()
    quiz_agent = QuizAgent()
    teach_back_agent = TeachBackAgent()

    userdata.personas.update(
        {
            "learn": learn_agent,
            "quiz": quiz_agent,
            "teach_back": teach_back_agent,
        }
    )

    session = AgentSession[UserData](userdata=userdata)

    # Start in Learn mode, but the very first thing LearnAgent should do is:
    # - greet the user
    # - explain the three modes (learn, quiz, teach_back)
    # - ask which one they prefer


    # Metrics collection
    usage_collector = metrics.UsageCollector() 
    @session.on("metrics_collected") 
    def _on_metrics_collected(ev: MetricsCollectedEvent): 
        metrics.log_metrics(ev.metrics) 
        usage_collector.collect(ev.metrics) 
    async def log_usage(): 
        summary = usage_collector.get_summary() 
        logger.info(f"Usage: {summary}") 
    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=learn_agent,
        room=ctx.room,
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
