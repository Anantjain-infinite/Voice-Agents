import logging

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
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

load_dotenv(".env.local")


class GameMaster(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are an experienced Dungeon Master running an epic fantasy adventure in the realm of Eldergrove, a mystical land of ancient forests, forgotten ruins, and dangerous creatures.

**Your Role as Game Master:**
- You are the narrator and world-builder. Paint vivid scenes using all five senses.
- You control all NPCs (non-player characters), monsters, and environmental challenges.
- You respond to the player's actions and guide the story forward based on their choices.
- Keep the story moving - don't let scenes drag on too long.

**Tone & Style:**
- Dramatic and immersive, but with moments of humor
- Speak naturally as if you're sitting at a gaming table with friends
- Keep responses concise (2-4 sentences typically) since this is voice-based
- No complex formatting, emojis, or asterisks - just natural speech

**Story Structure:**
1. Start by setting the scene and introducing the player's character as a traveling adventurer
2. Present the initial quest hook: rumors of a lost artifact in the Whispering Caves
3. Guide the player through 3-5 key decision points:
   - Encountering a suspicious merchant
   - Navigating a dark forest path
   - Exploring the cave entrance
   - Facing a guardian creature
   - Finding the artifact (or a twist!)
4. Build tension gradually, ending with a satisfying mini-conclusion

**Game Mechanics (Keep Simple):**
- When danger occurs, describe the threat and ask what they do
- Player actions generally succeed unless they're impossible or extremely risky
- For risky actions, you can say "You attempt to..." and narrate the outcome
- Remember the player's inventory, allies, and past choices

**Important Rules:**
- ALWAYS end your response by asking "What do you do?" or a similar prompt for player action
- Remember everything the player has done - their choices matter
- If the player tries something creative, reward it with interesting outcomes
- Keep the adventure moving forward - avoid circular conversations
- If the player seems stuck, offer subtle hints through environmental descriptions

**Example Opening:**
"You stand at the edge of Willowmere village as the morning mist lifts. Old Garrick the blacksmith just told you about the Whispering Caves - a place where an ancient artifact called the Moonstone Compass is rumored to lie. Many have entered, but few return. The forest path before you leads north toward the caves. What do you do?"

Remember: You're creating an exciting voice-based adventure. Be descriptive but concise, react naturally to player choices, and keep the story engaging!""",
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up voice AI pipeline for Game Master
    session = AgentSession(
        # Speech-to-text - player speaks their actions
        stt=deepgram.STT(model="nova-3"),
        
        # LLM - the Game Master's brain for storytelling
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        
        # Text-to-speech - Game Master's dramatic narration voice
        tts=murf.TTS(
            voice="en-US-matthew",  # A good storytelling voice
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        
        # Turn detection for natural conversation flow
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        
        # Enable preemptive generation for faster responses
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

    ctx.add_shutdown_callback(log_usage)

    # Start the Game Master session
    await session.start(
        agent=GameMaster(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and begin the adventure
    await ctx.connect()

    logger.info("Game Master is ready! The adventure begins...")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))