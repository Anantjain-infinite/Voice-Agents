import logging
import json
from datetime import datetime
from typing import Optional

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


class BaristaAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly and enthusiastic barista at Brew & Bean Coffee Shop. The user is interacting with you via voice.
            
            Your job is to take coffee orders and gather all necessary information:
            - Drink type (e.g., latte, cappuccino, espresso, americano, cold brew, mocha)
            - Size (small, medium, large)
            - Milk preference (whole, skim, oat, almond, soy, or none)
            - Extras (e.g., extra shot, vanilla syrup, caramel drizzle, whipped cream)
            - Customer name for the order
            
            Be conversational and warm. Ask clarifying questions one at a time to gather missing information.
            Suggest popular drinks if the customer seems unsure.
            Once you have all the information, confirm the complete order with the customer.
            When the customer confirms, use the save_order tool to save their order.
            
            Keep responses concise and natural, without complex formatting, emojis, or asterisks.
            Be helpful and make the ordering experience delightful.""",
        )
        
        # Initialize order state
        self.order_state = {
            "drinkType": None,
            "size": None,
            "milk": None,
            "extras": [],
            "name": None
        }

    @function_tool
    async def save_order(self, context: RunContext, 
                        drink_type: str, 
                        size: str, 
                        milk: str, 
                        extras: str,
                        name: str):
        """Save the completed coffee order to a JSON file.
        
        Use this tool ONLY when you have confirmed all order details with the customer
        and they have approved the complete order.
        
        Args:
            drink_type: The type of drink (e.g., latte, cappuccino, mocha)
            size: The size of the drink (small, medium, or large)
            milk: The milk preference (whole, skim, oat, almond, soy, or none)
            extras: Comma-separated list of extras (e.g., "extra shot, vanilla syrup" or "none")
            name: Customer's name for the order
        """
        
        # Parse extras into a list
        extras_list = [e.strip() for e in extras.split(",")] if extras.lower() != "none" else []
        
        # Create new order
        order = {
            "drinkType": drink_type,
            "size": size,
            "milk": milk,
            "extras": extras_list,
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "status": "confirmed"
        }
        
        # Save to single JSON file (append to existing orders)
        filename = "orders.json"
        
        try:
            # Read existing orders
            try:
                with open(filename, 'r') as f:
                    orders = json.load(f)
            except FileNotFoundError:
                # File doesn't exist yet, start with empty list
                orders = []
            
            # Append new order
            orders.append(order)
            
            # Write all orders back to file
            with open(filename, 'w') as f:
                json.dump(orders, f, indent=2)
            
            logger.info(f"Order saved successfully to {filename}")
            logger.info(f"Order details: {json.dumps(order, indent=2)}")
            
            return f"Perfect! Your order has been saved. Your {size} {drink_type} with {milk} milk will be ready soon, {name}. Thanks for choosing Brew & Bean!"
            
        except Exception as e:
            logger.error(f"Error saving order: {e}")
            return f"I've noted your order, but there was a technical issue saving it. Don't worry, I'll make sure your {size} {drink_type} gets made!"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

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

    ctx.add_shutdown_callback(log_usage)

    # Start the session with the barista agent
    await session.start(
        agent=BaristaAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))