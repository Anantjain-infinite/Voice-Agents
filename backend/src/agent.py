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
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from database import get_fraud_case_by_username, update_fraud_case

logger = logging.getLogger("agent")
load_dotenv(".env.local")


class FraudAlertAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a fraud detection representative from SecureBank's Fraud Prevention Department. 
            You are calling customers about suspicious transactions detected on their accounts.
            
            Your conversation flow:
            1. Greet the customer warmly and professionally
            2. Introduce yourself as calling from SecureBank Fraud Department
            3. Ask for their name to look up their case
            4. Once you have the fraud case details, verify their identity using the security question
            5. If verification passes, read out the suspicious transaction details clearly
            6. Ask if they authorized this transaction (yes or no)
            7. Based on their answer, mark the case appropriately and explain next steps
            8. Thank them and end the call
            
            IMPORTANT GUIDELINES:
            - Be calm, professional, and reassuring
            - Speak clearly and not too fast
            - Never ask for full card numbers, PINs, or passwords
            - Use only the security question from the database for verification
            - Keep responses concise and conversational
            - Do not use complex formatting, emojis, or asterisks
            - After getting the fraud case, always verify identity before proceeding
            - Make sure to call the appropriate tools to load and update fraud cases
            
            Your goal is to efficiently verify whether the transaction is legitimate or fraudulent.""",
        )
        self.current_fraud_case = None
        self.verification_passed = False

    @function_tool
    async def load_fraud_case(self, context: RunContext, username: str):
        """Load a pending fraud case for the given username from the database.
        
        This tool retrieves fraud case details including transaction information and security questions.
        Call this tool after getting the customer's name.
        
        Args:
            username: The customer's name to look up in the fraud cases database
        """
        logger.info(f"Loading fraud case for username: {username}")
        
        fraud_case = get_fraud_case_by_username(username)
        
        if fraud_case:
            self.current_fraud_case = fraud_case
            logger.info(f"Fraud case loaded: {fraud_case}")
            
            return f"""Fraud case found for {username}. 
            Transaction Details:
            - Card ending: {fraud_case['cardEnding']}
            - Merchant: {fraud_case['transactionName']}
            - Amount: ${fraud_case['transactionAmount']:.2f}
            - Time: {fraud_case['transactionTime']}
            - Location: {fraud_case['transactionLocation']}
            - Category: {fraud_case['transactionCategory']}
            - Source: {fraud_case['transactionSource']}
            
            Security Question: {fraud_case['securityQuestion']}
            
            Now verify the customer's identity using the security question before discussing the transaction."""
        else:
            return f"No pending fraud cases found for username '{username}'. Please verify the name or end the call politely."

    @function_tool
    async def verify_customer_identity(self, context: RunContext, customer_answer: str):
        """Verify the customer's identity by checking their answer to the security question.
        
        Compare the customer's answer with the expected answer from the fraud case.
        Call this tool after asking the security question.
        
        Args:
            customer_answer: The customer's answer to the security question
        """
        if not self.current_fraud_case:
            return "No fraud case loaded. Please load a fraud case first."
        
        expected_answer = self.current_fraud_case['securityAnswer'].lower().strip()
        provided_answer = customer_answer.lower().strip()
        
        logger.info(f"Verifying identity - Expected: {expected_answer}, Provided: {provided_answer}")
        
        if provided_answer == expected_answer:
            self.verification_passed = True
            return "Identity verification PASSED. You may now discuss the suspicious transaction with the customer."
        else:
            self.verification_passed = False
            return "Identity verification FAILED. Politely inform the customer that you cannot proceed and end the call for security reasons."

    @function_tool
    async def mark_transaction_safe(self, context: RunContext):
        """Mark the transaction as safe when the customer confirms they made the purchase.
        
        Call this tool when the customer says YES, they authorized the transaction.
        This updates the database and closes the fraud case.
        """
        if not self.current_fraud_case:
            return "No fraud case loaded."
        
        if not self.verification_passed:
            return "Customer identity not verified. Cannot update case."
        
        case_id = self.current_fraud_case['id']
        outcome = f"Customer {self.current_fraud_case['userName']} confirmed transaction as legitimate."
        
        update_fraud_case(case_id, "confirmed_safe", outcome)
        logger.info(f"Transaction marked as safe: {outcome}")
        
        return """Transaction marked as SAFE. 
        Tell the customer: 'Thank you for confirming. We've marked this transaction as authorized. 
        No further action is needed. Your card remains active. Have a great day!'"""

    @function_tool
    async def mark_transaction_fraudulent(self, context: RunContext):
        """Mark the transaction as fraudulent when the customer denies making the purchase.
        
        Call this tool when the customer says NO, they did not authorize the transaction.
        This updates the database, blocks the card, and initiates a dispute.
        """
        if not self.current_fraud_case:
            return "No fraud case loaded."
        
        if not self.verification_passed:
            return "Customer identity not verified. Cannot update case."
        
        case_id = self.current_fraud_case['id']
        card_ending = self.current_fraud_case['cardEnding']
        outcome = f"Customer {self.current_fraud_case['userName']} denied making the transaction. Card ending {card_ending} blocked. Dispute initiated."
        
        update_fraud_case(case_id, "confirmed_fraud", outcome)
        logger.info(f"Transaction marked as fraudulent: {outcome}")
        
        return f"""Transaction marked as FRAUDULENT. 
        Tell the customer: 'I understand. For your security, we have immediately blocked your card ending in {card_ending}. 
        We will reverse this charge and send you a new card within 5-7 business days. 
        A fraud dispute has been opened. Is there anything else I can help you with today?'"""


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

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
        agent=FraudAlertAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))