import logging
import json
from datetime import datetime
from pathlib import Path
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

# Load catalog
CATALOG_PATH = Path("catalog.json")
ORDERS_DIR = Path("orders")
ORDERS_DIR.mkdir(exist_ok=True)

def load_catalog():
    """Load the product catalog from JSON file"""
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, 'r') as f:
            return json.load(f)
    return {"items": []}

CATALOG = load_catalog()

# Recipe mappings for intelligent ingredient bundling
RECIPES = {
    "peanut butter sandwich": ["bread", "peanut butter"],
    "pbj sandwich": ["bread", "peanut butter", "jam"],
    "pasta": ["pasta", "pasta sauce", "olive oil"],
    "spaghetti": ["spaghetti", "pasta sauce", "olive oil"],
    "breakfast": ["eggs", "bread", "milk", "butter"],
    "cereal": ["cereal", "milk"],
    "sandwich": ["bread", "cheese", "lettuce", "tomato"],
    "salad": ["lettuce", "tomato", "cucumber", "olive oil"],
    "smoothie": ["banana", "yogurt", "berries"],
    "omelette": ["eggs", "cheese", "butter"],
}


class ShoppingCart:
    """Manages the shopping cart state"""
    def __init__(self):
        self.items = []
    
    def add_item(self, item_name: str, quantity: int = 1, notes: str = ""):
        """Add item to cart"""
        # Check if item exists in catalog
        catalog_item = next((i for i in CATALOG.get("items", []) 
                           if i["name"].lower() == item_name.lower()), None)
        
        if not catalog_item:
            return False, f"Item '{item_name}' not found in catalog"
        
        # Check if item already in cart
        existing = next((i for i in self.items 
                        if i["name"].lower() == item_name.lower()), None)
        
        if existing:
            existing["quantity"] += quantity
            return True, f"Updated {item_name} quantity to {existing['quantity']}"
        else:
            self.items.append({
                "name": catalog_item["name"],
                "quantity": quantity,
                "price": catalog_item["price"],
                "category": catalog_item["category"],
                "notes": notes
            })
            return True, f"Added {quantity} {item_name} to cart"
    
    def remove_item(self, item_name: str):
        """Remove item from cart"""
        self.items = [i for i in self.items 
                     if i["name"].lower() != item_name.lower()]
        return True, f"Removed {item_name} from cart"
    
    def update_quantity(self, item_name: str, quantity: int):
        """Update item quantity"""
        item = next((i for i in self.items 
                    if i["name"].lower() == item_name.lower()), None)
        if item:
            if quantity <= 0:
                return self.remove_item(item_name)
            item["quantity"] = quantity
            return True, f"Updated {item_name} quantity to {quantity}"
        return False, f"Item '{item_name}' not in cart"
    
    def get_cart_summary(self):
        """Get formatted cart summary"""
        if not self.items:
            return "Your cart is empty."
        
        summary = "Your cart contains:\n"
        total = 0
        for item in self.items:
            item_total = item["price"] * item["quantity"]
            total += item_total
            summary += f"- {item['quantity']}x {item['name']} (${item['price']:.2f} each) = ${item_total:.2f}\n"
        summary += f"\nTotal: ${total:.2f}"
        return summary
    
    def get_total(self):
        """Calculate cart total"""
        return sum(item["price"] * item["quantity"] for item in self.items)
    
    def clear(self):
        """Clear the cart"""
        self.items = []


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly voice assistant for FreshCart, a food and grocery ordering service.
            
            Your role:
            - Help users browse and order items from our catalog
            - Add items to their shopping cart with quantities
            - Handle intelligent requests like "ingredients for pasta" by adding all needed items
            - Keep track of what's in their cart
            - Place orders when they're ready
            
            Guidelines:
            - Be conversational and helpful
            - Confirm each action clearly (what was added, removed, or updated)
            - Ask for clarification on quantities, sizes, or preferences when needed
            - When users ask for "ingredients for X", add all relevant items intelligently
            - Offer to show cart contents periodically
            - Keep responses concise for voice interaction
            - When placing an order, summarize the cart and total before confirming
            
            Available tools:
            - add_to_cart: Add items with quantity
            - remove_from_cart: Remove items
            - update_cart_quantity: Change item quantities
            - view_cart: Show current cart contents
            - search_catalog: Find items in catalog
            - add_recipe_ingredients: Add all ingredients for a dish
            - place_order: Complete the order
            
            Do not use complex formatting, emojis, or symbols in your responses.""",
        )
        self.cart = ShoppingCart()

    @function_tool
    async def search_catalog(self, context: RunContext, query: str):
        """Search for items in the catalog by name or category.
        
        Args:
            query: Search term (item name, category, or keyword)
        """
        query_lower = query.lower()
        results = [
            item for item in CATALOG.get("items", [])
            if query_lower in item["name"].lower() 
            or query_lower in item["category"].lower()
            or query_lower in item.get("tags", [])
        ]
        
        if not results:
            return f"No items found matching '{query}'"
        
        response = f"Found {len(results)} items:\n"
        for item in results[:10]:  # Limit to 10 results
            response += f"- {item['name']} (${item['price']:.2f}) - {item['category']}\n"
        
        return response

    @function_tool
    async def add_to_cart(self, context: RunContext, item_name: str, quantity: int = 1):
        """Add an item to the shopping cart.
        
        Args:
            item_name: Name of the item to add
            quantity: Quantity to add (default: 1)
        """
        success, message = self.cart.add_item(item_name, quantity)
        logger.info(f"Add to cart: {item_name} x{quantity} - {message}")
        return message

    @function_tool
    async def remove_from_cart(self, context: RunContext, item_name: str):
        """Remove an item from the shopping cart.
        
        Args:
            item_name: Name of the item to remove
        """
        success, message = self.cart.remove_item(item_name)
        logger.info(f"Remove from cart: {item_name} - {message}")
        return message

    @function_tool
    async def update_cart_quantity(self, context: RunContext, item_name: str, quantity: int):
        """Update the quantity of an item in the cart.
        
        Args:
            item_name: Name of the item
            quantity: New quantity (use 0 to remove)
        """
        success, message = self.cart.update_quantity(item_name, quantity)
        logger.info(f"Update quantity: {item_name} to {quantity} - {message}")
        return message

    @function_tool
    async def view_cart(self, context: RunContext):
        """View all items currently in the shopping cart with prices and total."""
        summary = self.cart.get_cart_summary()
        logger.info(f"Cart viewed: {len(self.cart.items)} items")
        return summary

    @function_tool
    async def add_recipe_ingredients(self, context: RunContext, recipe_name: str):
        """Add all ingredients needed for a specific dish or recipe.
        
        Args:
            recipe_name: Name of the dish (e.g., "pasta", "sandwich", "breakfast")
        """
        recipe_lower = recipe_name.lower()
        
        # Find matching recipe
        ingredients = None
        for recipe, items in RECIPES.items():
            if recipe_lower in recipe or recipe in recipe_lower:
                ingredients = items
                break
        
        if not ingredients:
            return f"I don't have a recipe for '{recipe_name}'. Try searching the catalog or adding items individually."
        
        added = []
        for ingredient in ingredients:
            success, message = self.cart.add_item(ingredient, 1)
            if success:
                added.append(ingredient)
        
        if added:
            items_list = ", ".join(added)
            logger.info(f"Added recipe ingredients for {recipe_name}: {items_list}")
            return f"I've added ingredients for {recipe_name}: {items_list}"
        else:
            return f"Couldn't find some ingredients for {recipe_name} in the catalog"

    @function_tool
    async def place_order(self, context: RunContext, customer_name: str = "Guest"):
        """Place the order and save it to a file.
        
        Args:
            customer_name: Name for the order (optional)
        """
        if not self.cart.items:
            return "Your cart is empty. Add some items before placing an order."
        
        # Create order object
        order = {
            "order_id": f"ORD-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "customer_name": customer_name,
            "timestamp": datetime.now().isoformat(),
            "items": self.cart.items,
            "total": self.cart.get_total(),
            "status": "placed"
        }
        
        # Save to JSON file
        order_file = ORDERS_DIR / f"{order['order_id']}.json"
        with open(order_file, 'w') as f:
            json.dump(order, f, indent=2)
        
        logger.info(f"Order placed: {order['order_id']} - ${order['total']:.2f}")
        
        # Clear cart
        self.cart.clear()
        
        return f"Order {order['order_id']} has been placed successfully! Total: ${order['total']:.2f}. Your order has been saved and will be processed shortly."


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
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))