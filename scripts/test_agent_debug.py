"""
Debug script to test single agent tool calling
"""
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from src.agents.single_agent import run_single_agent
from utils.logger import get_logger

log = get_logger(__name__)

def test_tool_call():
    """Test if the agent can correctly parse and call tools"""
    test_queries = [
        "What is 5 plus 3?",
        "Multiply 10 and 7",
        "Create a note saying 'Hello World'",
        "Add 100 and 50",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        try:
            result = run_single_agent(query)
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                print(f"Last message type: {type(last_msg).__name__}")
                print(f"Content: {last_msg.content[:100] if hasattr(last_msg, 'content') else 'N/A'}")
                if hasattr(last_msg, 'tool_calls'):
                    print(f"Tool calls: {last_msg.tool_calls}")
                print(f"Full message: {last_msg}")
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_tool_call()
