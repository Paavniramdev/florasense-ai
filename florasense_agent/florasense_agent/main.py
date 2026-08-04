"""
main.py — CLI to run the FloraSense agent end to end.

Examples:
    python main.py "What pollinators visit sunflowers?"
    python main.py "What's wrong with my plant?" --image path/to/flower.jpg
    python main.py "Can this grow in Punjab right now?" --image flower.jpg --location "Patiala, Punjab"
"""
import argparse
import os

from langchain_core.messages import HumanMessage

from graph import build_agent


def extract_text(content) -> str:
    """LangChain message content is a plain string for some providers, but a
    list of content blocks (e.g. [{'type': 'text', 'text': '...'}]) for others
    — Gemini included. This normalizes either case to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def run(question: str, image_path: str = None, location: str = None, verbose: bool = True):
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit(
            "GOOGLE_API_KEY is not set. Get a free key (no credit card) at "
            "https://aistudio.google.com/apikey, then set it with:\n"
            "  export GOOGLE_API_KEY=your-key-here   (Mac/Linux)\n"
            "  set GOOGLE_API_KEY=your-key-here       (Windows cmd)"
        )

    agent = build_agent()

    prompt_parts = [question]
    if image_path:
        prompt_parts.append(f"[Attached image file path: {image_path}]")
    if location:
        prompt_parts.append(f"[User's location: {location}]")
    user_message = HumanMessage(content="\n".join(prompt_parts))

    final_state = None
    for step in agent.stream({"messages": [user_message]}, stream_mode="values"):
        final_state = step
        if verbose:
            last = step["messages"][-1]
            tool_calls = getattr(last, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    print(f"\n🔧 calling tool: {tc['name']}({tc['args']})")
            elif last.type == "tool":
                preview = extract_text(last.content)[:300]
                print(f"   ↳ result: {preview}{'...' if len(extract_text(last.content)) > 300 else ''}")

    final_answer = extract_text(final_state["messages"][-1].content)
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(final_answer)
    return final_answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--image", default=None, help="Path to a flower image")
    parser.add_argument("--location", default=None, help="e.g. 'Patiala, Punjab'")
    parser.add_argument("--quiet", action="store_true", help="Only print the final answer")
    args = parser.parse_args()

    run(args.question, args.image, args.location, verbose=not args.quiet)
