"""
Query the Deployed Medical Deep Research Agent
================================================
Usage:
    python deployment/query_deployed.py --agent-id AGENT_ENGINE_ID

Or set AGENT_ENGINE_ID env var.
"""

from __future__ import annotations

import argparse
import os
import sys

import vertexai
from vertexai.preview import reasoning_engines

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")


def get_agent_id() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", default=os.getenv("AGENT_ENGINE_ID", ""))
    args = parser.parse_args()

    agent_id = args.agent_id
    if not agent_id:
        # Try reading from file written by deploy.py
        try:
            with open(".deployed_agent_id") as f:
                agent_id = f.read().strip()
        except FileNotFoundError:
            pass

    if not agent_id:
        print("Error: Provide --agent-id or set AGENT_ENGINE_ID env var.")
        sys.exit(1)

    return agent_id


def main() -> None:
    agent_id = get_agent_id()

    vertexai.init(project=PROJECT_ID, location=LOCATION)

    print(f"Connecting to Agent Engine: {agent_id}")
    remote_agent = reasoning_engines.ReasoningEngine(
        f"projects/{PROJECT_ID}/locations/{LOCATION}/reasoningEngines/{agent_id}"
    )

    # Create a session
    session = remote_agent.create_session(user_id="physician-demo")
    session_id = session["id"]
    print(f"Session created: {session_id}")
    print("\n" + "=" * 60)
    print("Medical Deep Research Agent — Interactive Mode")
    print("Type your research topic or question. Ctrl+C to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            print("\nAgent: ", end="", flush=True)
            for event in remote_agent.stream_query(
                user_id="physician-demo",
                session_id=session_id,
                message=user_input,
            ):
                # Stream text chunks as they arrive
                for part in event.get("content", {}).get("parts", []):
                    if text := part.get("text"):
                        print(text, end="", flush=True)
            print("\n")

        except KeyboardInterrupt:
            print("\nExiting. Session preserved — resume with session_id:", session_id)
            break


if __name__ == "__main__":
    main()
