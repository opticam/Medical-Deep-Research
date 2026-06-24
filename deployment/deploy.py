"""
Deploy Medical Deep Research Agent to Vertex AI Agent Engine
=============================================================
Run from the project root:
    python deployment/deploy.py

Prerequisites:
    gcloud auth application-default login
    gcloud config set project YOUR_PROJECT_ID
    pip install google-cloud-aiplatform[adk,agent_engines] google-adk httpx pydantic
"""

from __future__ import annotations

import os
import sys

import vertexai
from vertexai.preview import reasoning_engines

# ---------------------------------------------------------------------------
# Configuration — override with env vars or edit directly
# ---------------------------------------------------------------------------

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
STAGING_BUCKET = os.getenv(
    "STAGING_BUCKET", f"gs://{PROJECT_ID}-medical-research-staging"
)
DISPLAY_NAME = "Medical Deep Research Agent"
DESCRIPTION = (
    "Physician-grade multi-agent literature review system. "
    "Sources: PubMed/NCBI + FDA openFDA. "
    "Output: Structured clinical report with citations."
)

# ---------------------------------------------------------------------------
# Extra packages bundled into the Agent Engine deployment
# ---------------------------------------------------------------------------

EXTRA_PACKAGES = [
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
]


def deploy() -> None:
    """Package and deploy the agent to Vertex AI Agent Engine."""
    print(f"Initializing Vertex AI — project={PROJECT_ID}, location={LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    # Import here so local dev doesn't require vertexai installed
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    # Import the root agent
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.agent import root_agent

    print("Wrapping agent in AdkApp...")
    adk_app = reasoning_engines.AdkApp(
        agent=root_agent,
        enable_tracing=True,   # Cloud Trace integration
    )

    print(f"Deploying to Agent Engine (staging bucket: {STAGING_BUCKET})...")
    print("This may take 3-5 minutes...")

    remote_agent = reasoning_engines.ReasoningEngine.create(
        adk_app,
        display_name=DISPLAY_NAME,
        description=DESCRIPTION,
        requirements=EXTRA_PACKAGES,
        staging_bucket=STAGING_BUCKET,
        extra_packages=[],   # add local wheel files here if needed
    )

    resource_name = remote_agent.resource_name
    agent_engine_id = resource_name.split("/")[-1]

    print("\n" + "=" * 60)
    print("✅  DEPLOYMENT SUCCESSFUL")
    print("=" * 60)
    print(f"Resource name : {resource_name}")
    print(f"Agent Engine ID: {agent_engine_id}")
    print(f"\nTo query the deployed agent:")
    print(f"  python deployment/query_deployed.py --agent-id {agent_engine_id}")
    print("=" * 60)

    # Save the agent ID for convenience
    with open(".deployed_agent_id", "w") as f:
        f.write(agent_engine_id)
    print(f"\nAgent ID saved to .deployed_agent_id")


if __name__ == "__main__":
    deploy()
