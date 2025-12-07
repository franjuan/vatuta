"""
Small Bedrock tool: invoke a model with auth from env (.env loaded).

Auth precedence:
- AWS_BEARER_TOKEN_BEDROCK (bearer token via botocore aws_bearer_token)
- AWS_PROFILE (boto3 session profile)
- Default credential chain (env/role)

Environment variables expected:
- AWS_BEARER_TOKEN_BEDROCK: Optional bearer token for Bedrock
- AWS_PROFILE: Optional profile name
- AWS_REGION: Region for Bedrock (default: us-east-1)
"""

import argparse
import logging
import os
from typing import Optional

import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


def _get_region() -> str:
    region_env = os.getenv("AWS_REGION")
    if region_env:
        logger.info("Using region from AWS_REGION: %s", region_env)
        return region_env
    logger.info("No region set in env; defaulting to us-east-1")
    return "us-east-1"


def _build_bedrock_client() -> any:
    """
    Build a Bedrock Runtime client honoring bearer token and profile from env.
    """
    logger.info("Starting Bedrock client construction")
    region = _get_region()
    profile = os.getenv("AWS_PROFILE")
    bearer = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

    logger.debug(
        "Auth env detection: AWS_PROFILE=%s, AWS_BEARER_TOKEN_BEDROCK present=%s",
        profile if profile else "(none)",
        "yes" if bearer else "no",
    )

    # Session creation
    try:
        if bearer:
            logger.info("Creating boto3 session with default credential chain")
            session = boto3.Session()
        else:
            logger.info("Creating boto3 session with AWS profile: %s", profile)
            session = boto3.Session(profile_name=profile)
    except Exception as e:
        logger.error("Failed to create boto3 session: %s", e)
        raise

    # Client creation
    try:
        if bearer:
            logger.info("Using bearer token auth for Bedrock runtime; region=%s", region)
        else:
            logger.info("Using non-bearer auth (profile/default chain); region=%s", region)
        client = session.client("bedrock-runtime", region_name=region)
        logger.info("Bedrock client created successfully")
        return client
    except Exception as e:
        logger.error("Failed to create Bedrock client: %s", e)
        raise


def invoke_bedrock(
    model_id: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.9,
) -> str:
    """
    Invoke a Bedrock chat model with a user prompt and optional system prompt.
    """
    logger.info("Preparing invocation: model_id=%s", model_id)
    logger.debug(
        "Generation params: temperature=%.3f, max_tokens=%s, top_p=%.3f",
        float(temperature),
        int(max_tokens),
        float(top_p),
    )

    client = _build_bedrock_client()

    llm = ChatBedrock(
        model_id=model_id,
        client=client,
        model_kwargs={
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
        },
    )

    messages = []
    if system_prompt:
        logger.info("System prompt provided: yes")
        messages.append(SystemMessage(content=system_prompt))
    else:
        logger.info("System prompt provided: no")
    messages.append(HumanMessage(content=prompt))

    logger.info("Invoking model...")
    response = llm.invoke(messages)
    content = getattr(response, "content", str(response))
    logger.info(
        "Model response received (%d chars)",
        len(content) if isinstance(content, str) else 0,
    )
    return content


def main():
    parser = argparse.ArgumentParser(description="Invoke an AWS Bedrock model using env-based auth")
    parser.add_argument(
        "prompt",
        help="User prompt to send to the model",
    )
    parser.add_argument(
        "--model-id",
        default="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        help="Bedrock model ID",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top_p",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    try:
        # Configure logging
        log_level = logging.DEBUG if args.verbose else os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logger.info("Log level set to %s", logging.getLevelName(logging.getLogger().level))

        output = invoke_bedrock(
            model_id=args.model_id,
            prompt=args.prompt,
            system_prompt=args.system,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
        print(output)
    except Exception as e:
        # Keep simple and concise output for CLI usage
        raise SystemExit(f"Bedrock invocation failed: {e}")


if __name__ == "__main__":
    main()
