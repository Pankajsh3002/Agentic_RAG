import os


def load_secrets() -> None:
    """On AWS, read secrets from Parameter Store into environment variables, once per container."""
    if os.getenv("_SECRETS_LOADED"):
        return
    param = os.getenv("OPENROUTER_API_KEY_PARAM")
    if param and not os.getenv("OPENROUTER_API_KEY"):
        import boto3
        value = boto3.client("ssm").get_parameter(Name=param, WithDecryption=True)["Parameter"]["Value"]
        os.environ["OPENROUTER_API_KEY"] = value
    os.environ["_SECRETS_LOADED"] = "1"