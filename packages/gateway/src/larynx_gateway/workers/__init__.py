"""In-process background workers.

Distinct from ``packages/<name>_worker`` which are standalone model
servers — these are asyncio tasks that live inside the gateway
process and share its lifespan.
"""
