"""Gateway-level HTTP middleware.

Keep these thin — anything that touches request semantics belongs in
a dedicated service, not a global middleware, so tests can exercise
it in isolation.
"""
