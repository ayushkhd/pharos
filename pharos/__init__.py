"""Pharos: monitor coding-agent trajectories with verifiable evidence.

The package has two layers joined by one contract:

* Deterministic infrastructure (``data``, ``render``, ``validate``, ``store``,
  ``runner``, ``compare``, ``metrics``, ``report``) that never trusts model
  output.
* Pluggable inference ``providers`` that turn a prompt plus a JSON schema
  into a raw response and know nothing about trajectories.
"""

__version__ = "0.1.0"
