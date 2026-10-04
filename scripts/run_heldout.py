"""Protocol freeze/preflight; held-out inference requires an explicit execution flag and hash."""

from collectiveeval.heldout_runner import main

if __name__ == "__main__":
    raise SystemExit(main())
