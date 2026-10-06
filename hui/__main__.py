"""``python -m hui`` runs the same CLI as the ``hui`` console script."""

from hui.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
