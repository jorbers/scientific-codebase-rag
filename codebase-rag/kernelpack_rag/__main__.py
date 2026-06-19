"""CLI entry point: python -m kernelpack_rag <subcommand>"""

import sys


def main() -> None:
    subcommands = ("ingest", "reembed", "verify", "query", "mcp")

    if len(sys.argv) < 2 or sys.argv[1] not in subcommands:
        print(f"Usage: python -m kernelpack_rag [{' | '.join(subcommands)}] [args]")
        sys.exit(1)

    subcmd = sys.argv.pop(1)  # remove subcommand before handing argv to the module

    if subcmd == "ingest":
        from kernelpack_rag.ingest import main as _main
    elif subcmd == "reembed":
        from kernelpack_rag.reembed import main as _main
    elif subcmd == "verify":
        from kernelpack_rag.verify import main as _main
    elif subcmd == "query":
        from kernelpack_rag.query import main as _main
    elif subcmd == "mcp":
        from kernelpack_rag.mcp_server import main as _main

    _main()


if __name__ == "__main__":
    main()