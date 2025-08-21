# src/demo.py
import argparse
import os

def main():
    p = argparse.ArgumentParser(description="GeoSpaider RAG queries")
    p.add_argument("queries", nargs="*", help="Queries to run (you can pass several)")
    p.add_argument("--mode", choices=["dynamic", "pure"], default="pure",
                   help="dynamic=semantic top-k then min-cloud; pure=vector (full coll for cloud/recent)")
    p.add_argument("--top-n", type=int, default=2, help="How many results to show for each query")
    p.add_argument("--k", type=int, default=15, help="Vector pool size for semantic retrieval")

    # Defaults you want:
    # - Open images ON by default (can disable with --no-open-images)
    # - Descriptions OFF by default (can enable with --describe)
    p.add_argument("--open-images", action="store_true", default=True,
                   help="Open previews in default image viewer (default ON)")
    p.add_argument("--no-open-images", action="store_true",
                   help="Disable opening previews")
    p.add_argument("--no-describe", action="store_true", default=True,
                   help="Disable LLM description step (default ON)")
    p.add_argument("--describe", action="store_true",
                   help="Enable LLM description step")

    args = p.parse_args()

    # Resolve final switches
    open_images = False if args.no_open_images else True  # default True
    describe = True if args.describe else (not args.no_describe)  # default False

    # Ensure OPEN_IMAGES is set before importing rag (so it’s read at import-time)
    if open_images:
        os.environ["OPEN_IMAGES"] = "1"
    else:
        os.environ.pop("OPEN_IMAGES", None)

    from src.rag import run_query_vector_only, run_query_dynamic

    # Built-in defaults if none provided
    if not args.queries:
        args.queries = [
            "show the most recent image of Bhopal",
            "show images with lowest cloud cover",
            "give me the highest cloud cover image available",
            "show me two random images for diversity",
        ]

    for q in args.queries:
        if args.mode == "dynamic":
            run_query_dynamic(q, top_n=args.top_n, open_images=open_images)
        else:
            run_query_vector_only(
                q,
                top_n=args.top_n,
                describe=describe,
                k=args.k,
                open_images=open_images,
            )

if __name__ == "__main__":
    main()
