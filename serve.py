"""Run the custom PneumoLens explorer and its local inference API."""
if __name__ == "__main__":
    print("Loading inference dependencies…", flush=True)
    from pneumolens.server import main

    main()
