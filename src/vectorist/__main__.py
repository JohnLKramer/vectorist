def main() -> None:
    import uvicorn
    uvicorn.run("vectorist.api.app:create_app", factory=True, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
