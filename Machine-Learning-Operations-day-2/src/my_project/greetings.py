import typer

app = typer.Typer()

@app.command()
def count(name: str, repeat: int):
    for _ in range(repeat):
        print(f"Hello {name}")


if __name__ == "__main__":
    app()